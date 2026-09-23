import time

from openai import (
    OpenAI,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)
# WHAT: OpenAI's Python client library.
# WHY: OpenRouter provides an OpenAI-compatible API.
#      We can use the same client by changing the base_url.
# PROJECT: Used to call OpenRouter's embedding endpoint.

from app.config import OPENROUTER_API_KEY, GEMINI_API_KEY, EMBEDDING_MODEL, EMBEDDING_PROVIDER


# ============================================================
# LOCAL BACKEND (sentence-transformers)
# ============================================================

_local_model = None
# WHAT: Module-level cache for the loaded model.
# WHY: Loading takes a few seconds. Load once, reuse for the whole run.


def _get_local_model():
    """
    Lazy-load the local embedding model.

    Why lazy → importing sentence_transformers is slow, and a run using
    the OpenRouter provider should never pay that cost.
    """
    global _local_model

    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading local embedding model: {EMBEDDING_MODEL}")
        _local_model = SentenceTransformer(EMBEDDING_MODEL)
        print("  Model loaded.")

    return _local_model


def _embed_local(texts):
    """
    Embed a list of strings on this machine. No network, no quota.
    """
    model = _get_local_model()

    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 50,
        # Only show the bar for big ingestions — it is noise for 2 chunks.
    )

    return vectors.tolist()
    # .tolist() → numpy array to plain Python lists.
    # WHY: ChromaDB and JSON expect lists, not numpy arrays.


# ============================================================
# OPENROUTER BACKEND
# ============================================================


_client = None
# WHAT: Module-level cache for the client.
# WHY: Building a client sets up a fresh HTTP connection pool. Creating one
#      per embed_text() call meant 1000 chunks built 1000 pools.
#      Build it once, reuse it for the whole run.


PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        # Google's OpenAI-compatible endpoint for Gemini models.
        "api_key": GEMINI_API_KEY,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_API_KEY,
    },
}
# WHAT: Where to send requests and which key to sign them with.
# WHY: The OpenAI class is just an HTTP client. Google, OpenRouter and Groq
#      all accept OpenAI's request format, so one client works for all of
#      them — only base_url and api_key differ. llm.py does the same for Groq.


def get_client():
    """
    Create (once) a client for the selected provider (Gemini or OpenRouter).
    """
    global _client

    if _client is None:
        provider = PROVIDERS[EMBEDDING_PROVIDER]

        if not provider["api_key"]:
            raise RuntimeError(
                f"No API key for EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r}. "
                f"Add it to .env."
            )

        _client = OpenAI(
            base_url=provider["base_url"],
            api_key=provider["api_key"],
        )

    return _client


class DailyQuotaExceeded(Exception):
    """
    Raised when OpenRouter's free-tier DAILY cap is hit.

    Why its own type → a per-minute rate limit is worth retrying a few
    seconds later, but a daily cap will not clear for hours. Retrying it
    just burns time. Callers can catch this and stop cleanly.
    """


def _is_daily_quota(err):
    """
    Distinguish 'too fast, slow down' from 'you are done for today'.

    OpenRouter reports the daily cap with limit_source=openrouter_free_tier_daily
    and a 'free-models-per-day' message.
    """
    text = str(err)
    return "free-models-per-day" in text or "openrouter_free_tier_daily" in text


def _call_with_retry(inputs, max_attempts=6):
    """
    POST one embedding request, retrying transient failures.

    Retries: per-minute rate limits, connection drops, timeouts.
    Does NOT retry: the daily quota cap (pointless) or bad requests.
    """
    client = get_client()
    delay = 2

    for attempt in range(1, max_attempts + 1):
        try:
            return client.embeddings.create(
                model=EMBEDDING_MODEL,
                # PROJECT: nvidia/nemotron-3-embed-1b:free

                input=inputs,
                # WHAT: Either a single string or a list of strings.
                # WHY: The OpenAI embeddings spec accepts both; sending a list
                #      returns every vector in ONE request instead of N.

                encoding_format="float",
                # WHAT: Requests plain JSON floats.
                # WHY: The SDK defaults to "base64" whenever numpy is installed.
                #      Nemotron on OpenRouter rejects base64 with a 400.
                #      Removing this line brings that 400 straight back.
            )

        except RateLimitError as err:
            if _is_daily_quota(err):
                raise DailyQuotaExceeded(
                    "OpenRouter free-tier daily limit reached. "
                    "Wait for the reset or add credits."
                ) from err
            # Otherwise it is a short-term rate limit → back off and retry.
            if attempt == max_attempts:
                raise
            print(f"  Rate limited, retrying in {delay}s "
                  f"(attempt {attempt}/{max_attempts})...")
            time.sleep(delay)
            delay *= 2
            # WHY exponential → each retry waits longer, giving the limit
            #     window time to roll over instead of hammering the API.

        except (APIConnectionError, APITimeoutError) as err:
            if attempt == max_attempts:
                raise
            print(f"  Connection problem ({type(err).__name__}), "
                  f"retrying in {delay}s (attempt {attempt}/{max_attempts})...")
            time.sleep(delay)
            delay *= 2


def embed_text(text):
    """
    Convert one piece of text into an embedding vector.

    Routes to whichever backend EMBEDDING_PROVIDER selects.
    """
    if EMBEDDING_PROVIDER == "local":
        return _embed_local([text])[0]

    response = _call_with_retry(text)
    return response.data[0].embedding


_batch_supported = None
# WHAT: Tri-state cache — None = untested, True = works, False = rejected.
# WHY: We could not verify batch support (the daily cap blocked the test
#      request), so the code finds out at runtime instead of assuming.
#      Learned once per process, not re-probed on every call.


def embed_texts(texts, batch_size=32):
    """
    Convert multiple texts into embedding vectors.

    Tries to send them in batches (one request per batch). If the API
    rejects a list input, permanently falls back to one request per text.

    Returns: list of vectors, SAME ORDER as `texts`.
    """
    global _batch_supported

    if not texts:
        return []

    if EMBEDDING_PROVIDER == "local":
        print(f"  Embedding {len(texts)} texts locally...")
        vectors = _embed_local(texts)
        print(f"  Embedded {len(texts)} texts (local).          ")
        return vectors

    embeddings = []

    if _batch_supported is not False:
        try:
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                print(f"  Embedding {start + len(batch)}/{len(texts)} "
                      f"(batched)...", end="\r")

                response = _call_with_retry(batch)

                ordered = sorted(response.data, key=lambda d: d.index or 0)
                # WHY sort → the spec says each item carries its .index, and
                #     nothing guarantees the response preserves input order.
                #     A vector paired with the wrong chunk is a silent,
                #     nearly untraceable retrieval bug.
                # WHY "or 0" → Gemini omits index on the FIRST item (Google
                #     drops zero-valued fields), so it arrives as None.
                #     Comparing None with 1 raises TypeError, which used to
                #     be misread as "batching unsupported".

                embeddings.extend(item.embedding for item in ordered)

            _batch_supported = True
            print(f"  Embedded {len(texts)} texts (batched).          ")
            return embeddings

        except (DailyQuotaExceeded, RateLimitError):
            raise
            # Quota is quota — falling back to single calls cannot help.
            # WHY RateLimitError too → a per-minute cap is not "batching is
            #     unsupported". Falling back to one request per text makes
            #     the rate limit WORSE (198 requests instead of 7).

        except Exception as err:
            print(f"  Batch input rejected ({type(err).__name__}), "
                  f"falling back to one request per text.")
            _batch_supported = False
            embeddings = []

    # Fallback path: one request per text.
    for i, text in enumerate(texts):
        print(f"  Embedding {i + 1}/{len(texts)}...", end="\r")
        embeddings.append(embed_text(text))

    print(f"  Embedded {len(texts)} texts.          ")
    return embeddings


def embed_documents(docs):
    """
    Convert document chunks into embedding vectors.

    Our loaders and the chunker all return Document objects
    (see app/ingestion/document.py), so we read the .text attribute.
    """
    texts = [doc.text for doc in docs]
    # WHAT: Extract the actual text from every Document/chunk.
    # WHY: The embedding API accepts strings, not our Document objects.

    print(f"  Embedding {len(texts)} chunks via {EMBEDDING_PROVIDER}...")
    embeddings = embed_texts(texts)

    if embeddings:
        print(f"  Done. Each vector: {len(embeddings[0])} dimensions.")

    return embeddings
