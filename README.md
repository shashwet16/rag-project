# RAG Project

[](LICENSE)

A multi-source Retrieval-Augmented Generation (RAG) pipeline: ingest PDFs, web pages, and YouTube transcripts, chunk and embed them into ChromaDB, then ask grounded questions through a FastAPI service or the CLI.

Answers are generated **only** from retrieved context — the system is instructed to say it doesn't know rather than guess, and every answer comes back with its source citations.

## How it works

```
PDF / URL / YouTube
        │
        ▼
   Loader (app/ingestion/)      → raw text as Document objects
        │
        ▼
   Cleaner (app/processing/cleaner.py)   → normalize unicode, whitespace, dedupe lines
        │
        ▼
   Chunker (app/processing/chunker.py)   → recursive character splitting + deterministic chunk IDs
        │
        ▼
   Embedder (app/vectorstore/embedder.py) → Gemini / OpenRouter / local sentence-transformers
        │
        ▼
   ChromaDB (app/vectorstore/store.py)    → persistent vector store, cosine similarity
        │
        ▼
   Retriever (app/retrieval/retriever.py) → top-K search + distance threshold filter
        │
        ▼
   LLM (app/generation/llm.py)            → answer grounded in retrieved chunks, with citations
```

`app/rag_pipeline.py` wires retrieval + generation together behind a single `ask(query)` call, and `app/api/main.py` exposes the whole pipeline over HTTP with FastAPI.

## Project layout

```
app/
├── config.py               # central config: models, chunk size, top-K, API keys
├── rag_pipeline.py         # ask() / format_response() — the end-to-end pipeline
├── ingestion/
│   ├── document.py         # Document — the shared data type every loader returns
│   ├── pdf_loader.py       # load_pdf(path) — one Document per page (pdfplumber)
│   ├── web_loader.py       # load_url(url)  — scrape + strip boilerplate (BeautifulSoup)
│   └── yt_loader.py        # load_youtube(url) — pull the video transcript
├── processing/
│   ├── cleaner.py          # clean_text/clean_documents — unicode + whitespace + dedupe
│   └── chunker.py          # chunk_documents — recursive splitting, deterministic chunk IDs
├── vectorstore/
│   ├── embedder.py         # embed_text/embed_texts/embed_documents — Gemini/OpenRouter/local
│   └── store.py            # ChromaDB client, upsert, similarity search
├── retrieval/
│   └── retriever.py        # retrieve() — embed query, search, filter by relevance threshold
├── generation/
│   └── llm.py              # generate_answer() — Gemini chat completion with a grounding prompt
├── api/
│   └── main.py              # FastAPI app: /health, /ingest/pdf, /ingest/url, /ingest/youtube, /query
└── static/
    └── index.html           # Web UI served at "/" (ingest + ask, no build step)

ingest.py                    # CLI: python3 ingest.py <path-to-pdf>
data/                        # sample PDF for local testing
chroma_db/                   # persisted vector store (gitignored, created on first ingest)
test_*.py                    # manual smoke-test scripts (see Testing below)
```

## Setup

**Requirements:** Python 3.10+ (a virtualenv is recommended — the project was developed against one at `venv/`).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy the env template and add your keys:

```bash
cp .env.example .env
```

You only need the keys your configuration actually calls — see `app/config.py`:

| Variable | Used for |
|---|---|
| `GEMINI_API_KEY` | Default embedding backend (`EMBEDDING_PROVIDER=gemini`) and default LLM generation (`app/generation/llm.py`) |
| `OPENROUTER_API_KEY` | Optional embedding backend (`EMBEDDING_PROVIDER=openrouter`, NVIDIA Nemotron) |
| `GROQ_API_KEY` | Reserved for an alternate LLM backend (not wired in by default — `llm.py` currently calls Gemini) |

Set `EMBEDDING_PROVIDER=local` (in `.env` or inline) to use `sentence-transformers` on your own machine with no API key and no quota:

```bash
EMBEDDING_PROVIDER=local python3 ingest.py data/sample.pdf
```

> **Note on free-tier quotas:** Gemini's free tier caps chat models at ~20 requests/day *per model*. If `/query` starts returning `429 RESOURCE_EXHAUSTED`, either wait for the daily reset, switch `LLM_MODEL` in `app/config.py` to a different Gemini model, or point `app/generation/llm.py` at another OpenAI-compatible provider (e.g. Groq) using the key you've already got in `.env`.

## Usage

### 1. Ingest documents

CLI (PDF only):

```bash
python3 ingest.py data/sample.pdf
```

Or via the API (see below) for PDFs, URLs, and YouTube videos.

### 2. Ask questions

```bash
python3 -c "
from app.rag_pipeline import ask, format_response
print(format_response(ask('What is the investment multiplier?')))
"
```

### 3. Run the API server

```bash
source venv/bin/activate
uvicorn app.api.main:app --reload
```

> Make sure `uvicorn` resolves to the one in your venv (`venv/bin/uvicorn` or `python -m uvicorn ...`) — a globally installed `uvicorn` on your `PATH` will run against system Python and miss the project's dependencies.

Open `http://127.0.0.1:8000/` for a built-in web UI (ingest sources and ask questions from the browser), or `http://127.0.0.1:8000/docs` for interactive Swagger docs. You can also call the API directly:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"}'

curl -X POST http://127.0.0.1:8000/ingest/youtube \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}'

curl -X POST http://127.0.0.1:8000/ingest/pdf \
  -F "file=@data/sample.pdf"

curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is reinforcement learning?", "top_k": 5}'
```

## Configuration

All tunables live in `app/config.py`:

| Setting | Default | What it controls |
|---|---|---|
| `EMBEDDING_PROVIDER` | `gemini` | `gemini`, `openrouter`, or `local` |
| `LLM_MODEL` | `gemini-3.6-flash` | Chat model used for generation |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Character-based chunking via `RecursiveCharacterTextSplitter` |
| `TOP_K` | `10` | Chunks retrieved per query |
| `RELEVANCE_THRESHOLD` (in `app/retrieval/retriever.py`) | `0.7` | Max cosine distance for a chunk to count as relevant |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Where the vector store is written to disk |
| `COLLECTION_NAME` | `documents_<provider>` | Namespaced per embedding provider, since each has a different vector dimension |

## Testing

The `test_*.py` files at the project root are manual smoke-test/inspection scripts (they print output rather than assert), used while building each stage of the pipeline — not a pytest suite. Some (`test_document.py`, `test_chunker.py`, `test_pdf.py`) reference an absolute local PDF path from development; point them at your own PDF (e.g. `data/sample.pdf`) before running.

Useful ones to try first:

```bash
python3 test_setup.py        # confirms config loads correctly
python3 test_web.py          # loads a Wikipedia page end-to-end
python3 test_idontknow.py    # checks in-scope questions are answered and out-of-scope ones are refused
```

`tests/` is a placeholder package for a future pytest suite.

## Known limitations

- `app/ingestion/web_loader.py` validates the URL scheme against `('https', 'https')` — this currently rejects plain `http://` URLs.
- `GROQ_API_KEY` is read into config but not yet wired into `app/generation/llm.py`, which always calls Gemini.
- Retrieval's relevance threshold (`0.7`) is looser than the actual distance gap observed between in-scope and out-of-scope queries in testing (~0.24–0.33 vs. ~0.39–0.49) — tightening it to ~0.36–0.38 would let empty retrieval (rather than the LLM) catch out-of-scope questions.

## License

[MIT](LICENSE)
