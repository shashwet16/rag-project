import os
from dotenv import load_dotenv

load_dotenv()
# Loads values from .env into Python's environment.
# This keeps secret API keys out of the source code.


# ============================================================
# API KEYS
# ============================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# OpenRouter key → used to access the embedding API.

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Groq key → used later to generate the final RAG answers.

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ============================================================
# MODELS
# ============================================================

LLM_MODEL = "gemini-3.6-flash"
# Gemini generates the final answer (see app/generation/llm.py).
#
# NOTE: gemini-1.5-flash was retired and gemini-2.5-flash is closed to new
# users — both 404 on this key. Google recommends gemini-3.6-flash.
# Check what you can actually use with:
#     client.models.list()


EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")
# Which backend produces our embeddings: "gemini", "openrouter" or "local".
#
# "gemini"     → Google gemini-embedding-001. Free tier, generous quota.
# "openrouter" → NVIDIA Nemotron. Stronger vectors, but the :free tier
#                allows 50 requests/day. Batching keeps a full ingest to
#                ~1 request, so that is rarely a problem now.
# "local"      → sentence-transformers on this machine. No key, no quota.
#                Use it if the daily cap ever blocks you mid-run.
#
# Switch for one run without editing this file:
#     EMBEDDING_PROVIDER=local python3 test_rag.py


if EMBEDDING_PROVIDER == "local":
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    # Small, fast local model (~80MB, cached after first download).

    EMBEDDING_DIMENSION = 384

elif EMBEDDING_PROVIDER == "gemini":
    EMBEDDING_MODEL = "gemini-embedding-2"
    # Google's embedding model, served via its OpenAI-compatible endpoint.
    # Available on this key: gemini-embedding-001, gemini-embedding-2,
    # gemini-embedding-2-preview. There is NO "-002".

    EMBEDDING_DIMENSION = 3072
    # Default output size for gemini-embedding-002.

else:
    EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b:free"
    # OpenRouter runs NVIDIA's embedding model.
    # ":free" selects the free OpenRouter endpoint.

    EMBEDDING_DIMENSION = 2048
    # Measured from the live API. It is NOT 4096 — a wrong value here does
    # not error immediately; Chroma pins the collection to whatever the
    # first vector's real length is, so the mismatch surfaces later as
    # confusing dimension errors.


# ============================================================
# CHUNKING
# ============================================================

CHUNK_SIZE = 1000
# Approximate amount of text we put into one chunk.

CHUNK_OVERLAP = 200
# Amount of text shared between neighboring chunks.
# This helps prevent useful context from being cut at chunk boundaries.


# ============================================================
# RETRIEVAL
# ============================================================

TOP_K = 10
# Number of relevant chunks we retrieve from ChromaDB
# when answering a user's question.


# ============================================================
# CHROMADB SETTINGS
# ============================================================

CHROMA_PERSIST_DIR = "./chroma_db"
# Local directory where ChromaDB stores the vector database.

COLLECTION_NAME = f"documents_{EMBEDDING_PROVIDER}"
# Name of the ChromaDB collection containing our embeddings.
#
# WHY the provider suffix → OpenRouter vectors are 2048-dim and local ones
# are 384-dim. A Chroma collection has ONE fixed dimension, so sharing a
# single name across providers would error out on the second one.