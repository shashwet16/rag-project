import shutil
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.rag_pipeline import ask
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.web_loader import load_url
from app.ingestion.yt_loader import load_youtube
from app.processing.cleaner import clean_documents
from app.processing.chunker import chunk_documents
from app.vectorstore.embedder import embed_documents
from app.vectorstore.store import store_documents

# WHAT: Creates the FastAPI application instance.
# WHY:  Everything — routes, middleware, docs — hangs off this object.
# ALTERNATIVE: Flask's app = Flask(__name__) does the same, but without
#              auto validation, async support, or auto-generated docs.
app = FastAPI(
    title="Multi-Source RAG Research Assistant",
    description="Ingest PDFs, URLs, YouTube → ask questions → get grounded answers.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def root():
    return FileResponse("app/static/index.html")

# ── Pydantic models ────────────────────────────────────────
# WHAT: Define the exact shape of incoming request data.
# WHY:  FastAPI validates every request against these models automatically.
#       Wrong type or missing field → 422 error, your code never runs.

class URLRequest(BaseModel):
    url: str

class YoutubeRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    # WHAT: top_k has a default of 5.
    # WHY:  Makes it optional — clients don't have to send it.


# ── Helper: ingest any document list ──────────────────────
def ingest_documents(docs):
    """
    Shared ingestion logic used by all three ingest endpoints.
    WHAT: Clean → chunk → embed → store in ChromaDB (same steps as ingest.py).
    WHY:  All three source types (PDF, URL, YouTube) produce the same
          Document format after loading, so ingestion is identical.
    """
    if not docs:
        raise HTTPException(status_code=400, detail="No content extracted.")

    docs = clean_documents(docs)
    chunks = chunk_documents(docs)
    # WHAT: chunks are Document objects, not dicts — embed_documents and
    #       store_documents read .text / .metadata / .doc_id themselves.
    embeddings = embed_documents(chunks)
    store_documents(chunks, embeddings)
    return len(chunks)


# ── Endpoints ──────────────────────────────────────────────

@app.get("/health")
def health():
    """
    WHAT: Simple liveness check.
    WHY:  Load balancers, monitoring tools, and deployment platforms
          ping this to check if the server is alive.
    """
    return {"status": "ok"}


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    WHAT: Accept a PDF file upload, ingest into ChromaDB.
    WHY:  UploadFile lets clients send binary files over HTTP (multipart form).
    ALTERNATIVE: Accept a file path string instead of upload — simpler but
                 only works if client and server share the same filesystem.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")

    # Save upload to a temp file because load_pdf needs a file path
    # WHAT: Temp file is auto-deleted when the `with` block exits.
    # WHY:  We don't want to permanently store uploaded files on the server.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        docs = load_pdf(tmp_path)
        # Override source metadata to use original filename, not temp path
        for doc in docs:
            doc.metadata["source"] = file.filename
        count = ingest_documents(docs)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        # WHAT: Delete the temp file after ingestion.
        # WHY:  Temp files accumulate. Always clean up.

    return {"message": f"Ingested {count} chunks from {file.filename}"}


@app.post("/ingest/url")
def ingest_url(request: URLRequest):
    """
    WHAT: Accept a URL, scrape content, ingest into ChromaDB.
    """
    try:
        docs = load_url(request.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load URL: {e}")

    count = ingest_documents(docs)
    return {"message": f"Ingested {count} chunks from {request.url}"}


@app.post("/ingest/youtube")
def ingest_youtube(request: YoutubeRequest):
    """
    WHAT: Accept a YouTube URL, fetch transcript, ingest into ChromaDB.
    """
    try:
        docs = load_youtube(request.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load YouTube: {e}")

    count = ingest_documents(docs)
    return {"message": f"Ingested {count} chunks from {request.url}"}


@app.post("/query")
def query(request: QueryRequest):
    """
    WHAT: Accept a question, run RAG pipeline, return answer + sources.
    WHY:  This is the core endpoint — the entire pipeline we built so far
          is exposed through this single function call.
    """
    result = ask(request.question, top_k=request.top_k)
    # WHAT: ask() returns {"answer": ..., "sources": [...], "query": ...}
    # WHY:  FastAPI auto-serializes Python dicts to JSON — no extra work.
    return result