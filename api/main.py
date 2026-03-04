# api/main.py
"""
FastAPI app for the Multimodal RAG pipeline.

Run locally:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health       — readiness check
    POST /query        — main RAG query
    GET  /docs         — auto-generated Swagger UI
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse, QueryRequest, QueryResponse, SourceItem
from src.pipeline.rag_pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Global pipeline instance — loaded once at startup
# ---------------------------------------------------------------------------
pipeline: RAGPipeline | None = None
CONFIG_PATH = Path("configs/config.yaml")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline on startup, clean up on shutdown."""
    global pipeline
    print("🔄 Loading RAG pipeline...")
    try:
        pipeline = RAGPipeline.from_config(CONFIG_PATH)
        print("✅ Pipeline ready")
    except Exception as e:
        print(f"❌ Pipeline failed to load: {e}")
        raise
    yield
    print("👋 Shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Multimodal RAG API",
    description="RAG over federated learning papers — text + image retrieval via CLIP & sentence-transformers",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    """Readiness check — confirms pipeline is loaded and returns index sizes."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    return HealthResponse(
        status="ok",
        text_chunks=len(pipeline.retriever.text_store),
        image_chunks=len(pipeline.retriever.image_store),
        device=pipeline.retriever.image_embedder.device,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(request: QueryRequest):
    """
    Run a multimodal RAG query.

    - **question**: natural language question about the documents
    - **top_k**: number of results to retrieve (default 5)
    - **mode**: multimodal | text_only | images_only
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        if request.mode == "text_only":
            results = pipeline.retriever.retrieve_text_only(request.question, top_k=request.top_k)
            answer  = pipeline.generator.generate(
                question=request.question,
                context=pipeline._build_context(results),
            )
            response_dict = {
                "question":   request.question,
                "answer":     answer,
                "sources":    pipeline._format_sources(results),
                "latency_ms": 0.0,
            }
        elif request.mode == "images_only":
            results = pipeline.retriever.retrieve_images_only(request.question, top_k=request.top_k)
            response_dict = {
                "question":   request.question,
                "answer":     "Image retrieval only — no answer generated.",
                "sources":    pipeline._format_sources(results),
                "latency_ms": 0.0,
            }
        else:
            response_dict = pipeline.query(request.question)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        question=response_dict["question"],
        answer=response_dict["answer"],
        sources=[SourceItem(**s) for s in response_dict["sources"]],
        latency_ms=response_dict["latency_ms"],
    )
