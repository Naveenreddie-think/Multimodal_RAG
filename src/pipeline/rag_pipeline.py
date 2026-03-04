# src/pipeline/rag_pipeline.py
"""
End-to-end Multimodal RAG Pipeline.

Usage:
    pipeline = RAGPipeline.from_config("configs/config.yaml")
    response = pipeline.query("How does federated learning aggregate model updates?")
    print(response["answer"])
    print(response["sources"])
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from src.embeddings.image_embedder import ImageEmbedder
from src.embeddings.text_embedder import TextEmbedder
from src.generation.llm import LLMGenerator
from src.retrieval.retriever import MultimodalRetriever
from src.retrieval.vector_store import VectorStore


class RAGPipeline:
    def __init__(
        self,
        retriever: MultimodalRetriever,
        generator: LLMGenerator,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Factory — build the full pipeline from config.yaml
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config_path: str | Path = "configs/config.yaml") -> "RAGPipeline":
        config_path = Path(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)

        base_dir   = config_path.parent.parent
        vector_dir = base_dir / config.get("vector_dir", "data/processed/vectors")
        text_dir   = vector_dir / "text"
        image_dir  = vector_dir / "image"

        # Load vector stores
        text_store  = VectorStore.load(text_dir)
        image_store = VectorStore.load(image_dir)

        # Load embedders
        text_embedder  = TextEmbedder(model_name=config.get("embed_model", "all-MiniLM-L6-v2"))
        image_embedder = ImageEmbedder(model_name=config.get("clip_model", "openai/clip-vit-base-patch32"))

        # Build retriever
        retriever = MultimodalRetriever(
            text_store=text_store,
            image_store=image_store,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            text_weight=config.get("text_weight", 0.6),
            image_weight=config.get("image_weight", 0.4),
            guaranteed_images=config.get("guaranteed_images", 2),
        )

        # Build generator
        generator = LLMGenerator(
            model_name=config.get("llm_model", "mistral"),
            temperature=config.get("temperature", 0.2),
            max_new_tokens=config.get("max_new_tokens", 512),
            use_openai=config.get("use_openai", False),
            use_ollama=config.get("use_ollama", True),
        )

        return cls(retriever=retriever, generator=generator, top_k=config.get("top_k", 5))

    # ------------------------------------------------------------------
    # Core query method
    # ------------------------------------------------------------------
    def query(self, question: str) -> dict[str, Any]:
        """
        Run end-to-end RAG: retrieve → build prompt → generate → return.

        Returns:
            {
                "question":   str,
                "answer":     str,
                "sources":    list[dict],
                "latency_ms": float
            }
        """
        t0 = time.perf_counter()

        # 1. Retrieve
        results = self.retriever.retrieve(question, top_k=self.top_k)

        # 2. Build context string for the LLM
        context = self._build_context(results)

        # 3. Generate answer
        answer = self.generator.generate(question=question, context=context)

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "question":   question,
            "answer":     answer,
            "sources":    self._format_sources(results),
            "latency_ms": latency_ms,
        }

    def retrieve_only(self, question: str) -> list[dict]:
        """Return raw retrieval results without generation — useful for evaluation."""
        return self.retriever.retrieve(question, top_k=self.top_k)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_context(self, results: list[dict]) -> str:
        parts = []
        for i, r in enumerate(results):
            if r["type"] == "text":
                parts.append(
                    f"[Source {i+1} — {r['source']} p.{r['page']+1}]\n{r.get('text', '')}"
                )
            else:
                parts.append(
                    f"[Source {i+1} — {r['source']} p.{r['page']+1}] "
                    f"<image: {r.get('image_path', '')}>"
                )
        return "\n\n".join(parts)

    def _format_sources(self, results: list[dict]) -> list[dict]:
        return [
            {
                "type":   r["type"],
                "source": r["source"],
                "page":   r["page"],
                "score":  r.get("weighted_score", r.get("score", 0)),
                **({"text_preview": r["text"][:200]} if r["type"] == "text" else
                   {"image_path": r.get("image_path", "")}),
            }
            for r in results
        ]
