# src/retrieval/retriever.py
import numpy as np

from src.embeddings.image_embedder import ImageEmbedder
from src.embeddings.text_embedder import TextEmbedder
from src.retrieval.vector_store import VectorStore


class MultimodalRetriever:
    """
    Unified retriever over text (sentence-transformers) and image (CLIP) indices.

    Score ranges differ between modalities — text-text cosine is typically
    0.3–0.7, while CLIP image-text cosine is 0.1–0.35. We use guaranteed_images
    to ensure image results are always surfaced alongside text results.
    """

    def __init__(
        self,
        text_store: VectorStore,
        image_store: VectorStore,
        text_embedder: TextEmbedder,
        image_embedder: ImageEmbedder,
        text_weight: float = 0.6,
        image_weight: float = 0.4,
        guaranteed_images: int = 2,
    ):
        self.text_store = text_store
        self.image_store = image_store
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.guaranteed_images = guaranteed_images

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        text_k = top_k - self.guaranteed_images

        t_emb = self.text_embedder.embed_query(query)
        i_emb = self.image_embedder.embed_query(query)

        text_results  = self.text_store.search(t_emb, top_k=text_k)
        image_results = self.image_store.search(i_emb, top_k=self.guaranteed_images)

        for r in text_results:
            r["type"] = "text"
            r["weighted_score"] = r["score"] * self.text_weight

        for r in image_results:
            r["type"] = "image"
            r["weighted_score"] = r["score"] * self.image_weight

        merged = text_results + image_results
        merged.sort(key=lambda x: x["weighted_score"], reverse=True)
        return merged

    def retrieve_text_only(self, query: str, top_k: int = 5) -> list[dict]:
        emb = self.text_embedder.embed_query(query)
        results = self.text_store.search(emb, top_k=top_k)
        for r in results:
            r["type"] = "text"
        return results

    def retrieve_images_only(self, query: str, top_k: int = 3) -> list[dict]:
        emb = self.image_embedder.embed_query(query)
        results = self.image_store.search(emb, top_k=top_k)
        for r in results:
            r["type"] = "image"
        return results
