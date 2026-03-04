# src/embeddings/text_embedder.py
import numpy as np
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings / norms).astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])
