# src/retrieval/vector_store.py
import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    """Thin FAISS wrapper — supports separate text and image indices."""

    def __init__(self, dim: int, index_type: str = "flat_ip"):
        self.dim = dim
        if index_type == "flat_ip":
            self.index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError(f"Unsupported index_type: {index_type}")
        self.metadata: list[dict] = []

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        assert embeddings.shape[1] == self.dim, "Embedding dim mismatch"
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            result = dict(self.metadata[idx])
            result["score"] = round(float(score), 4)
            results.append(result)
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        path = Path(path)
        index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "metadata.json") as f:
            metadata = json.load(f)
        store = cls.__new__(cls)
        store.dim = index.d
        store.index = index
        store.metadata = metadata
        return store

    def __len__(self) -> int:
        return self.index.ntotal
