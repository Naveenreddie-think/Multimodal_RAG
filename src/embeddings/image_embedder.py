# src/embeddings/image_embedder.py
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ImageEmbedder:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def embed_images(self, image_paths: list[str], batch_size: int = 16) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch = [Image.open(p).convert("RGB") for p in image_paths[i : i + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.vision_model(**inputs)
                feats = self.model.visual_projection(outputs.pooler_output)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeddings.append(feats.cpu().numpy())
        return np.vstack(all_embeddings).astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.text_model(**inputs)
            feats = self.model.text_projection(outputs.pooler_output)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")
