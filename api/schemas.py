# api/schemas.py
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, example="How does federated learning aggregate model updates?")
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="multimodal", pattern="^(multimodal|text_only|images_only)$")


class SourceItem(BaseModel):
    type: str           # "text" or "image"
    source: str         # PDF filename
    page: int
    score: float
    text_preview: str | None = None
    image_path: str | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    text_chunks: int
    image_chunks: int
    device: str
