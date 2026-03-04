# 🔍 Multimodal RAG — Federated Learning Research Assistant

A production-ready **Multimodal Retrieval-Augmented Generation** system that answers questions about federated learning research papers by retrieving both **text passages** and **figures/diagrams** from scanned and digital PDFs.

---

## 🎯 What It Does

Traditional RAG systems only retrieve text. This system retrieves **both modalities**:

| Query | Text retrieval | Image retrieval |
|-------|---------------|-----------------|
| "How does FL aggregate model updates?" | Relevant paragraphs from papers | Flow charts, architecture diagrams |
| "What does the accuracy curve look like?" | Experimental sections | Accuracy vs. rounds graphs |
| "Show the system model" | System description text | Client-server topology figures |

---

## 🏗️ Architecture

```
PDFs (scanned + digital)
        │
        ▼
┌───────────────────┐
│   PDF Parser      │  PyMuPDF — detect digital vs scanned pages
│  digital → text   │
│  scanned → PNG    │
└────────┬──────────┘
         │
    ┌────┴─────┐
    ▼          ▼
┌────────┐  ┌────────┐
│  Text  │  │ Image  │
│Embedder│  │Embedder│
│MiniLM  │  │  CLIP  │
│L6-v2   │  │ViT-B/32│
└────┬───┘  └───┬────┘
     │           │
     ▼           ▼
┌────────┐  ┌────────┐
│ FAISS  │  │ FAISS  │
│  Text  │  │ Image  │
│ Index  │  │ Index  │
└────┬───┘  └───┬────┘
     │           │
     └─────┬─────┘
           ▼
┌─────────────────────┐
│ Multimodal Retriever│  weighted score merging
│  text_weight=0.6    │  + guaranteed image slots
│  image_weight=0.4   │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   LLM Generator     │  Ollama / Mistral-7B
│  (context-grounded) │  OpenAI fallback
└──────────┬──────────┘
           ▼
    ┌──────┴──────┐
    ▼             ▼
 FastAPI        Gradio
  /query         UI
```

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| PDF parsing | PyMuPDF | Fast, handles scanned + digital |
| Text embeddings | `all-MiniLM-L6-v2` | Fast 384-dim, strong semantic search |
| Image embeddings | `CLIP ViT-B/32` | Shared text-image latent space |
| Vector store | FAISS `IndexFlatIP` | Local, no API costs, exact cosine search |
| LLM | Mistral-7B via Ollama | Open-source, reproducible, no API key |
| API | FastAPI | Auto Swagger docs, Pydantic validation |
| UI | Gradio | Rapid demo, image gallery support |
| Evaluation | RAGAS | Research-backed retrieval metrics |

---

## 📁 Repository Structure

```
multimodal-rag/
├── src/
│   ├── embeddings/
│   │   ├── text_embedder.py      # sentence-transformers wrapper
│   │   └── image_embedder.py     # CLIP wrapper
│   ├── retrieval/
│   │   ├── vector_store.py       # FAISS wrapper (save/load)
│   │   └── retriever.py          # unified multimodal retriever
│   ├── generation/
│   │   └── llm.py                # Ollama / OpenAI / HuggingFace
│   └── pipeline/
│       └── rag_pipeline.py       # end-to-end pipeline
├── api/
│   ├── main.py                   # FastAPI app
│   └── schemas.py                # Pydantic models
├── ui/
│   └── app.py                    # Gradio UI
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_text_rag_baseline.ipynb
│   └── 03_multimodal_extension.ipynb
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── configs/
│   └── config.yaml               # all hyperparameters
└── experiments/                  # experiment logs
```

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/Naveenreddie-think/Multimodal_RAG.git
cd Multimodal_RAG
pip install -r requirements.txt
```

### 2. Add PDFs
```
data/raw/your_paper.pdf
```

### 3. Run notebooks in order
```
notebooks/01_data_exploration.ipynb   # analyze PDFs
notebooks/02_text_rag_baseline.ipynb  # build text index
notebooks/03_multimodal_extension.ipynb  # build image index
```

### 4. Start Ollama + pull model
```bash
ollama pull mistral
```

### 5. Run the API
```bash
uvicorn api.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### 6. Run the Gradio UI
```bash
python ui/app.py
# UI: http://localhost:7860
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| PDFs processed | 5 |
| Text chunks | 235 |
| Image chunks | 156 |
| Text embedding dim | 384 (MiniLM) |
| Image embedding dim | 512 (CLIP) |
| API latency (Ollama CPU) | ~9s |
| Retrieval mode | Multimodal (text + image) |

---

## 🔑 Key Design Decisions

**Why guaranteed image slots?**
CLIP image-text cosine similarity scores (0.1–0.35) are inherently lower than text-text cosine scores (0.3–0.7). Naive score merging always buries image results. We reserve `guaranteed_images=2` slots to ensure visual context is always surfaced.

**Why FAISS over Pinecone/ChromaDB?**
Local-first, no API costs, reproducible experiments. `IndexFlatIP` gives exact cosine search — sufficient for our corpus size.

**Why Ollama over HuggingFace pipeline?**
Ollama serves quantized models (4-bit GGUF) which run 10x faster on CPU than full-precision HuggingFace pipelines. Latency drops from 10+ minutes to ~9 seconds.

**What failed and what I learned:**
- CLIP's `get_image_features()` returns `BaseModelOutputWithPooling` in some versions — must call `vision_model()` + `visual_projection()` directly
- Score range mismatch between modalities requires either normalization or guaranteed slots
- Mistral-7B full precision on CPU is unusable for real-time API — quantized models via Ollama are the practical solution

---

## 🗺️ Roadmap

- [ ] OCR pipeline for fully scanned pages (Tesseract)
- [ ] W&B experiment tracking
- [ ] RAGAS evaluation with ground-truth QA pairs
- [ ] HuggingFace Spaces deployment
- [ ] Re-ranking with cross-encoder
- [ ] Support for tables as a third modality

---

## 📄 License

MIT
