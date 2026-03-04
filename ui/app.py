from __future__ import annotations
import sys
import shutil
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from src.pipeline.rag_pipeline import RAGPipeline

# Resolve paths
BASE_DIR  = Path(__file__).parent.parent
IMAGE_DIR = BASE_DIR / "data" / "processed" / "images"
TMP_DIR   = tempfile.gettempdir()

print("Loading RAG pipeline...")
pipeline = RAGPipeline.from_config(BASE_DIR / "configs" / "config.yaml")
print("Pipeline ready")


def resolve_image(raw_path: str) -> str | None:
    """
    image_chunks.json stores relative paths like '..\\data\\processed\\images\\Sample_3\\page_010.png'
    Resolve them to absolute paths so Gradio can serve them.
    """
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return str(p)
    # Try resolving relative to BASE_DIR
    candidate = (BASE_DIR / p).resolve()
    if candidate.exists():
        return str(candidate)
    # Try resolving the relative path directly
    candidate2 = p.resolve()
    if candidate2.exists():
        return str(candidate2)
    # Search by filename inside IMAGE_DIR
    matches = list(IMAGE_DIR.rglob(p.name))
    if matches:
        return str(matches[0])
    return None


def copy_to_tmp(abs_path: str) -> str:
    """Copy image to temp dir so Gradio can serve it."""
    dst = Path(TMP_DIR) / Path(abs_path).name
    shutil.copy(abs_path, str(dst))
    return str(dst)


def format_sources(sources):
    if not sources:
        return "No sources retrieved."
    rows = []
    for i, s in enumerate(sources):
        stype   = "Text" if s["type"] == "text" else "Image"
        preview = s.get("text_preview", "")[:80] if s.get("text_preview") else s.get("image_path", "")[-30:]
        rows.append(f"| {i+1} | {stype} | {s['source']} | p.{s['page']+1} | {s['score']:.4f} | {preview} |")
    header = "| # | Type | Source | Page | Score | Preview |\n|---|------|--------|------|-------|---------|"
    return header + "\n" + "\n".join(rows)


def query(question, top_k, mode):
    if not question.strip():
        return "Please enter a question.", "", "No sources", []
    try:
        if mode == "Multimodal":
            result = pipeline.query(question)
        elif mode == "Text only":
            results = pipeline.retriever.retrieve_text_only(question, top_k=top_k)
            answer  = pipeline.generator.generate(
                question=question,
                context=pipeline._build_context(results)
            )
            result = {"question": question, "answer": answer,
                      "sources": pipeline._format_sources(results), "latency_ms": 0}
        else:
            results = pipeline.retriever.retrieve_images_only(question, top_k=top_k)
            result  = {"question": question, "answer": "Image retrieval only.",
                       "sources": pipeline._format_sources(results), "latency_ms": 0}

        # Resolve and copy images to temp dir for Gradio
        images = []
        for s in result["sources"]:
            if s["type"] == "image" and s.get("image_path"):
                abs_path = resolve_image(s["image_path"])
                if abs_path:
                    images.append(copy_to_tmp(abs_path))

        return (
            result["answer"],
            f"⏱ {result['latency_ms']}ms",
            format_sources(result["sources"]),
            images,
        )

    except Exception as e:
        return f"Error: {e}", "", "No sources", []


EXAMPLES = [
    ["How does federated learning aggregate model updates?", 5, "Multimodal"],
    ["What are the privacy challenges in federated learning?", 5, "Text only"],
    ["Show me diagrams of federated learning architecture", 3, "Images only"],
    ["What datasets are used in the experiments?", 5, "Multimodal"],
]

with gr.Blocks(title="Multimodal RAG") as demo:
    gr.Markdown(
        "# Multimodal RAG — Federated Learning Research Assistant\n"
        "Ask questions about federated learning papers. "
        "Retrieves both **text passages** and **figures/diagrams**."
    )

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Your Question",
                placeholder="e.g. How does federated learning aggregate model updates?",
                lines=2,
            )
        with gr.Column(scale=1):
            top_k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top-K results")
            mode_radio   = gr.Radio(
                choices=["Multimodal", "Text only", "Images only"],
                value="Multimodal",
                label="Retrieval mode",
            )

    submit_btn = gr.Button("Ask", variant="primary")

    gr.Markdown("### Answer")
    answer_box  = gr.Textbox(label="", lines=6, interactive=False)
    latency_box = gr.Textbox(label="Latency", interactive=False)

    gr.Markdown("### Retrieved Sources")
    sources_box = gr.Markdown()

    gr.Markdown("### Retrieved Images")
    gallery = gr.Gallery(label="", show_label=False, columns=3, height=400)

    gr.Examples(examples=EXAMPLES, inputs=[question_box, top_k_slider, mode_radio])

    submit_btn.click(
        fn=query,
        inputs=[question_box, top_k_slider, mode_radio],
        outputs=[answer_box, latency_box, sources_box, gallery],
    )
    question_box.submit(
        fn=query,
        inputs=[question_box, top_k_slider, mode_radio],
        outputs=[answer_box, latency_box, sources_box, gallery],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        allowed_paths=[str(IMAGE_DIR), TMP_DIR],
    )