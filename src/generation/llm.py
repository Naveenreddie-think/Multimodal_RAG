# src/generation/llm.py
"""
LLM wrapper for answer generation.

Backends (set via config.yaml):
  - Ollama      : use_ollama: true  — fast local inference, recommended for CPU
  - OpenAI      : use_openai: true  — requires OPENAI_API_KEY in .env
  - HuggingFace : default           — heavy, best with GPU

Docker note:
  When running in Docker, Ollama runs as a separate container.
  Set OLLAMA_HOST=http://ollama:11434 in docker-compose environment.
  Locally it defaults to http://localhost:11434.
"""
from __future__ import annotations

import os


class LLMGenerator:
    def __init__(
        self,
        model_name: str = "mistral",
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        use_openai: bool = False,
        use_ollama: bool = True,
    ):
        self.model_name     = model_name
        self.temperature    = temperature
        self.max_new_tokens = max_new_tokens
        self.use_openai     = use_openai
        self.use_ollama     = use_ollama
        self._pipeline      = None  # lazy load for HuggingFace

        # Ollama host — overridden by env var in Docker
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # ------------------------------------------------------------------
    # Prompt template
    # ------------------------------------------------------------------
    PROMPT_TEMPLATE = """You are a research assistant specialising in federated learning.
Answer the question below using ONLY the provided context.
If the context does not contain enough information, say "I don't have enough context to answer."
Be concise and precise. Do not hallucinate.

Context:
{context}

Question: {question}

Answer:"""

    # ------------------------------------------------------------------
    # Generate — routes to correct backend
    # ------------------------------------------------------------------
    def generate(self, question: str, context: str) -> str:
        prompt = self.PROMPT_TEMPLATE.format(context=context, question=question)
        if self.use_openai:
            return self._generate_openai(prompt)
        if self.use_ollama:
            return self._generate_ollama(prompt)
        return self._generate_hf(prompt)

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------
    def _generate_ollama(self, prompt: str) -> str:
        import requests
        print(f"🔄 Querying Ollama ({self.ollama_host}): {self.model_name}")
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_new_tokens,
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Ollama is not running at {self.ollama_host}.\n"
                "Local: run 'ollama serve'\n"
                "Docker: ensure the ollama container is up"
            )

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------
    def _generate_openai(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # HuggingFace local (best with GPU, slow on CPU)
    # ------------------------------------------------------------------
    def _generate_hf(self, prompt: str) -> str:
        if self._pipeline is None:
            self._pipeline = self._load_hf_pipeline()
        output = self._pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            return_full_text=False,
        )
        return output[0]["generated_text"].strip()

    def _load_hf_pipeline(self):
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        print(f"🔄 Loading LLM: {self.model_name} (device={device})")
        return pipeline(
            "text-generation",
            model=self.model_name,
            device=device,
            dtype="auto",
        )
