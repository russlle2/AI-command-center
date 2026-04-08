"""
AI Command Center — configuration.

All user-specific settings are loaded from environment variables or a local
`.env` file so that no secrets are ever committed to source control.

Copy `.env.example` to `.env` and fill in your values:
    cp .env.example .env
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────
_env_file = Path(__file__).parent.parent / ".env"
load_dotenv(_env_file, override=False)


# ── HuggingFace ────────────────────────────────────────────────────────────
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# ── GitHub (optional — for the system agent to push code) ──────────────────
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")

# ── Main orchestrator model (GGUF loaded via llama-cpp-python) ─────────────
ORCHESTRATOR_MODEL_PATH: str = os.getenv(
    "ORCHESTRATOR_MODEL_PATH",
    r"C:\AI-models\llama31-70b-uncensored-Q4_K_M.gguf",
)
ORCHESTRATOR_N_GPU_LAYERS: int = int(os.getenv("ORCHESTRATOR_N_GPU_LAYERS", "28"))
ORCHESTRATOR_N_CTX: int = int(os.getenv("ORCHESTRATOR_N_CTX", "4096"))
ORCHESTRATOR_N_THREADS: int = int(os.getenv("ORCHESTRATOR_N_THREADS", "24"))
ORCHESTRATOR_TEMPERATURE: float = float(os.getenv("ORCHESTRATOR_TEMPERATURE", "0.7"))
ORCHESTRATOR_MAX_TOKENS: int = int(os.getenv("ORCHESTRATOR_MAX_TOKENS", "2048"))

# ── Text-to-image (FLUX.1-dev via diffusers) ───────────────────────────────
T2I_MODEL_ID: str = os.getenv("T2I_MODEL_ID", "black-forest-labs/FLUX.1-dev")
T2I_DEVICE: str = os.getenv("T2I_DEVICE", "cuda")

# ── Text-to-video (CogVideoX-5b via diffusers) ────────────────────────────
T2V_MODEL_ID: str = os.getenv("T2V_MODEL_ID", "THUDM/CogVideoX-5b")
T2V_DEVICE: str = os.getenv("T2V_DEVICE", "cuda")
T2V_NUM_FRAMES: int = int(os.getenv("T2V_NUM_FRAMES", "49"))
T2V_FPS: int = int(os.getenv("T2V_FPS", "8"))

# ── VLM (Qwen2-VL-7B-Instruct via transformers) ───────────────────────────
VLM_MODEL_ID: str = os.getenv("VLM_MODEL_ID", "Qwen/Qwen2-VL-7B-Instruct")
VLM_DEVICE: str = os.getenv("VLM_DEVICE", "cuda")

# ── Code / reasoning model (DeepSeek-R1-8B via llama-cpp) ─────────────────
CODE_MODEL_PATH: str = os.getenv(
    "CODE_MODEL_PATH",
    r"C:\AI-models\DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf",
)
CODE_N_GPU_LAYERS: int = int(os.getenv("CODE_N_GPU_LAYERS", "99"))

# ── RAG index ──────────────────────────────────────────────────────────────
RAG_DOCS_DIR: str = os.getenv(
    "RAG_DOCS_DIR",
    r"C:\Users\Administrator\Documents",
)
RAG_INDEX_DIR: str = os.getenv(
    "RAG_INDEX_DIR",
    r"C:\Users\Administrator\Documents\rag_index",
)
RAG_EMBEDDING_MODEL: str = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "sentence-transformers/all-mpnet-base-v2",
)
RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "512"))
RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "6"))

# ── Output directories ─────────────────────────────────────────────────────
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", r"C:\AI-command-center\outputs")
IMAGE_OUTPUT_DIR: str = os.path.join(OUTPUT_DIR, "images")
VIDEO_OUTPUT_DIR: str = os.path.join(OUTPUT_DIR, "videos")

# ── UI ─────────────────────────────────────────────────────────────────────
GRADIO_PORT: int = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_SHARE: bool = os.getenv("GRADIO_SHARE", "false").lower() == "true"
