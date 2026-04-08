# 🤖 AI Command Center

A fully local, uncensored multi-agent AI system built around a
**QLoRA fine-tuned Llama-3.1-70B** orchestrator with specialist agents for
text-to-image, text-to-video, vision, coding, and system automation —
all plugged into a RAG index over your local documents.

## Architecture

```
User prompt
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Orchestrator (Llama-3.1-70B-Uncensored GGUF)   │  ← planner / router
│  + RAG context from Google Drive documents       │
└──────────┬──────────────────────────────────────┘
           │  routes to specialist agent
    ┌──────┼────────────────────────────────────────────┐
    │      │                                            │
    ▼      ▼           ▼             ▼         ▼        ▼
  Chat  Text-to-   Text-to-    Vision (VLM)  Code   System
        Image      Video       Qwen2-VL-7B   Agent  Agent
       (FLUX.1)  (CogVideoX)               DeepSeek (shell/
                                            -R1-8B   install)
```

## Quick Start

```
1. Fine-tune on Google Colab H100  →  notebooks/01_qlora_finetuning.ipynb
2. Quantize & export GGUF          →  notebooks/02_quantize_export.ipynb
3. Install locally                 →  INSTALLATION.md
4. Launch UI                       →  python -m ai_command_center.main
```

## Fine-tuning stack (Colab H100)

| Component | Implementation |
|---|---|
| 4-bit quantization | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)` |
| FSDP | `accelerate` config — `FULL_SHARD`, `TRANSFORMER_BASED_WRAP` on `LlamaDecoderLayer` |
| Gradient checkpointing | `prepare_model_for_kbit_training(..., use_gradient_checkpointing=True)` with `use_reentrant=False` |
| Micro-batch | `per_device_train_batch_size=1`, `gradient_accumulation_steps=16` |
| Checkpoints | Every **500 steps** → Google Drive (G:\\) |
| Attention | Flash Attention 2 (H100 native) |

## Models Used

| Role | Model | VRAM |
|---|---|---|
| Orchestrator / Chat | Llama-3.1-70B-Instruct (QLoRA fine-tuned, Q4_K_M GGUF) | ~22 GB |
| Code / Reasoning | DeepSeek-R1-Distill-Llama-8B (Q8_0 GGUF) | ~9 GB |
| Text-to-Image | FLUX.1-dev | ~24 GB |
| Text-to-Video | CogVideoX-5b | ~20 GB |
| Vision (VLM) | Qwen2-VL-7B-Instruct | ~16 GB |

## Features

- 💬 **Uncensored chat** — fine-tuned on ZombitX64/UncensoredOssbit
- 🎨 **Text-to-image** — FLUX.1-dev via Diffusers
- 🎬 **Text-to-video** — CogVideoX-5b via Diffusers
- 👁️ **Vision Q&A** — Qwen2-VL-7B answers questions about images
- 🖥️ **System agent** — run shell commands, install apps, download files
- 📚 **RAG index** — semantic search over 48 GB+ of Google Drive documents
- 🔗 **Chain-of-command** — the orchestrator decomposes complex tasks and
  executes multiple agent calls sequentially

## Hardware Requirements (local inference)

| Component | Minimum | Tested |
|---|---|---|
| GPU | RTX 3090 24 GB | RTX 5090 24 GB |
| RAM | 32 GB | 64 GB |
| Storage | 200 GB free | 5 TB NVMe |
| OS | Windows 10/11 or Linux | Windows 11 |

## Repository Structure

```
AI-command-center/
├── notebooks/
│   ├── 01_qlora_finetuning.ipynb     # Google Colab H100: QLoRA+FSDP training
│   └── 02_quantize_export.ipynb      # Google Colab: GGUF export & upload
├── ai_command_center/
│   ├── config.py                     # Central configuration (env vars)
│   ├── orchestrator.py               # Main planner / router
│   ├── main.py                       # Gradio UI entry point
│   ├── agents/
│   │   ├── text_to_image.py          # FLUX.1-dev image generation
│   │   ├── text_to_video.py          # CogVideoX-5b video generation
│   │   ├── vlm_agent.py              # Qwen2-VL-7B vision Q&A
│   │   ├── system_agent.py           # Shell/install/download
│   │   └── code_agent.py             # DeepSeek-R1 code generation
│   └── rag/
│       └── rag_index.py              # ChromaDB RAG over local documents
├── .env.example                      # Environment variable template
├── requirements.txt
├── INSTALLATION.md                   # Full setup guide
└── README.md
```

## Setup

See **[INSTALLATION.md](INSTALLATION.md)** for the full step-by-step guide.

```powershell
# Quick start (after fine-tuning and downloading models)
git clone https://github.com/russlle2/AI-command-center.git
cd AI-command-center
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
Copy-Item .env.example .env   # then edit .env with your paths and tokens
python -m ai_command_center.main
```
