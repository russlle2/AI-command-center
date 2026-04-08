# AI Command Center — Local Installation Guide

> Platform: **Windows 11**, RTX 5090 (24 GB VRAM), Intel Core Ultra 9 285K,
> 64 GB RAM, 5 TB NVMe SSD

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Step 1 — Fine-tune the model on Colab](#step-1--fine-tune-the-model-on-colab)
3. [Step 2 — Download the GGUF model](#step-2--download-the-gguf-model)
4. [Step 3 — Install llama.cpp (CUDA build)](#step-3--install-llamacpp-cuda-build)
5. [Step 4 — Clone this repo & create Python env](#step-4--clone-this-repo--create-python-env)
6. [Step 5 — Install Python dependencies](#step-5--install-python-dependencies)
7. [Step 6 — Configure environment variables](#step-6--configure-environment-variables)
8. [Step 7 — Build the RAG index](#step-7--build-the-rag-index)
9. [Step 8 — Download specialist model files](#step-8--download-specialist-model-files)
10. [Step 9 — Launch the AI Command Center](#step-9--launch-the-ai-command-center)
11. [Optional — Run the orchestrator as a background service](#optional--run-the-orchestrator-as-a-background-service)
12. [GPU memory allocation strategy](#gpu-memory-allocation-strategy)
13. [Troubleshooting](#troubleshooting)

---

## 1. Prerequisites

| Software | Version | Notes |
|---|---|---|
| Windows 11 | 23H2+ | |
| NVIDIA Driver | ≥ 576 | RTX 5090 requires driver 576+ |
| CUDA Toolkit | 12.1 | https://developer.nvidia.com/cuda-downloads |
| Python | 3.11 | Use the official installer; add to PATH |
| Git | Latest | https://git-scm.com/download/win |
| winget / chocolatey | Latest | For app installation via SystemAgent |

---

## Step 1 — Fine-tune the model on Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload **`notebooks/01_qlora_finetuning.ipynb`** (File → Upload notebook)
3. Set runtime to **H100 GPU** (Runtime → Change runtime type → H100).  
   For a 70B QLoRA job, **H100 80 GB** is the practical choice on Colab; smaller GPUs (T4, L4, A100 40 GB) will **not** reliably fit this workload.  
   If you use **Kaggle** (often two H100s), the notebook’s FSDP auto-detects multiple GPUs and uses both.
4. **(Recommended)** Store your Hugging Face token in Colab **Secrets** (Runtime → Secrets), name it `HF_TOKEN`, grant access to the notebook session.  
   Do **not** paste tokens into chat or commit them to git; the notebooks read `HF_TOKEN` from the environment or Secrets.
5. Run cells top-to-bottom. If you did not set Secrets, you will be prompted for the token.
6. Training takes ~2–3 hours on H100 80 GB.  Checkpoints auto-save every 500 steps to:
   - Colab: `/content/drive/MyDrive/AI-command-center/checkpoints/`
   - Windows G: drive: `G:\AI-command-center\checkpoints\`
7. After training, open **`notebooks/02_quantize_export.ipynb`** and run it to export the GGUF.

**What the notebook does (FSDP + QLoRA stack):**
- **NF4 4-bit** quantization via `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`
- **FSDP** via `accelerate` config (`FULL_SHARD`, `TRANSFORMER_BASED_WRAP` on `LlamaDecoderLayer`)
- **Gradient checkpointing** with `use_reentrant=False` (required for FSDP)
- `per_device_train_batch_size=1` + `gradient_accumulation_steps=16`
- **Flash Attention 2** for H100-native speed
- Saves every **500 steps** to Google Drive (G: drive on Windows)

> **No H100?**  Use one of these alternatives:
> - [Kaggle](https://kaggle.com) — free 2× H100 × 30 h per week (FSDP will auto-use both GPUs)
> - [Lambda Labs](https://lambdalabs.com) — ~$2/h for H100
> - [RunPod](https://runpod.io) — ~$2–3/h for A100 80 GB

---

## Step 2 — Download the GGUF model

After the Colab notebook finishes, download the GGUF file to your PC:

```powershell
# Install huggingface-cli if not already installed
pip install huggingface_hub[cli]

# Download Q4_K_M (recommended — best speed/quality for 24 GB VRAM + 64 GB RAM)
huggingface-cli download YOUR-USERNAME/llama31-70b-uncensored-GGUF `
    llama31-70b-uncensored-Q4_K_M.gguf `
    --local-dir C:\AI-models\
```

Also download the DeepSeek-R1 code model (free, no gating):

```powershell
huggingface-cli download bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF `
    DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf `
    --local-dir C:\AI-models\
```

---

## Step 3 — Install llama.cpp (CUDA build)

`llama-cpp-python` embeds llama.cpp and exposes a Python API.  We need the
CUDA-enabled wheel so the RTX 5090 is used.

```powershell
# In your Python venv (see Step 4 first):
pip install llama-cpp-python `
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

Verify GPU detection:

```python
from llama_cpp import Llama
m = Llama(model_path=r"C:\AI-models\llama31-70b-uncensored-Q4_K_M.gguf",
          n_gpu_layers=28, n_ctx=512, verbose=True)
print(m("Hello"))
```

You should see lines like `ggml_cuda_init: found 1 CUDA devices`.

---

## Step 4 — Clone this repo & create Python env

```powershell
cd C:\
git clone https://github.com/russlle2/AI-command-center.git
cd AI-command-center

# Create isolated Python 3.11 virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## Step 5 — Install Python dependencies

```powershell
# Core dependencies
pip install -r requirements.txt

# llama-cpp-python with CUDA (must be done separately)
pip install llama-cpp-python `
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

> **Note on FLUX.1-dev (text-to-image):** FLUX requires a HuggingFace token
> and your account must accept the model license at
> https://huggingface.co/black-forest-labs/FLUX.1-dev.  The model weights
> (~24 GB) are auto-downloaded on first use.

---

## Step 6 — Configure environment variables

```powershell
# Copy the example .env file
Copy-Item .env.example .env

# Edit with your favourite editor
notepad .env
```

Fill in:
- `HF_TOKEN` — your HuggingFace read/write token
- `GITHUB_TOKEN` — optional, for the system agent to push code
- `ORCHESTRATOR_MODEL_PATH` — path to your GGUF file (e.g. `C:\AI-models\llama31-70b-uncensored-Q4_K_M.gguf`)
- `CODE_MODEL_PATH` — path to DeepSeek-R1 GGUF
- All other paths as needed (the defaults work if you followed the guide)

---

## Step 7 — Build the RAG index

Your Google Drive is already synced to `C:\Users\Administrator\Documents`.
Run the indexer to make all those files searchable:

```powershell
python -c "
from ai_command_center.rag.rag_index import RAGIndex
idx = RAGIndex()
stats = idx.build()
print(stats)
"
```

This will:
1. Scan `C:\Users\Administrator\Documents` recursively
2. Parse PDF, DOCX, TXT, MD, HTML, CSV files
3. Split into chunks and embed with `all-mpnet-base-v2`
4. Store in ChromaDB at `C:\Users\Administrator\Documents\rag_index`

Subsequent runs are incremental — only changed files are re-indexed.

---

## Step 8 — Download specialist model files

The specialist models are downloaded automatically on first use (via
HuggingFace Hub), but you can pre-download them to avoid delays:

```powershell
# VLM — Qwen2-VL-7B-Instruct (~16 GB)
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir C:\AI-models\Qwen2-VL-7B

# Text-to-Image — FLUX.1-dev (~24 GB, requires HF token & license acceptance)
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir C:\AI-models\FLUX1-dev

# Text-to-Video — CogVideoX-5b (~20 GB)
huggingface-cli download THUDM/CogVideoX-5b --local-dir C:\AI-models\CogVideoX-5b
```

> **GPU memory note:** These models are loaded lazily (only when you use
> that tab) and unloaded between uses.  See the [GPU memory strategy](#gpu-memory-allocation-strategy) section below.

---

## Step 9 — Launch the AI Command Center

```powershell
cd C:\AI-command-center
.\.venv\Scripts\Activate.ps1
python -m ai_command_center.main
```

A Gradio UI will open automatically at http://localhost:7860

---

## Optional — Run the orchestrator as a background service

To have the AI Command Center start automatically with Windows:

1. Create `C:\AI-command-center\start.bat`:
```batch
@echo off
cd /d C:\AI-command-center
call .venv\Scripts\activate.bat
python -m ai_command_center.main
```

2. Press `Win+R` → `shell:startup` → create a shortcut to `start.bat`.

Or use [NSSM](https://nssm.cc/) to register it as a Windows Service:
```powershell
nssm install AICommandCenter "C:\AI-command-center\start.bat"
nssm set AICommandCenter AppDirectory "C:\AI-command-center"
nssm start AICommandCenter
```

---

## GPU memory allocation strategy

With 24 GB VRAM (RTX 5090) and 64 GB RAM, the system uses a time-sharing
strategy — only one heavy model is loaded at a time:

| Component | VRAM | RAM (CPU) | Notes |
|---|---|---|---|
| Llama-3.1-70B Q4_K_M (orchestrator) | ~22 GB | ~20 GB | 28/80 layers on GPU |
| DeepSeek-R1-8B Q8_0 (code) | ~9 GB | 0 | Fits entirely on GPU |
| FLUX.1-dev (t2i) | ~24 GB | ~8 GB | Loaded on demand |
| CogVideoX-5b (t2v) | ~20 GB | ~10 GB | Loaded on demand |
| Qwen2-VL-7B (vlm) | ~16 GB | ~4 GB | Loaded on demand |

**Recommendation:** Keep the orchestrator and code model always loaded.
Diffusion models (FLUX, CogVideoX) are large — if you want faster image/video
generation, close the orchestrator before generating, or add more VRAM via
a second GPU in NVLink.

---

## Troubleshooting

### `CUDA out of memory`
- Reduce `ORCHESTRATOR_N_GPU_LAYERS` in `.env` (try 20 instead of 28)
- Set `T2I_DEVICE=cpu` or `T2V_DEVICE=cpu` to offload those models to RAM

### `ImportError: llama_cpp`
- Make sure you installed the CUDA wheel: `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121`
- Verify CUDA 12.1 is installed: `nvcc --version`

### Fine-tuning crashes on Colab with OOM
- Use H100 80 GB (not T4 or V100) — 70B requires ≥ 40 GB VRAM with NF4 4-bit
- Reduce `MAX_SEQ_LENGTH` to 1024 in the configuration cell
- The notebook already uses `per_device_train_batch_size=1` (minimum possible)
- If still OOM, set `GRAD_ACCUM = 8` and `LORA_R = 8` in the config cell
- The FSDP config cell sets `fsdp_cpu_ram_efficient_loading: false` intentionally —
  bitsandbytes NF4 quantization requires CUDA and cannot run on CPU; each rank loads
  a full 4-bit copy (~35 GB) of the base model to its own GPU

### RAG index is empty / finds nothing
- Check that `RAG_DOCS_DIR` points to the right folder
- Run `python -c "from ai_command_center.rag.rag_index import RAGIndex; print(RAGIndex().stats())"` to see what was indexed
- Make sure documents are in a supported format: PDF, DOCX, TXT, MD, HTML, CSV

### Model refuses to answer (still censored after fine-tuning)
- Increase `EPOCHS` in the Colab notebook (try 2–3 epochs)
- Check the dataset format — make sure the formatter in Step 7 of the notebook matched the correct columns
- Increase `LORA_R` to 32 or 64 for more adapter capacity
