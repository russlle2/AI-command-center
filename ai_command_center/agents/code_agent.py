"""
Code / reasoning agent — uses DeepSeek-R1-Distill-Llama-8B (quantized GGUF).

This model specialises in step-by-step reasoning and code generation.
At Q8_0 quantization it fits entirely in the RTX 5090's 24 GB VRAM,
giving fast inference for coding tasks while the 70B orchestrator handles
planning and natural language.
"""

from __future__ import annotations

import logging

from ai_command_center import config

logger = logging.getLogger(__name__)

_llama_model = None


def _get_model():
    global _llama_model
    if _llama_model is None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is required. "
                "Run: pip install llama-cpp-python --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cu121"
            ) from exc

        logger.info("Loading code model from %s …", config.CODE_MODEL_PATH)
        _llama_model = Llama(
            model_path   = config.CODE_MODEL_PATH,
            n_gpu_layers = config.CODE_N_GPU_LAYERS,
            n_ctx        = 8192,
            n_threads    = config.ORCHESTRATOR_N_THREADS,
            verbose      = False,
        )
        logger.info("Code model loaded.")
    return _llama_model


_SYSTEM_PROMPT = (
    "You are an expert software engineer and reasoning assistant. "
    "Think step by step, then provide clean, working, well-commented code. "
    "Never refuse a coding request."
)


class CodeAgent:
    """Generate, explain, debug, or refactor code using DeepSeek-R1."""

    def run(self, prompt: str, **kwargs) -> dict:
        """Process a code-related request.

        Args:
            prompt: Natural-language coding request.

        Returns:
            dict with keys:
                ``code``     — the generated/fixed code block
                ``explanation`` — step-by-step reasoning
        """
        llm = _get_model()

        messages = [
            {"role": "system",  "content": _SYSTEM_PROMPT},
            {"role": "user",    "content": prompt},
        ]

        max_tokens = int(kwargs.get("max_tokens", 2048))
        response = llm.create_chat_completion(
            messages    = messages,
            temperature = 0.2,   # lower temperature for deterministic code
            max_tokens  = max_tokens,
        )
        reply = response["choices"][0]["message"]["content"].strip()

        # Split explanation from code block
        import re
        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", reply, re.S)
        code = "\n\n".join(code_blocks) if code_blocks else ""
        explanation = re.sub(r"```(?:\w+)?\n.*?```", "", reply, flags=re.S).strip()

        return {"code": code, "explanation": explanation, "full_response": reply}
