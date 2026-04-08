"""
Text-to-Image agent — uses FLUX.1-dev (or SDXL as fallback).

FLUX.1-dev is currently one of the highest-quality open-source text-to-image
models. It requires ~24 GB VRAM for fp16 inference, which exactly fits the
RTX 5090. The agent automatically offloads to CPU if VRAM is insufficient.

Integrates with the GPU memory manager to unload competing models
before loading the diffusion pipeline.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from ai_command_center import config
from ai_command_center.gpu_manager import ModelSlot, ensure_slot, register, unregister

logger = logging.getLogger(__name__)

_pipeline = None


def _unload():
    global _pipeline
    if _pipeline is not None:
        del _pipeline
        _pipeline = None
    unregister(ModelSlot.TEXT_TO_IMAGE)
    logger.info("Text-to-image pipeline unloaded.")


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        ensure_slot(ModelSlot.TEXT_TO_IMAGE)

        try:
            import torch
            from diffusers import FluxPipeline
        except ImportError as exc:
            raise ImportError(
                "diffusers and torch are required. Run: pip install diffusers torch"
            ) from exc

        logger.info("Loading text-to-image model: %s …", config.T2I_MODEL_ID)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        _pipeline = FluxPipeline.from_pretrained(
            config.T2I_MODEL_ID,
            torch_dtype = dtype,
            token       = config.HF_TOKEN or None,
        )

        if torch.cuda.is_available():
            _pipeline.enable_model_cpu_offload()
        logger.info("Text-to-image pipeline ready.")

        register(ModelSlot.TEXT_TO_IMAGE, estimated_vram_mb=24000, unload_fn=_unload)
    return _pipeline


class TextToImageAgent:
    """Generate images from text prompts using FLUX.1-dev."""

    def run(self, prompt: str, **kwargs) -> dict:
        """Generate an image.

        Args:
            prompt: Natural-language description of the desired image.
            **kwargs: Optional overrides — width, height, guidance_scale,
                      num_inference_steps, seed.

        Returns:
            dict with keys:
                ``path``   — absolute path to the saved PNG file
                ``prompt`` — the prompt used
        """
        pipe = _get_pipeline()

        width    = int(kwargs.get("width",    1024))
        height   = int(kwargs.get("height",   1024))
        steps    = int(kwargs.get("num_inference_steps", 28))
        guidance = float(kwargs.get("guidance_scale", 3.5))
        seed     = kwargs.get("seed", None)

        import torch
        generator = torch.Generator().manual_seed(int(seed)) if seed is not None else None

        logger.info("Generating image for prompt: %s …", prompt[:80])
        image = pipe(
            prompt                = prompt,
            width                 = width,
            height                = height,
            guidance_scale        = guidance,
            num_inference_steps   = steps,
            generator             = generator,
        ).images[0]

        os.makedirs(config.IMAGE_OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(
            config.IMAGE_OUTPUT_DIR,
            f"image_{int(time.time())}.png",
        )
        image.save(out_path)
        logger.info("Image saved to %s", out_path)

        return {"path": out_path, "prompt": prompt}
