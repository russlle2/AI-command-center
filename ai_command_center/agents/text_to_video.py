"""
Text-to-Video agent — uses CogVideoX-5b.

CogVideoX-5b is one of the best open-source text-to-video models available.
It generates 49-frame (≈6 s) 720×480 videos.  With the RTX 5090 it runs in
fp16 with CPU offloading for the VAE decode step.
"""

from __future__ import annotations

import logging
import os
import time

from ai_command_center import config

logger = logging.getLogger(__name__)

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            import torch
            from diffusers import CogVideoXPipeline
            from diffusers.utils import export_to_video
        except ImportError as exc:
            raise ImportError(
                "diffusers>=0.30 and torch are required. "
                "Run: pip install 'diffusers>=0.30' torch"
            ) from exc

        logger.info("Loading text-to-video model: %s …", config.T2V_MODEL_ID)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        _pipeline = CogVideoXPipeline.from_pretrained(
            config.T2V_MODEL_ID,
            torch_dtype = dtype,
            token       = config.HF_TOKEN or None,
        )

        if torch.cuda.is_available():
            _pipeline.enable_model_cpu_offload()
            _pipeline.vae.enable_slicing()   # reduces peak VRAM during VAE
            _pipeline.vae.enable_tiling()
        logger.info("Text-to-video pipeline ready.")
    return _pipeline


class TextToVideoAgent:
    """Generate short videos from text prompts using CogVideoX-5b."""

    def run(self, prompt: str, **kwargs) -> dict:
        """Generate a video.

        Args:
            prompt: Description of the video to generate.
            **kwargs: Optional overrides — num_frames, fps, guidance_scale,
                      num_inference_steps, seed.

        Returns:
            dict with keys:
                ``path``   — absolute path to the saved MP4 file
                ``prompt`` — the prompt used
        """
        from diffusers.utils import export_to_video

        pipe = _get_pipeline()

        num_frames = int(kwargs.get("num_frames",        config.T2V_NUM_FRAMES))
        fps        = int(kwargs.get("fps",               config.T2V_FPS))
        steps      = int(kwargs.get("num_inference_steps", 50))
        guidance   = float(kwargs.get("guidance_scale",  6.0))
        seed       = kwargs.get("seed", None)

        import torch
        generator = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else None

        logger.info("Generating video for prompt: %s …", prompt[:80])
        frames = pipe(
            prompt               = prompt,
            num_frames           = num_frames,
            guidance_scale       = guidance,
            num_inference_steps  = steps,
            generator            = generator,
        ).frames[0]

        os.makedirs(config.VIDEO_OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(
            config.VIDEO_OUTPUT_DIR,
            f"video_{int(time.time())}.mp4",
        )
        export_to_video(frames, out_path, fps=fps)
        logger.info("Video saved to %s", out_path)

        return {"path": out_path, "prompt": prompt, "frames": len(frames), "fps": fps}
