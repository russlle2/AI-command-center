"""
VLM (Visual Language Model) agent — uses Qwen2-VL-7B-Instruct.

Qwen2-VL-7B is one of the strongest open-source VLMs. It fits comfortably in
the RTX 5090's 24 GB VRAM in 4-bit quantization and supports:
  - Image description / captioning
  - OCR (reading text in images)
  - Visual Q&A (answering questions about an image)
  - Multi-image reasoning
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Union

from ai_command_center import config

logger = logging.getLogger(__name__)

_model = None
_processor = None


def _get_model():
    global _model, _processor
    if _model is None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "transformers>=4.45 and torch are required."
            ) from exc

        logger.info("Loading VLM model: %s …", config.VLM_MODEL_ID)

        _processor = AutoProcessor.from_pretrained(
            config.VLM_MODEL_ID,
            token = config.HF_TOKEN or None,
        )

        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.VLM_MODEL_ID,
            torch_dtype = "auto",
            device_map  = "auto",
            token       = config.HF_TOKEN or None,
        )
        logger.info("VLM ready.")
    return _model, _processor


class VLMAgent:
    """Answer questions about images using Qwen2-VL-7B-Instruct."""

    def run(
        self,
        prompt: str,
        image_path: Union[str, Path, None] = None,
        **kwargs,
    ) -> dict:
        """Analyse an image and answer the prompt.

        Args:
            prompt: The question or instruction (e.g. "Describe this image.").
            image_path: Path to the image file on disk (PNG/JPG/WEBP).
                        If None, the agent will try to extract a path from
                        the prompt string (format: "… [image: /path/to/file]").

        Returns:
            dict with keys:
                ``answer``     — the model's text answer
                ``image_path`` — path that was analysed (or None)
        """
        from PIL import Image

        # ── Resolve image path from prompt if not given explicitly ──────────
        if image_path is None:
            import re
            match = re.search(r"\[image:\s*(.+?)\]", prompt)
            if match:
                image_path = match.group(1).strip()
                prompt = re.sub(r"\[image:\s*.+?\]", "", prompt).strip()

        model, processor = _get_model()
        import torch

        if image_path:
            image = Image.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text": prompt},
                    ],
                }
            ]
        else:
            # Text-only fallback
            messages = [{"role": "user", "content": prompt}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs = [image] if image_path else []
        inputs = processor(
            text   = [text],
            images = image_inputs if image_inputs else None,
            padding = True,
            return_tensors = "pt",
        ).to(model.device)

        max_new_tokens = int(kwargs.get("max_new_tokens", 1024))
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids)
        ]
        answer = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        logger.info("VLM answer (first 200 chars): %s", answer[:200])
        return {"answer": answer, "image_path": str(image_path) if image_path else None}
