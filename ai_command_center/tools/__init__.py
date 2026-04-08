"""
Tool registry for the AI Command Center.

Each tool maps a name to a schema (used for LLM function-calling prompts)
and an executor function.  The orchestrator references these definitions
when building the system prompt so the LLM knows exactly what tools
are available, what arguments they accept, and what they return.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

_registry: dict[str, "ToolDef"] = {}


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]
    executor: Callable[..., Any] | None = None

    def to_schema(self) -> dict:
        """Return an OpenAI-style function schema for LLM prompting."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def register_tool(tool: ToolDef) -> None:
    _registry[tool.name] = tool


def get_tool(name: str) -> ToolDef | None:
    return _registry.get(name)


def all_tools() -> list[ToolDef]:
    return list(_registry.values())


def all_schemas() -> list[dict]:
    return [t.to_schema() for t in _registry.values()]


# ── Built-in tool definitions ──────────────────────────────────────────────

register_tool(ToolDef(
    name="text_to_image",
    description="Generate a photorealistic or artistic image from a text prompt using FLUX.1-dev.",
    parameters={
        "type": "object",
        "properties": {
            "prompt":  {"type": "string", "description": "Detailed description of the image to generate."},
            "width":   {"type": "integer", "default": 1024},
            "height":  {"type": "integer", "default": 1024},
            "num_inference_steps": {"type": "integer", "default": 28},
            "seed":    {"type": "integer", "description": "Optional seed for reproducibility."},
        },
        "required": ["prompt"],
    },
))

register_tool(ToolDef(
    name="text_to_video",
    description="Generate a short video clip from a text prompt using CogVideoX-5b.",
    parameters={
        "type": "object",
        "properties": {
            "prompt":     {"type": "string", "description": "Description of the video to generate."},
            "num_frames": {"type": "integer", "default": 49},
            "fps":        {"type": "integer", "default": 8},
            "seed":       {"type": "integer", "description": "Optional seed for reproducibility."},
        },
        "required": ["prompt"],
    },
))

register_tool(ToolDef(
    name="vlm",
    description="Analyse an image using Qwen2-VL-7B: describe contents, OCR text, answer visual questions.",
    parameters={
        "type": "object",
        "properties": {
            "prompt":     {"type": "string", "description": "Question or instruction about the image."},
            "image_path": {"type": "string", "description": "Path to the image file on disk."},
        },
        "required": ["prompt"],
    },
))

register_tool(ToolDef(
    name="system",
    description="Execute shell commands, install applications, download files, or manage the local system.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to execute or action to perform (e.g. 'install ffmpeg', 'run: dir C:\\\\')."},
        },
        "required": ["command"],
    },
))

register_tool(ToolDef(
    name="code",
    description="Write, debug, fix, refactor, or explain code using DeepSeek-R1-8B.",
    parameters={
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "The coding request (e.g. 'Write a Python function that …')."},
        },
        "required": ["prompt"],
    },
))

register_tool(ToolDef(
    name="rag",
    description="Semantic search over local documents (PDF, DOCX, TXT, MD, HTML, CSV) in the RAG index.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "top_k": {"type": "integer", "default": 6},
        },
        "required": ["query"],
    },
))
