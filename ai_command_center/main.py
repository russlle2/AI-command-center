"""
AI Command Center — Gradio web UI entry point.

Launch with:
    python -m ai_command_center.main
or:
    python ai_command_center/main.py

The UI provides:
  - Multi-turn chat with the orchestrator (chain-of-command execution)
  - Image generation tab (text -> FLUX image)
  - Video generation tab (text -> CogVideoX video)
  - VLM tab (image + question -> answer)
  - System tab (run shell commands, install software)
  - RAG management tab (build / query the document index)
  - GPU status dashboard (VRAM usage, loaded models)
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers = [logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from ai_command_center import config
from ai_command_center.orchestrator import Orchestrator
from ai_command_center.rag.rag_index import RAGIndex

rag_index   : RAGIndex    | None = None
orchestrator: Orchestrator | None = None


def _init():
    global rag_index, orchestrator
    if orchestrator is not None:
        return

    logger.info("Initialising RAG index …")
    rag_index = RAGIndex()

    logger.info("Initialising Orchestrator …")
    orchestrator = Orchestrator(rag_index=rag_index)


def build_ui():
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError("gradio is required. Run: pip install gradio") from exc

    _init()

    # ── Chat tab ──────────────────────────────────────────────────────────
    def chat_fn(message: str, history: list):
        hist_dicts = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": m}
            for i, (m, _) in enumerate(history)
        ] if history else []
        reply = orchestrator.chat(message, history=hist_dicts)
        return reply

    # ── Image tab ─────────────────────────────────────────────────────────
    def generate_image_fn(prompt: str, width: int, height: int, steps: int, seed: str):
        from ai_command_center.agents.text_to_image import TextToImageAgent
        agent = TextToImageAgent()
        kw = {"width": width, "height": height, "num_inference_steps": steps}
        if seed.strip():
            kw["seed"] = int(seed)
        result = agent.run(prompt, **kw)
        return result["path"]

    # ── Video tab ─────────────────────────────────────────────────────────
    def generate_video_fn(prompt: str, num_frames: int, fps: int, seed: str):
        from ai_command_center.agents.text_to_video import TextToVideoAgent
        agent = TextToVideoAgent()
        kw = {"num_frames": num_frames, "fps": fps}
        if seed.strip():
            kw["seed"] = int(seed)
        result = agent.run(prompt, **kw)
        return result["path"]

    # ── VLM tab ───────────────────────────────────────────────────────────
    def vlm_fn(image, question: str):
        import tempfile
        from ai_command_center.agents.vlm_agent import VLMAgent
        agent = VLMAgent()
        if image is not None:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                from PIL import Image as PILImage
                PILImage.fromarray(image).save(tmp.name)
                tmp_path = tmp.name
            result = agent.run(question, image_path=tmp_path)
        else:
            result = agent.run(question)
        return result["answer"]

    # ── System tab ────────────────────────────────────────────────────────
    def system_fn(command: str):
        from ai_command_center.agents.system_agent import SystemAgent
        agent = SystemAgent()
        result = agent.run(command)
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        rc     = result.get("returncode", "")
        out = ""
        if stdout:
            out += f"STDOUT:\n{stdout}\n\n"
        if stderr:
            out += f"STDERR:\n{stderr}\n\n"
        out += f"Return code: {rc}"
        return out.strip()

    # ── RAG tab ───────────────────────────────────────────────────────────
    def rag_build_fn(force: bool):
        stats = rag_index.build(force=force)
        return (
            f"Index built!\n"
            f"  Indexed : {stats['indexed']} files\n"
            f"  Skipped : {stats['skipped']} files\n"
            f"  Errors  : {stats['errors']} files"
        )

    def rag_query_fn(query: str, top_k: int):
        docs = rag_index.query(query, top_k=top_k)
        if not docs:
            return "No results found."
        parts = []
        for i, d in enumerate(docs, 1):
            parts.append(
                f"[{i}] Score: {d['score']:.3f} | Source: {d['source']}\n{d['text'][:400]}"
            )
        return "\n\n---\n\n".join(parts)

    def rag_stats_fn():
        s = rag_index.stats()
        return (
            f"Total chunks  : {s['total_chunks']:,}\n"
            f"Indexed files : {s['indexed_files']:,}\n"
            f"Docs dir      : {s['docs_dir']}\n"
            f"Index dir     : {s['index_dir']}"
        )

    # ── GPU dashboard ─────────────────────────────────────────────────────
    def gpu_status_fn():
        from ai_command_center.gpu_manager import status
        s = status()
        lines = [
            f"VRAM Total : {s['vram_total_mb']:,} MB",
            f"VRAM Used  : {s['vram_used_mb']:,} MB",
            f"VRAM Free  : {s['vram_free_mb']:,} MB",
            "",
            "Loaded models:",
        ]
        for name, info in s["slots"].items():
            marker = "[LOADED]" if info["loaded"] else "[      ]"
            vram = f"~{info['estimated_vram_mb']:,} MB" if info["loaded"] else "-"
            lines.append(f"  {marker} {name:<20s} {vram}")
        return "\n".join(lines)

    # ── Build Gradio app ──────────────────────────────────────────────────
    with gr.Blocks(title="AI Command Center", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# AI Command Center\n"
            "Fine-tuned Llama-3.1-70B-Uncensored orchestrator with specialist agents "
            "for image generation, video generation, vision, code, system automation, "
            "and RAG search over your local documents."
        )

        with gr.Tabs():
            # Chat
            with gr.Tab("Chat"):
                gr.ChatInterface(
                    fn          = chat_fn,
                    title       = "Orchestrator Chat",
                    description = (
                        "Ask anything — the orchestrator will answer directly or delegate "
                        "to specialist agents. For complex tasks, it can chain multiple "
                        "agents together automatically."
                    ),
                    examples    = [
                        "Explain how a nuclear reactor works in detail.",
                        "Generate an image of a cyberpunk city at night.",
                        "Install the latest version of ffmpeg.",
                        "Search my documents for anything about machine learning.",
                        "Write a Python function that scrapes a website and saves the data as CSV.",
                    ],
                )

            # Image generation
            with gr.Tab("Text-to-Image"):
                with gr.Row():
                    t2i_prompt  = gr.Textbox(label="Prompt", lines=3,
                                             placeholder="A photorealistic portrait of a cyberpunk samurai …")
                    with gr.Column():
                        t2i_width  = gr.Slider(256, 2048, value=1024, step=64, label="Width")
                        t2i_height = gr.Slider(256, 2048, value=1024, step=64, label="Height")
                        t2i_steps  = gr.Slider(10, 50, value=28, step=1, label="Steps")
                        t2i_seed   = gr.Textbox(label="Seed (optional)", value="")
                t2i_btn    = gr.Button("Generate Image", variant="primary")
                t2i_output = gr.Image(label="Generated Image", type="filepath")
                t2i_btn.click(generate_image_fn,
                              inputs=[t2i_prompt, t2i_width, t2i_height, t2i_steps, t2i_seed],
                              outputs=t2i_output)

            # Video generation
            with gr.Tab("Text-to-Video"):
                t2v_prompt  = gr.Textbox(label="Prompt", lines=3,
                                         placeholder="A time-lapse of a blooming rose …")
                with gr.Row():
                    t2v_frames = gr.Slider(17, 97, value=49, step=8, label="Frames")
                    t2v_fps    = gr.Slider(4, 24, value=8, step=1, label="FPS")
                    t2v_seed   = gr.Textbox(label="Seed (optional)", value="")
                t2v_btn    = gr.Button("Generate Video", variant="primary")
                t2v_output = gr.Video(label="Generated Video")
                t2v_btn.click(generate_video_fn,
                              inputs=[t2v_prompt, t2v_frames, t2v_fps, t2v_seed],
                              outputs=t2v_output)

            # VLM
            with gr.Tab("Vision (VLM)"):
                with gr.Row():
                    vlm_image    = gr.Image(label="Upload Image", type="numpy")
                    vlm_question = gr.Textbox(label="Question", lines=4,
                                              placeholder="Describe this image in detail.")
                vlm_btn    = gr.Button("Analyse", variant="primary")
                vlm_output = gr.Textbox(label="Answer", lines=8)
                vlm_btn.click(vlm_fn, inputs=[vlm_image, vlm_question], outputs=vlm_output)

            # System agent
            with gr.Tab("System"):
                gr.Markdown(
                    "Run shell commands, install apps, download files.\n\n"
                    "Examples: `install ffmpeg`, `run: dir C:\\\\`, `download https://example.com/file.zip`"
                )
                sys_input  = gr.Textbox(label="Command", lines=3,
                                        placeholder="install ffmpeg")
                sys_btn    = gr.Button("Execute", variant="primary")
                sys_output = gr.Textbox(label="Output", lines=12)
                sys_btn.click(system_fn, inputs=sys_input, outputs=sys_output)

            # RAG
            with gr.Tab("RAG Index"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"**Documents folder:** `{config.RAG_DOCS_DIR}`")
                        rag_force   = gr.Checkbox(label="Force full re-index", value=False)
                        rag_build_btn = gr.Button("Build / Update Index", variant="primary")
                        rag_build_out = gr.Textbox(label="Build result", lines=5)
                        rag_stats_btn = gr.Button("Show Stats")
                        rag_stats_out = gr.Textbox(label="Stats", lines=5)
                    with gr.Column():
                        rag_query_in  = gr.Textbox(label="Search query", lines=2)
                        rag_top_k     = gr.Slider(1, 20, value=6, step=1, label="Top-K results")
                        rag_query_btn = gr.Button("Search", variant="primary")
                        rag_query_out = gr.Textbox(label="Results", lines=20)

                rag_build_btn.click(rag_build_fn, inputs=rag_force, outputs=rag_build_out)
                rag_stats_btn.click(rag_stats_fn, inputs=None, outputs=rag_stats_out)
                rag_query_btn.click(rag_query_fn,
                                    inputs=[rag_query_in, rag_top_k],
                                    outputs=rag_query_out)

            # GPU dashboard
            with gr.Tab("GPU Status"):
                gr.Markdown(
                    "**GPU Memory Dashboard**\n\n"
                    "Shows current VRAM usage and which models are loaded. "
                    "Only one heavy model (T2I, T2V, VLM) is loaded at a time — "
                    "the GPU manager automatically unloads inactive models."
                )
                gpu_output  = gr.Textbox(label="GPU Status", lines=12, interactive=False)
                gpu_btn     = gr.Button("Refresh", variant="primary")
                gpu_btn.click(gpu_status_fn, inputs=None, outputs=gpu_output)

    return demo


def main():
    demo = build_ui()
    demo.queue()
    demo.launch(
        server_port = config.GRADIO_PORT,
        share       = config.GRADIO_SHARE,
        inbrowser   = True,
    )


if __name__ == "__main__":
    main()
