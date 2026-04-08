"""
GPU Memory Manager — coordinates VRAM usage across all agents.

With only 24 GB VRAM (RTX 5090), we cannot keep the 70B orchestrator, a
diffusion model, and a VLM loaded simultaneously.  This manager tracks which
model is active, unloads it before loading another, and provides telemetry
so the UI can display current VRAM usage.

Strategy:
    Always-resident : orchestrator GGUF (~22 GB with partial offload)
    On-demand       : T2I, T2V, VLM, Code (loaded/unloaded per request)

Each agent registers with the manager so it can be asked to release VRAM
before a conflicting agent loads.
"""

from __future__ import annotations

import gc
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)

_lock = threading.Lock()


class ModelSlot(Enum):
    ORCHESTRATOR = auto()
    CODE = auto()
    TEXT_TO_IMAGE = auto()
    TEXT_TO_VIDEO = auto()
    VLM = auto()


@dataclass
class _SlotInfo:
    loaded: bool = False
    estimated_vram_mb: int = 0
    unload_fn: Callable[[], None] | None = None


_slots: dict[ModelSlot, _SlotInfo] = {slot: _SlotInfo() for slot in ModelSlot}

ALWAYS_RESIDENT = {ModelSlot.ORCHESTRATOR}

EXCLUSIVE_HEAVY = {
    ModelSlot.TEXT_TO_IMAGE,
    ModelSlot.TEXT_TO_VIDEO,
    ModelSlot.VLM,
}


def _torch_gc() -> None:
    """Force Python + CUDA garbage collection."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def register(
    slot: ModelSlot,
    estimated_vram_mb: int,
    unload_fn: Callable[[], None],
) -> None:
    """Register a model that was just loaded into VRAM."""
    with _lock:
        info = _slots[slot]
        info.loaded = True
        info.estimated_vram_mb = estimated_vram_mb
        info.unload_fn = unload_fn
        logger.info(
            "GPU Manager: registered %s (~%d MB VRAM)", slot.name, estimated_vram_mb
        )


def unregister(slot: ModelSlot) -> None:
    """Mark a slot as no longer loaded (called after manual unload)."""
    with _lock:
        info = _slots[slot]
        info.loaded = False
        info.unload_fn = None
        logger.info("GPU Manager: unregistered %s", slot.name)


def ensure_slot(slot: ModelSlot) -> None:
    """Ensure that ``slot`` can load by evicting conflicting heavy models.

    Models in ``ALWAYS_RESIDENT`` are never evicted.  Of the ``EXCLUSIVE_HEAVY``
    set, only one may be loaded at a time — loading any of them will evict the
    other heavy models.
    """
    if slot in ALWAYS_RESIDENT:
        return

    with _lock:
        if slot in EXCLUSIVE_HEAVY:
            for other_slot in EXCLUSIVE_HEAVY:
                if other_slot == slot:
                    continue
                info = _slots[other_slot]
                if info.loaded and info.unload_fn is not None:
                    logger.info(
                        "GPU Manager: evicting %s to make room for %s",
                        other_slot.name,
                        slot.name,
                    )
                    try:
                        info.unload_fn()
                    except Exception:
                        logger.exception("Error unloading %s", other_slot.name)
                    info.loaded = False
                    info.unload_fn = None

    _torch_gc()


def status() -> dict[str, Any]:
    """Return a snapshot of GPU memory state for the UI."""
    vram_total = 0
    vram_free = 0
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_total = props.total_mem // (1024 * 1024)
            vram_free = (
                props.total_mem - torch.cuda.memory_reserved(0)
            ) // (1024 * 1024)
    except ImportError:
        pass

    slots_info = {}
    with _lock:
        for slot, info in _slots.items():
            slots_info[slot.name] = {
                "loaded": info.loaded,
                "estimated_vram_mb": info.estimated_vram_mb,
            }

    return {
        "vram_total_mb": vram_total,
        "vram_free_mb": vram_free,
        "vram_used_mb": vram_total - vram_free,
        "slots": slots_info,
    }
