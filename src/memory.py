"""GPU memory fail-safes: use the largest micro-batch that fits, and back off to smaller ones on out-of-memory.

The effective batch size (config.EFFECTIVE_BATCH) never changes; only how it is split between micro-batches
and gradient accumulation does, so training maths stays the same and only speed and peak VRAM move.
"""
import gc

import torch

from config import EFFECTIVE_BATCH

CANDIDATES = (16, 8, 4, 2, 1)
VRAM_FRACTION = 0.80   # the probe omits optimizer and trainer overhead, so keep headroom
LAST_PROBE = {}        # micro-batch size -> peak bytes measured by the most recent pick_micro_batch call


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vram_budget_bytes(fraction=VRAM_FRACTION):
    """Usable VRAM: what is free on the card now plus what this process already holds, times a safety fraction.
    Other programs (the desktop, a browser) may already be using part of a laptop GPU, so total memory is not used."""
    if not torch.cuda.is_available():
        return None
    free, _ = torch.cuda.mem_get_info()
    return int((free + torch.cuda.memory_reserved()) * fraction)


def pick_micro_batch(try_batch, effective=EFFECTIVE_BATCH, budget_bytes=None, peak_bytes=None, candidates=CANDIDATES):
    """Largest candidate that divides `effective`, runs try_batch(bs) without OOM and peaks under the budget.
    try_batch runs one forward+backward on a worst-case batch. Falls back to 1 when nothing fits."""
    peak_bytes = peak_bytes or (lambda: torch.cuda.max_memory_allocated())
    LAST_PROBE.clear()
    for bs in candidates:
        if effective % bs:
            continue
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            try_batch(bs)
        except torch.cuda.OutOfMemoryError:
            free_cuda()
            continue
        peak = peak_bytes()
        LAST_PROBE[bs] = peak
        free_cuda()
        if budget_bytes is None or peak <= budget_bytes:
            return bs
    return 1


def next_smaller(bs, effective=EFFECTIVE_BATCH):
    """Next micro-batch size below bs that divides the effective batch, or None when already at 1."""
    for c in CANDIDATES:
        if c < bs and effective % c == 0:
            return c
    return None


def run_with_backoff(fn, items, start_batch, min_batch=1):
    """Apply fn(chunk) over items in chunks of start_batch; on OOM halve the chunk size and retry that chunk.
    Returns the concatenated list results and the batch size it ended on."""
    out, i, bs = [], 0, start_batch
    while i < len(items):
        try:
            out += fn(items[i:i + bs])
            i += bs
        except torch.cuda.OutOfMemoryError:
            free_cuda()
            if bs <= min_batch:
                raise
            bs = max(min_batch, bs // 2)
    return out, bs
