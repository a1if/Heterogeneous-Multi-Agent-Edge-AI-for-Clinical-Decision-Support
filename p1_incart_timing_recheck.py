"""Deviation 21a: re-measure INCART time to first token at N = 50 under the GPU memory cap. GPU.

Calls the frozen p1_item7_filtered.timing("incart") unchanged; from outside it (1) caps PyTorch at 90% of the
card's dedicated memory, (2) releases cached allocator blocks before each timed call (the frozen routine
builds its SchemaJsonProcessor just before starting the clock, so the release happens outside the timer),
(3) keeps only the N = 50 windows (select() then takes the same 21 windows as the original run), and
(4) writes results/p1_incart_timing_n50_capped.json instead of the original file, which is left unchanged.

Run (from repo root):  python p1_incart_timing_recheck.py
"""
import gc
from pathlib import Path

import torch

import p1_item7_filtered as F
import reasoning.constrained_json as cj

OUT = "results/p1_incart_timing_n50_capped.json"
MEM_FRACTION = 0.90


class _ReleasingProcessor(cj.SchemaJsonProcessor):
    def __init__(self, *a, **k):
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        super().__init__(*a, **k)


def _path(p, *rest):
    return Path(OUT) if str(p).replace("\\", "/") == "results/p1_incart_timing.json" else Path(p, *rest)


def main():
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, 0)
    original_windows = F.windows
    F.windows = lambda split="ds2v2": [w for w in original_windows(split) if w["n"] == 50]
    F.Path = _path
    cj.SchemaJsonProcessor = _ReleasingProcessor
    F.timing("incart")


if __name__ == "__main__":
    main()
