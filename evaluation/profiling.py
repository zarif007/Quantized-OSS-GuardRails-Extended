"""
profiling.py — backend-aware memory and latency profiling.

Memory
------
Where the weights live depends on the backend, so a single measurement
strategy cannot be correct everywhere:

  cuda   vram_mb    = NVML delta around the load call (weights + KV cache +
                      compute buffers + CUDA context)
         host_mb    = RSS delta (host-side staging and bookkeeping)
         total      = vram_mb + host_mb
  metal  weights_mb = GGUF file size (unified memory; RSS cannot see it)
         host_mb    = RSS delta
         total      = weights_mb + host_mb
  cpu    total      = max(RSS delta, GGUF size)
                      llama.cpp mmaps the file, so RSS under-reports until
                      pages are faulted in; the file size is the floor.

`weights_mb` is always reported separately.  It is the one hardware-
independent memory figure and is the honest x-axis for a Pareto plot.

Partial offload
---------------
When VRAM is short, llama.cpp silently places some layers on the host and
runs a hybrid.  Timings from such a run are not comparable to a fully
offloaded one, so `offload_ok` flags it and callers abort.

Latency
-------
`InferenceProfiler` measures wall clock only.  Aggregation is median-based
(see `latency_summary`): on a shared cloud host a single scheduling hiccup
moves a mean but not a median.
"""

import math
import os
import time
from typing import Dict, List, Optional, Sequence

from evaluation.hardware import CPU, CUDA, METAL, MemorySampler

# A full offload should account for at least this share of the weight bytes.
OFFLOAD_TOLERANCE = 0.90


class ModelProfiler:
    """
    Wraps a model load to capture memory usage on whichever backend is active.

    Usage
    -----
    profiler = ModelProfiler(model_path, backend="cuda")
    profiler.before_load()
    model = LLMGuard(...)
    profiler.after_load()
    stats = profiler.stats()
    """

    def __init__(self, model_path: str, backend: str = CPU):
        self.model_path = model_path
        self.backend = backend
        self._sampler = MemorySampler(backend)

        self._host_before = 0.0
        self._host_after = 0.0
        self._device_before = float("nan")
        self._device_after = float("nan")

        self.weights_mb = 0.0
        self.host_mb = 0.0
        self.vram_mb = float("nan")
        self.total_memory_mb = 0.0
        self.offload_ok: Optional[bool] = None
        self.notes: List[str] = []

    def before_load(self) -> None:
        self._host_before = self._sampler.host_rss_mb()
        self._device_before = self._sampler.device_mb()

    def after_load(self) -> None:
        self._host_after = self._sampler.host_rss_mb()
        self._device_after = self._sampler.device_mb()

        self.weights_mb = os.path.getsize(self.model_path) / (1024 * 1024)
        self.host_mb = max(0.0, self._host_after - self._host_before)

        if self.backend == CUDA:
            if math.isnan(self._device_after) or math.isnan(self._device_before):
                self.vram_mb = float("nan")
                self.total_memory_mb = self.weights_mb + self.host_mb
                self.offload_ok = None
                self.notes.append(
                    "NVML unavailable; VRAM not measured and offload not verified. "
                    "Install nvidia-ml-py (pip install nvidia-ml-py)."
                )
            else:
                self.vram_mb = max(0.0, self._device_after - self._device_before)
                self.total_memory_mb = self.vram_mb + self.host_mb
                self.offload_ok = self.vram_mb >= OFFLOAD_TOLERANCE * self.weights_mb
                if not self.offload_ok:
                    self.notes.append(
                        f"VRAM delta {self.vram_mb:.0f} MB is below "
                        f"{OFFLOAD_TOLERANCE:.0%} of the {self.weights_mb:.0f} MB weight file: "
                        f"llama.cpp likely offloaded only some layers and is running a "
                        f"CPU/GPU hybrid. Timings from this run are not comparable."
                    )
        elif self.backend == METAL:
            self.total_memory_mb = self.weights_mb + self.host_mb
            self.offload_ok = True
        else:
            self.total_memory_mb = max(self.host_mb, self.weights_mb)
            self.offload_ok = True

        self._sampler.close()

    def stats(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "weights_mb": round(self.weights_mb, 1),
            "vram_mb": None if math.isnan(self.vram_mb) else round(self.vram_mb, 1),
            "host_mb": round(self.host_mb, 1),
            "total_memory_mb": round(self.total_memory_mb, 1),
            "offload_ok": self.offload_ok,
        }

    def summary(self) -> str:
        vram = "n/a" if math.isnan(self.vram_mb) else f"{self.vram_mb:>8.1f} MB"
        lines = [
            f"  backend:                            {self.backend}",
            f"  Weight file (GGUF on disk):         {self.weights_mb:>8.1f} MB",
            f"  Device memory (VRAM delta):         {vram}",
            f"  Host RSS delta:                     {self.host_mb:>8.1f} MB",
            f"  Total reported memory:              {self.total_memory_mb:>8.1f} MB",
        ]
        for note in self.notes:
            lines.append(f"  ! {note}")
        return "\n".join(lines)


class InferenceProfiler:
    """Wall-clock latency for a single scoring call."""

    def __init__(self):
        self._t0: float = 0.0
        self.latency_sec: float = 0.0

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        self.latency_sec = time.perf_counter() - self._t0
        return self.latency_sec


def latency_summary(values: Sequence[float]) -> Dict[str, float]:
    """
    Median-centred latency statistics.

    The median and IQR are reported instead of the mean because these runs
    share a host with other tenants; p95 is kept because a guardrail's tail
    latency is what a deployment actually feels.
    """
    import numpy as np

    clean = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if clean.size == 0:
        return {k: float("nan") for k in
                ("latency_median_sec", "latency_p25_sec", "latency_p75_sec",
                 "latency_p95_sec", "latency_mean_sec", "latency_std_sec", "latency_n")}
    return {
        "latency_median_sec": float(np.median(clean)),
        "latency_p25_sec": float(np.percentile(clean, 25)),
        "latency_p75_sec": float(np.percentile(clean, 75)),
        "latency_p95_sec": float(np.percentile(clean, 95)),
        "latency_mean_sec": float(np.mean(clean)),
        "latency_std_sec": float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0,
        "latency_n": int(clean.size),
    }
