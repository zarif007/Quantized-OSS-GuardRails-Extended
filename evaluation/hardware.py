"""
hardware.py — backend detection, device metadata, and memory sampling.

Why this module exists
----------------------
Latency, throughput and memory are properties of the
(model, quantization, inference engine, hardware) configuration — not of
quantization alone.  A cross-precision comparison of those metrics is valid
only if every other element of that tuple is held fixed.

So every run records a full environment fingerprint, and `analyze.py` refuses
to build deployment tables from prediction files whose fingerprints disagree.
That turns "we ran everything on the same pod" from an assumption into a
checked precondition.

Memory measurement is backend-dependent:

  cuda   — weights live in VRAM.  psutil.rss is blind to them.  We sample
           NVML around the load call (per-process where the driver exposes
           it, device-wide otherwise) and report the delta.
  metal  — unified memory.  RSS is blind to GPU-resident weights, so the
           GGUF file size is the authoritative weight figure and the RSS
           delta captures CPU-side overhead.
  cpu    — everything is in host RAM, so the RSS delta is the whole story.
"""

import hashlib
import json
import os
import platform
import subprocess
from typing import Dict, Optional

import psutil

CUDA = "cuda"
METAL = "metal"
CPU = "cpu"


# ----------------------------------------------------------------------
# Backend detection
# ----------------------------------------------------------------------

def cuda_available() -> bool:
    try:
        import pynvml  # noqa: F401
    except ImportError:
        return _nvidia_smi_present()
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return count > 0
    except Exception:
        return _nvidia_smi_present()


def _nvidia_smi_present() -> bool:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10)
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:
        return False


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def detect_backend(n_gpu_layers: int) -> str:
    """
    Resolve the backend that llama.cpp will actually use.

    n_gpu_layers == 0 means CPU regardless of what hardware is present.
    """
    if n_gpu_layers == 0:
        return CPU
    if cuda_available():
        return CUDA
    if is_apple_silicon():
        return METAL
    return CPU


def torch_device(preferred: str = "auto") -> str:
    """Device string for the PyTorch paths (layer sweep, fake quant)."""
    if preferred != "auto":
        return preferred
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# float32 is not an option for the 8B guard models: the weights alone need
# ~32 GB, which does not fit the experiment hardware.  It also buys nothing
# for the fake-quantization probe, whose perturbation is a 3-4 bit RTN
# round-trip -- orders of magnitude coarser than bfloat16 mantissa rounding
# in the baseline.  bfloat16 is preferred over float16 because it keeps
# float32's exponent range, so logits cannot saturate.
UNSUPPORTED_DTYPES = ("float32", "float64", "double")


def default_torch_dtype(device: str) -> str:
    """bfloat16 everywhere; float16 on MPS, where bfloat16 support is patchy."""
    return "float16" if device == "mps" else "bfloat16"


def check_dtype_supported(dtype: str, device: str) -> None:
    """Reject dtypes the experiment hardware cannot hold."""
    if dtype.lower().replace("torch.", "") in UNSUPPORTED_DTYPES:
        raise ValueError(
            f"dtype '{dtype}' is not supported for the 8B guard models: the weights "
            f"alone need ~32 GB, beyond the experiment hardware, and the extra "
            f"mantissa is irrelevant to a 3-4 bit RTN perturbation. "
            f"Use bfloat16 (default on {device}) or float16."
        )


# ----------------------------------------------------------------------
# Device metadata
# ----------------------------------------------------------------------

def _nvml_handles():
    import pynvml
    pynvml.nvmlInit()
    return pynvml, [pynvml.nvmlDeviceGetHandleByIndex(i)
                    for i in range(pynvml.nvmlDeviceGetCount())]


def gpu_info() -> Dict[str, object]:
    """Name, driver, CUDA version and total memory for the visible GPU(s)."""
    info: Dict[str, object] = {
        "gpu_name": None,
        "gpu_count": 0,
        "gpu_total_memory_mb": None,
        "driver_version": None,
        "cuda_version": None,
    }
    try:
        pynvml, handles = _nvml_handles()
    except Exception:
        return _gpu_info_from_smi(info)

    try:
        names = []
        totals = []
        for handle in handles:
            name = pynvml.nvmlDeviceGetName(handle)
            names.append(name.decode() if isinstance(name, bytes) else name)
            totals.append(pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 * 1024))
        driver = pynvml.nvmlSystemGetDriverVersion()
        info.update({
            "gpu_name": names[0] if names else None,
            "gpu_count": len(handles),
            "gpu_total_memory_mb": round(totals[0], 1) if totals else None,
            "driver_version": driver.decode() if isinstance(driver, bytes) else driver,
        })
        try:
            raw = pynvml.nvmlSystemGetCudaDriverVersion()
            info["cuda_version"] = f"{raw // 1000}.{(raw % 1000) // 10}"
        except Exception:
            pass
    except Exception:
        pass
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return info


def _gpu_info_from_smi(info: Dict[str, object]) -> Dict[str, object]:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return info
        lines = [l for l in out.stdout.strip().splitlines() if l.strip()]
        name, total, driver = [p.strip() for p in lines[0].split(",")]
        info.update({
            "gpu_name": name,
            "gpu_count": len(lines),
            "gpu_total_memory_mb": float(total),
            "driver_version": driver,
        })
    except Exception:
        pass
    return info


def llama_cpp_version() -> Optional[str]:
    try:
        import llama_cpp
        return getattr(llama_cpp, "__version__", None)
    except Exception:
        return None


def llama_cpp_supports_offload() -> Optional[bool]:
    """
    True when the installed llama-cpp-python wheel was built with GPU support.

    A plain `pip install llama-cpp-python` produces a CPU-only wheel, which
    would silently run on the host CPU on a rented GPU pod.  Callers use this
    to fail loudly instead.
    """
    try:
        import llama_cpp
        fn = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        return bool(fn()) if fn is not None else None
    except Exception:
        return None


def env_fingerprint(n_gpu_layers: int = -1) -> Dict[str, object]:
    """The full configuration that latency/throughput/memory are conditional on."""
    backend = detect_backend(n_gpu_layers)
    fp: Dict[str, object] = {
        "backend": backend,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "host_total_memory_mb": round(psutil.virtual_memory().total / (1024 * 1024), 1),
        "python": platform.python_version(),
        "llama_cpp_version": llama_cpp_version(),
        "llama_cpp_gpu_offload_build": llama_cpp_supports_offload(),
    }
    if backend == CUDA:
        fp.update(gpu_info())
    try:
        import torch
        fp["torch_version"] = torch.__version__
        fp["torch_cuda"] = getattr(torch.version, "cuda", None)
    except Exception:
        pass
    return fp


# Fields that must match across every run in one comparison.  Host RAM,
# CPU counts and library patch versions are recorded but not enforced,
# because they do not change the measured configuration on a fixed pod.
FINGERPRINT_KEYS = ("backend", "gpu_name", "driver_version", "llama_cpp_version", "processor")


def fingerprint_hash(fp: Dict[str, object]) -> str:
    payload = {k: fp.get(k) for k in FINGERPRINT_KEYS}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ----------------------------------------------------------------------
# Memory sampling
# ----------------------------------------------------------------------

class MemorySampler:
    """
    Backend-aware memory sampler.

    On CUDA it reads NVML: per-process where the driver reports it (the
    reliable figure on a shared host), device-wide otherwise.  On Metal and
    CPU it reads process RSS.
    """

    def __init__(self, backend: str, pid: Optional[int] = None):
        self.backend = backend
        self.pid = pid or os.getpid()
        self._process = psutil.Process(self.pid)
        self._nvml_ok = False
        if backend == CUDA:
            try:
                import pynvml
                pynvml.nvmlInit()
                self._pynvml = pynvml
                self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                                 for i in range(pynvml.nvmlDeviceGetCount())]
                self._nvml_ok = len(self._handles) > 0
            except Exception:
                self._nvml_ok = False

    def host_rss_mb(self) -> float:
        return self._process.memory_info().rss / (1024 * 1024)

    def device_mb(self) -> float:
        """VRAM attributable to this process; NaN when unavailable."""
        if not self._nvml_ok:
            return float("nan")
        pynvml = self._pynvml
        for handle in self._handles:
            try:
                for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    if proc.pid == self.pid and proc.usedGpuMemory:
                        return proc.usedGpuMemory / (1024 * 1024)
            except Exception:
                continue
        # Driver did not attribute memory per process (common in containers).
        try:
            return pynvml.nvmlDeviceGetMemoryInfo(self._handles[0]).used / (1024 * 1024)
        except Exception:
            return float("nan")

    def close(self):
        if self._nvml_ok:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
