from typing import Dict, List, Optional

import torch


def rtn_quantize_dequantize(weight: torch.Tensor, bits: int = 4, group_size: int = 32,
                            symmetric: bool = True) -> torch.Tensor:
    original_shape = weight.shape
    flat = weight.reshape(-1).float()

    pad = (-flat.numel()) % group_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])
    groups = flat.reshape(-1, group_size)

    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        scale = groups.abs().amax(dim=1, keepdim=True) / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.clamp(torch.round(groups / scale), -qmax - 1, qmax)
        deq = q * scale
    else:
        qmax = 2**bits - 1
        lo = groups.amin(dim=1, keepdim=True)
        hi = groups.amax(dim=1, keepdim=True)
        scale = (hi - lo) / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.clamp(torch.round((groups - lo) / scale), 0, qmax)
        deq = q * scale + lo

    out = deq.reshape(-1)
    if pad:
        out = out[:-pad]
    return out.reshape(original_shape).to(weight.dtype)


class LayerPerturbation:
    def __init__(self, module: torch.nn.Module, bits: int = 4, group_size: int = 32,
                 symmetric: bool = True, targets: Optional[List[str]] = None):
        self.module = module
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.targets = targets
        self._saved: Dict[str, torch.Tensor] = {}

    def _selected(self):
        for name, sub in self.module.named_modules():
            if not isinstance(sub, torch.nn.Linear):
                continue
            if self.targets and not any(t in name for t in self.targets):
                continue
            yield name, sub

    def __enter__(self):
        for name, sub in self._selected():
            self._saved[name] = sub.weight.data.clone()
            sub.weight.data = rtn_quantize_dequantize(
                sub.weight.data, self.bits, self.group_size, self.symmetric
            )
        return self

    def __exit__(self, *exc):
        for name, sub in self.module.named_modules():
            if name in self._saved:
                sub.weight.data = self._saved[name]
        self._saved.clear()
        return False


def quantization_error(weight: torch.Tensor, bits: int = 4, group_size: int = 32) -> Dict[str, float]:
    deq = rtn_quantize_dequantize(weight, bits, group_size)
    err = (deq - weight).float()
    return {
        "mse": float(err.pow(2).mean()),
        "max_abs_error": float(err.abs().max()),
        "relative_frobenius": float(err.norm() / max(float(weight.float().norm()), 1e-12)),
    }


def output_projection_asymmetry(lm_head_weight: torch.Tensor, safe_ids: List[int],
                                unsafe_ids: List[int], bits: int = 4,
                                group_size: int = 32) -> Dict[str, float]:
    deq = rtn_quantize_dequantize(lm_head_weight, bits, group_size)
    delta = (deq - lm_head_weight).float()

    def group_stats(ids):
        rows = delta[ids].float()
        base = lm_head_weight[ids].float()
        return {
            "mean_perturbation": float(rows.mean()),
            "mean_abs_perturbation": float(rows.abs().mean()),
            "relative_norm": float(rows.norm() / max(float(base.norm()), 1e-12)),
            "norm_change": float(deq[ids].float().norm() - base.norm()),
        }

    safe_stats = group_stats(safe_ids)
    unsafe_stats = group_stats(unsafe_ids)
    return {
        **{f"safe_{k}": v for k, v in safe_stats.items()},
        **{f"unsafe_{k}": v for k, v in unsafe_stats.items()},
        "asymmetry_mean": unsafe_stats["mean_perturbation"] - safe_stats["mean_perturbation"],
        "asymmetry_relative_norm": unsafe_stats["relative_norm"] - safe_stats["relative_norm"],
        "all_rows_relative_norm": float(delta.norm() / max(float(lm_head_weight.float().norm()), 1e-12)),
    }
