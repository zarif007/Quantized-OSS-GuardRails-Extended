import math
import os
from typing import Dict, List, Optional

import torch

from evaluation.hardware import check_dtype_supported, default_torch_dtype, torch_device
from models.registry import get_config
from models.templates import get_template, template_fingerprint


class HFGuard:
    """
    PyTorch scorer for the fake-quantization layer sweep.

    dtype
    -----
    bfloat16 by default (float16 on MPS).  float32 is rejected: an 8B model
    in float32 needs ~32 GB, beyond the experiment hardware, and the extra
    mantissa is irrelevant to a 3-4 bit RTN perturbation.  bfloat16 is
    preferred over float16 because it keeps float32's exponent range, so
    logits cannot saturate.
    """

    def __init__(self, family: str = "llama-guard-3-8b", dtype: str = "auto",
                 device: str = "auto", hf_token: Optional[str] = None,
                 cache_dir: str = "./models/hf"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch_device(device)
        if dtype == "auto":
            dtype = default_torch_dtype(device)
        check_dtype_supported(dtype, device)
        hf_token = hf_token or os.environ.get("HF_TOKEN") or None

        config = get_config(f"{family}:fp16")
        self.family = family
        self.hf_id = config["hf_id"]
        self.template = get_template(config["template"])
        self.template_fingerprint = template_fingerprint(self.template)
        self.device = device
        self.torch_dtype = getattr(torch, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id, token=hf_token, cache_dir=cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            token=hf_token,
            cache_dir=cache_dir,
        ).to(device)
        self.model.eval()

        self.safe_ids = self._resolve(self.template.safe_variants)
        self.unsafe_ids = self._resolve(self.template.unsafe_variants)
        if set(self.safe_ids) & set(self.unsafe_ids):
            raise RuntimeError("safe and unsafe label tokens collide")

        self._prefix_cache = None
        self._prefix_len = 0
        self._cache_supports_crop = False

    def _resolve(self, variants) -> List[int]:
        ids = []
        for v in variants:
            toks = self.tokenizer.encode(v, add_special_tokens=False)
            if toks and toks[0] not in ids:
                ids.append(toks[0])
        return ids

    def layers(self):
        for attr in ("model.layers", "transformer.h", "model.decoder.layers"):
            obj = self.model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise RuntimeError("Could not locate transformer layers on this architecture")

    def lm_head_weight(self) -> torch.Tensor:
        return self.model.get_output_embeddings().weight.data

    @torch.no_grad()
    def build_prefix_cache(self):
        ids = self.tokenizer(
            self.template.cacheable_prefix(), return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        out = self.model(ids, use_cache=True)
        self._prefix_cache = out.past_key_values
        self._prefix_len = ids.shape[1]
        self._cache_supports_crop = hasattr(self._prefix_cache, "crop")

    def invalidate_prefix_cache(self):
        self._prefix_cache = None
        self._prefix_len = 0
        self._cache_supports_crop = False

    def _restore_prefix_cache(self):
        """
        Return the KV cache to just the prefix after a prompt has extended it.

        Modern transformers caches (DynamicCache) are mutated in place by the
        forward pass, so the prefix has to be restored between prompts.  A
        crop is O(1) slicing; the deep copy it replaces allocated a fresh copy
        of the whole cache on every single prompt, which on GPU is a VRAM
        allocation per call.
        """
        if self._prefix_cache is None:
            return
        if self._cache_supports_crop:
            self._prefix_cache.crop(self._prefix_len)
        # Legacy tuple caches are immutable: the forward pass builds new
        # tensors and leaves the originals alone, so nothing to restore.

    @torch.no_grad()
    def predict_score(self, prompt: str) -> Dict[str, float]:
        tail = self.template.tail(prompt)
        if self._prefix_cache is not None:
            ids = self.tokenizer(tail, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            out = self.model(ids, past_key_values=self._prefix_cache, use_cache=True)
            self._restore_prefix_cache()
        else:
            full = self.template.render(prompt)
            ids = self.tokenizer(full, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            out = self.model(ids, use_cache=False)

        logits = out.logits[0, -1, :].float()
        logit_safe = float(logits[self.safe_ids].max())
        logit_unsafe = float(logits[self.unsafe_ids].max())
        margin = logit_unsafe - logit_safe
        return {
            "prediction": "unsafe" if margin > 0 else "safe",
            "p_unsafe": 1.0 / (1.0 + math.exp(-max(min(margin, 500), -500))),
            "logit_safe": logit_safe,
            "logit_unsafe": logit_unsafe,
            "margin": margin,
            "raw_output": self.tokenizer.decode([int(logits.argmax())]),
        }

    def score_frame(self, prompts) -> List[Dict[str, float]]:
        return [self.predict_score(p) for p in prompts]
