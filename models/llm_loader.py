import math
import os
from typing import Dict, List, Optional

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from llama_cpp import Llama

from models.registry import MODEL_CONFIGS, PRECISION_ORDER, get_config, parse_key
from models.templates import GuardTemplate, get_template, template_fingerprint



class ModelFileNotFound(RuntimeError):
    pass


def resolve_model_path(config: dict, download_dir: str = "./models/weights") -> str:
    override = config.get("local_path")
    if override:
        if not os.path.exists(override):
            raise ModelFileNotFound(f"local_path does not exist: {override}")
        return override
    try:
        return hf_hub_download(
            repo_id=config["repo"],
            filename=config["filename"],
            cache_dir=download_dir,
            token=os.environ.get("HF_TOKEN") or None,
        )
    except Exception as exc:
        try:
            available = sorted(
                f for f in list_repo_files(config["repo"]) if f.endswith(".gguf")
            )
        except Exception:
            available = []
        listing = "\n  ".join(available) if available else "<could not list repo>"
        raise ModelFileNotFound(
            f"Could not fetch {config['filename']} from {config['repo']}.\n"
            f"Original error: {exc}\n"
            f"GGUF files present in that repo:\n  {listing}\n"
            f"Update MODEL_CONFIGS with the correct filename."
        ) from exc


class LabelTokens:
    def __init__(self, llm: Llama, template: GuardTemplate):
        self.safe_ids = self._resolve(llm, template.safe_variants)
        self.unsafe_ids = self._resolve(llm, template.unsafe_variants)
        overlap = self.safe_ids & self.unsafe_ids
        if overlap:
            raise RuntimeError(
                f"Safe and unsafe label tokens collide on ids {sorted(overlap)}. "
                f"The scorer cannot separate the classes."
            )
        if not self.safe_ids or not self.unsafe_ids:
            raise RuntimeError("Failed to resolve label token ids for this tokenizer.")

    @staticmethod
    def _resolve(llm: Llama, variants) -> set:
        ids = set()
        for variant in variants:
            toks = llm.tokenize(variant.encode("utf-8"), add_bos=False, special=False)
            if toks:
                ids.add(int(toks[0]))
        return ids

    def describe(self, llm: Llama) -> str:
        def render(ids):
            return ", ".join(
                f"{i}:{llm.detokenize([i]).decode('utf-8', errors='replace')!r}"
                for i in sorted(ids)
            )

        return f"safe -> [{render(self.safe_ids)}]\nunsafe -> [{render(self.unsafe_ids)}]"


class LLMGuard:
    def __init__(
        self,
        quant_level: str,
        download_dir: str = "./models/weights",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        n_batch: Optional[int] = None,
        flash_attn: bool = False,
        seed: int = 42,
        use_prefix_cache: bool = True,
        config_override: Optional[dict] = None,
    ):
        config = config_override or get_config(quant_level)
        self.quant_level = config["key"]
        self.precision = config["precision"]
        self.family = config["family"]
        self.config = config
        self.template = get_template(config["template"])
        self.template_fingerprint = template_fingerprint(self.template)
        self.use_prefix_cache = use_prefix_cache

        # Runtime knobs are recorded because latency and throughput are
        # conditional on them, not just on the quantization level.
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.flash_attn = flash_attn

        self.model_path = resolve_model_path(config, download_dir)

        kwargs = dict(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            logits_all=False,
            verbose=False,
            seed=seed,
        )
        if n_threads is not None:
            kwargs["n_threads"] = n_threads
        if n_batch is not None:
            kwargs["n_batch"] = n_batch
        if flash_attn:
            kwargs["flash_attn"] = True

        try:
            self.llm = Llama(**kwargs)
        except TypeError as exc:
            # Older llama-cpp-python builds reject flash_attn / n_batch.
            for optional in ("flash_attn", "n_batch"):
                kwargs.pop(optional, None)
            print(f"  note: retrying load without optional kwargs ({exc})")
            self.llm = Llama(**kwargs)
        self.labels = LabelTokens(self.llm, self.template)

        self._prefix_tokens: List[int] = self.llm.tokenize(
            self.template.cacheable_prefix().encode("utf-8"), add_bos=False, special=True
        )
        self._prefix_state = None
        self._last_eval_tokens = 0
        if self.use_prefix_cache:
            self._build_prefix_state()

    def _build_prefix_state(self):
        self.llm.reset()
        self.llm.eval(self._prefix_tokens)
        self._prefix_state = self.llm.save_state()

    def _n_vocab(self) -> int:
        for accessor in ("n_vocab", "_n_vocab"):
            value = getattr(self.llm, accessor, None)
            if value is None:
                continue
            try:
                return int(value() if callable(value) else value)
            except Exception:
                continue
        raise RuntimeError("Could not determine the vocabulary size from llama_cpp.")

    def _last_logits(self) -> np.ndarray:
        """
        Logits for the final evaluated token.

        Read straight from the context's logit buffer rather than from
        `Llama.scores`.  With `logits_all=False`, llama-cpp-python >= 0.3 no
        longer copies logits into `scores` at all — the copy is commented out
        inside `Llama.eval` — so `scores` stays a zero matrix.  Reading it
        would silently yield margin == 0 and p_unsafe == 0.5 for every prompt,
        which looks like a working run and is not one.

        `scores` is kept as a fallback for older builds that do populate it;
        note the correct row there is `n_tokens - 1`, not `-1`, because the
        buffer is preallocated to n_ctx rows.
        """
        n_vocab = self._n_vocab()
        logits = None

        try:
            pointer = self.llm._ctx.get_logits()
            if pointer:
                logits = np.ctypeslib.as_array(pointer, shape=(n_vocab,)).astype(np.float64)
        except Exception:
            logits = None

        if logits is None or not np.any(logits):
            scores = self.llm.scores
            if scores is not None and len(scores):
                row = max(0, int(getattr(self.llm, "n_tokens", len(scores))) - 1)
                candidate = np.asarray(scores[row], dtype=np.float64)
                if np.any(candidate):
                    logits = candidate

        if logits is None or not np.any(logits):
            raise RuntimeError(
                "llama_cpp returned an empty logit vector. The context exposed no "
                "logits for the last token and Llama.scores is all zeros. This is "
                "usually a llama-cpp-python version mismatch; check that eval() "
                "succeeded and that the build is intact."
            )

        if logits.size < 1000 or not np.isfinite(logits).any():
            raise RuntimeError(
                f"Logit vector looks wrong (size={logits.size}). "
                f"Expected the full vocabulary."
            )
        return logits

    def _eval_prompt(self, prompt: str) -> np.ndarray:
        tail = self.template.tail(prompt)
        if self.use_prefix_cache and self._prefix_state is not None:
            self.llm.load_state(self._prefix_state)
            tokens = self.llm.tokenize(tail.encode("utf-8"), add_bos=False, special=True)
        else:
            self.llm.reset()
            full = self.template.render(prompt)
            tokens = self.llm.tokenize(full.encode("utf-8"), add_bos=False, special=True)
        self._last_eval_tokens = len(tokens)
        self.llm.eval(tokens)
        return self._last_logits()

    def predict_score(self, prompt: str) -> dict:
        logits = self._eval_prompt(prompt)
        logit_safe = float(max(logits[i] for i in self.labels.safe_ids))
        logit_unsafe = float(max(logits[i] for i in self.labels.unsafe_ids))
        margin = logit_unsafe - logit_safe
        p_unsafe = 1.0 / (1.0 + math.exp(-margin)) if abs(margin) < 500 else float(margin > 0)
        top_id = int(np.argmax(logits))
        return {
            "eval_tokens": int(self._last_eval_tokens),
            "context_tokens": int(self._last_eval_tokens +
                                  (len(self._prefix_tokens) if self.use_prefix_cache else 0)),
            "prediction": "unsafe" if margin > 0.0 else "safe",
            "p_unsafe": p_unsafe,
            "logit_safe": logit_safe,
            "logit_unsafe": logit_unsafe,
            "margin": margin,
            "raw_output": self.llm.detokenize([top_id]).decode("utf-8", errors="replace"),
        }

    def predict(self, prompt: str) -> str:
        return self.predict_score(prompt)["prediction"]

    def predict_generate(self, prompt: str, max_tokens: int = 16) -> str:
        self.llm.reset()
        output = self.llm(
            self.template.render(prompt),
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=["<|eot_id|>", "<|im_end|>"],
        )
        return output["choices"][0]["text"].strip()

    def token_report(self) -> str:
        return self.labels.describe(self.llm)

    def prefix_token_count(self) -> int:
        return len(self._prefix_tokens)

    def run_metadata(self) -> dict:
        return {
            "model_key": self.config["key"],
            "precision": self.config["precision"],
            "family": self.config["family"],
            "algorithm": self.config["algorithm"],
            "bits": self.config["bits"],
            "size_gb": self.config["size_gb"],
            "repo": self.config["repo"],
            "filename": self.config["filename"],
            "quantization_method": self.config["quantization_method"],
            "template_name": self.template.name,
            "template_fingerprint": self.template_fingerprint,
            "prefix_tokens": len(self._prefix_tokens),
            "prefix_cache": self.use_prefix_cache,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads": self.n_threads,
            "n_batch": self.n_batch,
            "flash_attn": self.flash_attn,
        }
