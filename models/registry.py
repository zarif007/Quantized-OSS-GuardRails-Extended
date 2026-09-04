from typing import Dict, List, Optional, Tuple

NONE = "none"
KQUANT = "k-quant"
LEGACY = "legacy-rtn"
IQUANT = "i-quant"

LLAMA_8B_PRECISIONS = {
    "fp16":    {"file": "Llama-Guard-3-8B.f16.gguf",    "size_gb": 16.07, "bits": 16.0, "algorithm": NONE},
    "bf16":    {"file": "Llama-Guard-3-8B.BF16.gguf",   "size_gb": 16.07, "bits": 16.0, "algorithm": NONE},
    "q8_0":    {"file": "Llama-Guard-3-8B.Q8_0.gguf",   "size_gb": 8.54,  "bits": 8.5,  "algorithm": LEGACY},
    "q6_k":    {"file": "Llama-Guard-3-8B.Q6_K.gguf",   "size_gb": 6.59,  "bits": 6.6,  "algorithm": KQUANT},
    "q5_k_m":  {"file": "Llama-Guard-3-8B.Q5_K_M.gguf", "size_gb": 5.73,  "bits": 5.7,  "algorithm": KQUANT},
    "q5_k_s":  {"file": "Llama-Guard-3-8B.Q5_K_S.gguf", "size_gb": 5.60,  "bits": 5.5,  "algorithm": KQUANT},
    "q4_k_m":  {"file": "Llama-Guard-3-8B.Q4_K_M.gguf", "size_gb": 4.92,  "bits": 4.8,  "algorithm": KQUANT},
    "q4_0":    {"file": "Llama-Guard-3-8B.Q4_0.gguf",   "size_gb": 4.66,  "bits": 4.5,  "algorithm": LEGACY},
    "iq4_xs":  {"file": "Llama-Guard-3-8B.IQ4_XS.gguf", "size_gb": 4.45,  "bits": 4.25, "algorithm": IQUANT},
    "q3_k_l":  {"file": "Llama-Guard-3-8B.Q3_K_L.gguf", "size_gb": 4.32,  "bits": 4.2,  "algorithm": KQUANT},
    "q3_k_m":  {"file": "Llama-Guard-3-8B.Q3_K_M.gguf", "size_gb": 3.93,  "bits": 3.9,  "algorithm": KQUANT},
    "q3_k_s":  {"file": "Llama-Guard-3-8B.Q3_K_S.gguf", "size_gb": 3.66,  "bits": 3.6,  "algorithm": KQUANT},
    "iq3_xs":  {"file": "Llama-Guard-3-8B.IQ3_XS.gguf", "size_gb": 3.52,  "bits": 3.4,  "algorithm": IQUANT},
    "q2_k":    {"file": "Llama-Guard-3-8B.Q2_K.gguf",   "size_gb": 3.18,  "bits": 3.0,  "algorithm": KQUANT},
}

QWEN_8B_PRECISIONS = {
    "fp16":   {"file": "Qwen3Guard-Gen-8B.f16.gguf",    "size_gb": 16.4, "bits": 16.0, "algorithm": NONE},
    "q8_0":   {"file": "Qwen3Guard-Gen-8B.Q8_0.gguf",   "size_gb": 8.71, "bits": 8.5,  "algorithm": LEGACY},
    "q6_k":   {"file": "Qwen3Guard-Gen-8B.Q6_K.gguf",   "size_gb": 6.73, "bits": 6.6,  "algorithm": KQUANT},
    "q5_k_m": {"file": "Qwen3Guard-Gen-8B.Q5_K_M.gguf", "size_gb": 5.85, "bits": 5.7,  "algorithm": KQUANT},
    "q4_k_m": {"file": "Qwen3Guard-Gen-8B.Q4_K_M.gguf", "size_gb": 5.03, "bits": 4.8,  "algorithm": KQUANT},
    "q3_k_m": {"file": "Qwen3Guard-Gen-8B.Q3_K_M.gguf", "size_gb": 4.01, "bits": 3.9,  "algorithm": KQUANT},
    "q2_k":   {"file": "Qwen3Guard-Gen-8B.Q2_K.gguf",   "size_gb": 3.28, "bits": 3.0,  "algorithm": KQUANT},
}

FAMILIES = {
    "llama-guard-3-8b": {
        "repo": "mradermacher/Llama-Guard-3-8B-GGUF",
        "hf_id": "meta-llama/Llama-Guard-3-8B",
        "template": "llama_guard_3",
        "n_layers": 32,
        "precisions": LLAMA_8B_PRECISIONS,
        "role": "primary",
    },
    "qwen3guard-gen-8b": {
        "repo": "mradermacher/Qwen3Guard-Gen-8B-GGUF",
        "hf_id": "Qwen/Qwen3Guard-Gen-8B",
        "template": "qwen3guard_gen",
        "n_layers": 36,
        "precisions": QWEN_8B_PRECISIONS,
        "role": "replication",
    },
}

DEFAULT_FAMILY = "llama-guard-3-8b"

ALIASES = {
    "fp16": "llama-guard-3-8b:fp16",
    "bf16": "llama-guard-3-8b:bf16",
    "q8": "llama-guard-3-8b:q8_0",
    "q6": "llama-guard-3-8b:q6_k",
    "q5": "llama-guard-3-8b:q5_k_m",
    "q4": "llama-guard-3-8b:q4_k_m",
    "q3": "llama-guard-3-8b:q3_k_m",
    "q2": "llama-guard-3-8b:q2_k",
}

PRECISION_ORDER = [
    "fp16", "bf16", "q8_0", "q6_k", "q5_k_m", "q5_k_s",
    "q4_k_m", "q4_0", "iq4_xs",
    "q3_k_l", "q3_k_m", "q3_k_s", "iq3_xs", "q2_k",
]

# Scope of THIS paper: the two bit ladders, 14 models.  The question is what
# lower precision does to a guard's operating point.
ACTIVE_GROUP = "all-families-ladder"

BIT_LADDER = ["fp16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"]

# Reserved for the follow-up paper on the algorithm axis ("at a fixed bit
# budget, does the compression method change the decision?").  These stay in
# the registry, and evaluation/gates.py keeps a working test for them, so that
# work resumes from a known-good state -- but nothing in the default path
# downloads or runs them.  Their claims read NOT_TESTED in
# claims_to_evidence.csv, which is the correct record for this paper.
RESERVED_PRECISIONS = ["bf16", "q5_k_s", "q4_0", "iq4_xs", "q3_k_l", "q3_k_s", "iq3_xs"]
# One representative per algorithm family at ~4 bits: legacy round-to-nearest,
# k-quant, and the IQ codebook scheme.
ALGORITHM_AXIS_4BIT = ["q4_0", "q4_k_m", "iq4_xs"]
ALGORITHM_AXIS_3BIT = ["q3_k_s", "q3_k_m", "q3_k_l", "iq3_xs"]


def parse_key(key: str) -> Tuple[str, str]:
    if key in ALIASES:
        key = ALIASES[key]
    if ":" in key:
        family, precision = key.split(":", 1)
    else:
        family, precision = DEFAULT_FAMILY, key
    if family not in FAMILIES:
        raise ValueError(f"Unknown model family '{family}'. Known: {sorted(FAMILIES)}")
    if precision not in FAMILIES[family]["precisions"]:
        raise ValueError(
            f"Unknown precision '{precision}' for {family}. "
            f"Known: {sorted(FAMILIES[family]['precisions'])}"
        )
    return family, precision


def get_config(key: str) -> dict:
    family, precision = parse_key(key)
    fam = FAMILIES[family]
    prec = fam["precisions"][precision]
    return {
        "key": f"{family}:{precision}",
        "family": family,
        "precision": precision,
        "repo": fam["repo"],
        "hf_id": fam["hf_id"],
        "filename": prec["file"],
        "size_gb": prec["size_gb"],
        "bits": prec["bits"],
        "algorithm": prec["algorithm"],
        "template": fam["template"],
        "n_layers": fam["n_layers"],
        "role": fam["role"],
        "quantization_method": "gguf",
    }


def expand(spec: str) -> List[str]:
    if spec == "bit-ladder":
        return [f"{DEFAULT_FAMILY}:{p}" for p in BIT_LADDER]
    if spec == "algorithm-4bit":
        return [f"{DEFAULT_FAMILY}:{p}" for p in ALGORITHM_AXIS_4BIT]
    if spec == "algorithm-3bit":
        return [f"{DEFAULT_FAMILY}:{p}" for p in ALGORITHM_AXIS_3BIT]
    if spec.endswith(":all"):
        family = spec.split(":")[0]
        return [f"{family}:{p}" for p in FAMILIES[family]["precisions"]]
    if spec == "all-families-ladder":
        out = []
        for family, fam in FAMILIES.items():
            out.extend(f"{family}:{p}" for p in BIT_LADDER if p in fam["precisions"])
        return out
    return [get_config(spec)["key"]]


def expand_many(specs: List[str]) -> List[str]:
    seen, out = set(), []
    for spec in specs:
        for key in expand(spec):
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def sort_keys(keys: List[str]) -> List[str]:
    fam_rank = {f: i for i, f in enumerate(FAMILIES)}
    prec_rank = {p: i for i, p in enumerate(PRECISION_ORDER)}

    def rank(key):
        family, precision = parse_key(key)
        return (fam_rank.get(family, 99), prec_rank.get(precision, 99))

    return sorted(keys, key=rank)


def total_disk_gb(keys: List[str]) -> float:
    return sum(get_config(k)["size_gb"] for k in keys)


MODEL_CONFIGS: Dict[str, dict] = {}
for _family, _fam in FAMILIES.items():
    for _precision in _fam["precisions"]:
        _key = f"{_family}:{_precision}"
        MODEL_CONFIGS[_key] = get_config(_key)
for _alias, _target in ALIASES.items():
    MODEL_CONFIGS[_alias] = get_config(_target)
