"""
The datasets this paper scores, and how to normalize each into the same shape.

Scope note.  This registry once held 25 specs across six tiers (over-refusal,
category/severity, adversarial, multilingual, response-level).  Those tiers
answered Phase 8 questions -- different tasks, reported in their own tables --
and none of them fed the four gates.  They were removed rather than left
declared but unused; the git history has them if Phase 8 is ever picked up.

What is left is the five prompt-level English safety sets the paper actually
runs.  They all ask one question -- given a prompt, is it harmful? -- so
pooling them is legitimate and the pooled AUROC means something.  Mixing in a
multilingual or response-level set would not be: the gates would then answer
"does quantization shift the boundary across a grab-bag of tasks", which is
not a question anyone asked.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

TIER_A = "A"

TIER_NAMES = {
    TIER_A: "core prompt-level safety",
}


@dataclass
class DatasetSpec:
    name: str
    tier: str
    hf_id: Optional[str]
    config: Optional[str]
    split: str
    normalize: Callable
    level: str = "prompt"
    local_csv: Optional[str] = None
    notes: str = ""
    approx_rows: int = 0
    verified: bool = False
    # Gated on the Hub.  All of these are 'auto' gating: accepting the terms on
    # the dataset page grants access immediately, no manual review.
    gated: bool = False

    @property
    def hub_url(self) -> str:
        return f"https://huggingface.co/datasets/{self.hf_id}" if self.hf_id else ""


def _row(prompt, label, category=None, language="en", source_extra=None):
    out = {"prompt": prompt, "ground_truth": label, "category": category, "language": language}
    if source_extra:
        out.update(source_extra)
    return out


def norm_harmbench(ds):
    return [_row(r.get("prompt") or r.get("Behavior"), "unsafe", r.get("category") or r.get("SemanticCategory"))
            for r in ds]


def norm_xstest(ds):
    out = []
    for r in ds:
        prompt = r.get("prompt") or r.get("Prompt")
        raw = (r.get("label") or r.get("type") or "").lower()
        label = "unsafe" if raw.startswith("contrast") or raw == "unsafe" else "safe"
        out.append(_row(prompt, label, r.get("type")))
    return out


def norm_toxicchat(ds):
    out = []
    for r in ds:
        label = "unsafe" if int(r.get("toxicity", 0)) == 1 else "safe"
        out.append(_row(r.get("user_input"), label, "jailbreak" if r.get("jailbreaking") else None))
    return out


def norm_wildguard(ds):
    out = []
    for r in ds:
        harm = (r.get("prompt_harm_label") or "").lower()
        if harm not in ("harmful", "unharmful"):
            continue
        out.append(_row(r.get("prompt"), "unsafe" if harm == "harmful" else "safe", r.get("subcategory")))
    return out


OPENAI_MOD_FLAGS = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]


def norm_openai_moderation(ds):
    out = []
    for r in ds:
        flags = [f for f in OPENAI_MOD_FLAGS if int(r.get(f, 0) or 0) == 1]
        out.append(_row(r.get("prompt"), "unsafe" if flags else "safe", ",".join(flags) or None))
    return out


DATASETS: Dict[str, DatasetSpec] = {
    "harmbench": DatasetSpec("harmbench", TIER_A, "walledai/HarmBench", "standard", "train",
                             norm_harmbench, approx_rows=200, verified=True,
                             local_csv="datasets/harmbench/harmbench.csv"),
    "xstest": DatasetSpec("xstest", TIER_A, "walledai/XSTest", None, "test",
                          norm_xstest, approx_rows=450, verified=True,
                          local_csv="datasets/xstest/xstest.csv"),
    "toxicchat": DatasetSpec("toxicchat", TIER_A, "lmsys/toxic-chat", "toxicchat0124", "test",
                             norm_toxicchat, approx_rows=5083,
                             notes="real user traffic, naturally low toxic base rate"),
    "wildguardtest": DatasetSpec("wildguardtest", TIER_A, "allenai/wildguardmix", "wildguardtest", "test",
                                 norm_wildguard, approx_rows=1725, gated=True,
                                 notes="gated (auto-approve); accept terms on the hub first"),
    "openai_moderation": DatasetSpec("openai_moderation", TIER_A,
                                     "mmathys/openai-moderation-api-evaluation", None, "train",
                                     norm_openai_moderation, approx_rows=1680),
}

# The two committed to git, which every gate is computed on.  The other three
# scale the sample so Gate C's equivalence test has the power to return a
# verdict instead of UNDERPOWERED.
CORE_DATASETS = ["harmbench", "xstest"]


def gated_datasets() -> List[str]:
    return sorted(name for name, spec in DATASETS.items() if spec.gated)


def gated_urls() -> Dict[str, str]:
    """Dataset page per gated repo, deduplicated -- one click each."""
    urls = {}
    for name, spec in DATASETS.items():
        if spec.gated and spec.hf_id:
            urls.setdefault(spec.hf_id, spec.hub_url)
    return urls


def by_tier(tier: str) -> List[str]:
    return [name for name, spec in DATASETS.items() if spec.tier == tier]


def get_spec(name: str) -> DatasetSpec:
    key = name.lower()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Known: {sorted(DATASETS)}")
    return DATASETS[key]


def tier_summary() -> str:
    lines = []
    for tier in sorted(TIER_NAMES):
        lines.append(f"Tier {tier} - {TIER_NAMES[tier]}")
        for name in by_tier(tier):
            spec = DATASETS[name]
            mark = "verified" if spec.verified else "unverified"
            lines.append(f"  {name:<22} {spec.hf_id or 'local':<48} ~{spec.approx_rows:>7} rows  [{mark}]")
    return "\n".join(lines)
