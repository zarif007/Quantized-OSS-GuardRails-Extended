from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

TIER_A = "A"
TIER_B = "B"
TIER_C = "C"
TIER_D = "D"
TIER_E = "E"
TIER_F = "F"

TIER_NAMES = {
    TIER_A: "core prompt-level safety",
    TIER_B: "over-refusal / false positives",
    TIER_C: "fine-grained category and severity",
    TIER_D: "adversarial robustness",
    TIER_E: "multilingual",
    TIER_F: "response-level moderation",
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


def norm_wildguard_response(ds):
    out = []
    for r in ds:
        harm = (r.get("response_harm_label") or "").lower()
        if harm not in ("harmful", "unharmful"):
            continue
        row = _row(r.get("response"), "unsafe" if harm == "harmful" else "safe", r.get("subcategory"))
        row["context_prompt"] = r.get("prompt")
        out.append(row)
    return out


OPENAI_MOD_FLAGS = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]


def norm_openai_moderation(ds):
    out = []
    for r in ds:
        flags = [f for f in OPENAI_MOD_FLAGS if int(r.get(f, 0) or 0) == 1]
        out.append(_row(r.get("prompt"), "unsafe" if flags else "safe", ",".join(flags) or None))
    return out


def norm_beavertails(ds):
    out = []
    for r in ds:
        cats = [k for k, v in (r.get("category") or {}).items() if v]
        out.append(_row(r.get("prompt"), "safe" if r.get("is_safe") else "unsafe", ",".join(cats) or None))
    return out


def norm_beavertails_response(ds):
    out = []
    for r in ds:
        cats = [k for k, v in (r.get("category") or {}).items() if v]
        row = _row(r.get("response"), "safe" if r.get("is_safe") else "unsafe", ",".join(cats) or None)
        row["context_prompt"] = r.get("prompt")
        out.append(row)
    return out


def norm_orbench_benign(ds):
    return [_row(r.get("prompt"), "safe", r.get("category")) for r in ds]


def norm_orbench_toxic(ds):
    return [_row(r.get("prompt"), "unsafe", r.get("category")) for r in ds]


def norm_phtest(ds):
    return [_row(r.get("prompt"), "safe", r.get("harm_category") or r.get("category")) for r in ds]


def norm_sorrybench(ds):
    return [_row((r.get("turns") or [r.get("question")])[0] if isinstance(r.get("turns"), list) else r.get("question"),
                 "unsafe", str(r.get("category")))
            for r in ds]


def norm_saladbench(ds):
    return [_row(r.get("question"), "unsafe", r.get("2-category") or r.get("1-category")) for r in ds]


def norm_simplesafetytests(ds):
    return [_row(r.get("prompt") or r.get("prompts_final"), "unsafe", r.get("category") or r.get("harm_area"))
            for r in ds]


def norm_aegis(ds):
    out = []
    for r in ds:
        label = (r.get("labels_0") or r.get("text_type") or "").lower()
        gt = "safe" if "safe" in label and "unsafe" not in label else "unsafe"
        out.append(_row(r.get("text"), gt, label or None))
    return out


def norm_jailbreakbench(ds):
    return [_row(r.get("Goal") or r.get("goal"), "unsafe", r.get("Category") or r.get("category")) for r in ds]


def norm_strongreject(ds):
    return [_row(r.get("prompt") or r.get("forbidden_prompt"), "unsafe", r.get("category")) for r in ds]


def norm_inthewild(ds):
    return [_row(r.get("prompt"), "unsafe", "jailbreak") for r in ds]


def norm_multijail(ds):
    out = []
    langs = ["en", "zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"]
    for r in ds:
        for lang in langs:
            if r.get(lang):
                out.append(_row(r[lang], "unsafe", r.get("tags"), language=lang))
    return out


def norm_xsafety(ds):
    return [_row(r.get("text") or r.get("prompt"), "unsafe", r.get("category"),
                 language=r.get("language", "unknown")) for r in ds]


def norm_rtplx(ds):
    return [_row(r.get("Prompt") or r.get("prompt"), "unsafe", "toxicity",
                 language=r.get("Language") or r.get("language", "unknown")) for r in ds]


def norm_aya_redteam(ds):
    return [_row(r.get("prompt"), "unsafe", r.get("harm_category"), language=r.get("language", "unknown"))
            for r in ds]


def norm_lmsys_benign(ds):
    out = []
    for r in ds:
        conv = r.get("conversation") or []
        if conv and conv[0].get("role") == "user":
            out.append(_row(conv[0].get("content"), "safe", "benign_traffic"))
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
                                 norm_wildguard, approx_rows=1725,
                                 notes="gated dataset; accept terms on the hub first"),
    "openai_moderation": DatasetSpec("openai_moderation", TIER_A,
                                     "mmathys/openai-moderation-api-evaluation", None, "train",
                                     norm_openai_moderation, approx_rows=1680),
    "beavertails": DatasetSpec("beavertails", TIER_A, "PKU-Alignment/BeaverTails", None, "330k_test",
                               norm_beavertails, approx_rows=33000),
    "orbench_hard": DatasetSpec("orbench_hard", TIER_B, "bench-llm/or-bench", "or-bench-hard-1k", "train",
                                norm_orbench_benign, approx_rows=1000),
    "orbench_80k": DatasetSpec("orbench_80k", TIER_B, "bench-llm/or-bench", "or-bench-80k", "train",
                               norm_orbench_benign, approx_rows=80000),
    "orbench_toxic": DatasetSpec("orbench_toxic", TIER_B, "bench-llm/or-bench", "or-bench-toxic", "train",
                                 norm_orbench_toxic, approx_rows=655),
    "phtest": DatasetSpec("phtest", TIER_B, "furonghuang-lab/PHTest", None, "train",
                          norm_phtest, approx_rows=3260),
    "sorrybench": DatasetSpec("sorrybench", TIER_C, "sorry-bench/sorry-bench-202503", None, "train",
                              norm_sorrybench, approx_rows=440),
    "saladbench": DatasetSpec("saladbench", TIER_C, "OpenSafetyLab/Salad-Data", "base_set", "train",
                              norm_saladbench, approx_rows=21318),
    "simplesafetytests": DatasetSpec("simplesafetytests", TIER_C, "Bertievidgen/SimpleSafetyTests", None, "test",
                                     norm_simplesafetytests, approx_rows=100),
    "aegis": DatasetSpec("aegis", TIER_C, "nvidia/Aegis-AI-Content-Safety-Dataset-1.0", None, "test",
                         norm_aegis, approx_rows=1964),
    "jailbreakbench": DatasetSpec("jailbreakbench", TIER_D, "JailbreakBench/JBB-Behaviors", "behaviors", "harmful",
                                  norm_jailbreakbench, approx_rows=100),
    "strongreject": DatasetSpec("strongreject", TIER_D, "walledai/StrongREJECT", None, "train",
                                norm_strongreject, approx_rows=313),
    "inthewild_jailbreak": DatasetSpec("inthewild_jailbreak", TIER_D, "TrustAIRLab/in-the-wild-jailbreak-prompts",
                                       "jailbreak_2023_12_25", "train", norm_inthewild, approx_rows=1405),
    "multijail": DatasetSpec("multijail", TIER_E, "DAMO-NLP-SG/MultiJail", None, "train",
                             norm_multijail, approx_rows=3150),
    "xsafety": DatasetSpec("xsafety", TIER_E, "ToxicityPrompts/XSafety", None, "train",
                           norm_xsafety, approx_rows=28000),
    "rtplx": DatasetSpec("rtplx", TIER_E, "ToxicityPrompts/RTP-LX", None, "train",
                         norm_rtplx, approx_rows=28000),
    "aya_redteam": DatasetSpec("aya_redteam", TIER_E, "CohereForAI/aya_redteaming", None, "train",
                               norm_aya_redteam, approx_rows=7419),
    "wildguard_response": DatasetSpec("wildguard_response", TIER_F, "allenai/wildguardmix", "wildguardtest", "test",
                                      norm_wildguard_response, level="response", approx_rows=1725),
    "beavertails_response": DatasetSpec("beavertails_response", TIER_F, "PKU-Alignment/BeaverTails", None,
                                        "330k_test", norm_beavertails_response, level="response",
                                        approx_rows=33000),
    "lmsys_benign": DatasetSpec("lmsys_benign", TIER_A, "lmsys/lmsys-chat-1m", None, "train",
                                norm_lmsys_benign, approx_rows=1000000,
                                notes="gated; used only to build the realistic-traffic composite"),
}


def by_tier(tier: str) -> List[str]:
    return [name for name, spec in DATASETS.items() if spec.tier == tier]


def get_spec(name: str) -> DatasetSpec:
    key = name.lower()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Known: {sorted(DATASETS)}")
    return DATASETS[key]


def tier_summary() -> str:
    lines = []
    for tier in [TIER_A, TIER_B, TIER_C, TIER_D, TIER_E, TIER_F]:
        lines.append(f"Tier {tier} - {TIER_NAMES[tier]}")
        for name in by_tier(tier):
            spec = DATASETS[name]
            mark = "verified" if spec.verified else "unverified"
            lines.append(f"  {name:<22} {spec.hf_id or 'local':<48} ~{spec.approx_rows:>7} rows  [{mark}]")
    return "\n".join(lines)
