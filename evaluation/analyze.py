import argparse
import glob
import itertools
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.calibration import recalibrate, reliability_curve
from evaluation.categories import (
    category_degradation,
    error_decomposition,
    expected_cost,
    per_category,
    per_language,
    severity_weighted_risk,
)
from evaluation.deployment import (
    iso_memory_comparison,
    pareto_front,
    safety_per_gb,
    uncertainty_cascade,
)
from evaluation.disagreement import agreement_matrix, borderline_subset, flip_report
from evaluation.gates import (
    algorithm_axis_report,
    algorithm_divergence,
    base_rate_crossover,
    claims_table,
    gate_a,
    gate_c,
    gate_d,
    hardware_consistency,
)
from evaluation.metrics import calculate_metrics, calculate_threshold_free_metrics
from evaluation.statistical_tests import delong_test, holm_correction, mcnemar_exact
from evaluation.threshold_analysis import (
    BASE_RATES,
    FPR_TARGETS,
    accuracy_at_base_rate,
    rates_at,
    roc_curve,
    threshold_sweep,
    to_labels,
)
from models.registry import (
    ALGORITHM_AXIS_3BIT,
    ALGORITHM_AXIS_4BIT,
    DEFAULT_FAMILY,
    get_config,
    sort_keys,
)

BUDGETS_GB = [3.5, 4.0, 5.0, 6.0, 7.0, 9.0, 17.0]


def load_predictions(predictions_dir):
    files = sorted(glob.glob(os.path.join(predictions_dir, "predictions_*.csv")))
    if not files:
        raise FileNotFoundError(f"No prediction CSVs in {predictions_dir}")
    frames = [pd.read_csv(p) for p in files]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["prediction"] != "error"].reset_index(drop=True)
    if "prompt_id" not in combined.columns:
        combined["prompt_id"] = combined["dataset"].astype(str) + "_" + combined.index.astype(str)
    return combined


def has_scores(df):
    return "p_unsafe" in df.columns and df["p_unsafe"].notna().any()


def build_summary(combined, models):
    rows = []
    for model in models:
        sub = combined[combined["model"] == model]
        row = calculate_metrics(sub)
        row["model"] = model
        try:
            config = get_config(model)
            # config["precision"] is the quantization level ("q4_k_m"), which
            # would otherwise overwrite the classification precision metric.
            row.update({k: config[k] for k in ("family", "bits", "algorithm", "size_gb")})
            row["quant_precision"] = config["precision"]
        except Exception:
            pass
        if has_scores(sub):
            row.update(calculate_threshold_free_metrics(sub))
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_tests(combined, models):
    rows = []
    for a_key, b_key in itertools.combinations(models, 2):
        a = combined[combined["model"] == a_key].set_index("prompt_id")
        b = combined[combined["model"] == b_key].set_index("prompt_id")
        shared = a.index.intersection(b.index)
        if len(shared) == 0:
            continue
        a, b = a.loc[shared].sort_index(), b.loc[shared].sort_index()
        mc = mcnemar_exact(
            (a["prediction"] == a["ground_truth"]).values,
            (b["prediction"] == b["ground_truth"]).values,
        )
        row = {
            "model_a": a_key,
            "model_b": b_key,
            "n_paired": len(shared),
            "a_only_correct": mc["a_only_correct"],
            "b_only_correct": mc["b_only_correct"],
            "mcnemar_p": mc["p_value"],
        }
        if has_scores(a) and has_scores(b):
            dl = delong_test(to_labels(a["ground_truth"].values), a["p_unsafe"].values, b["p_unsafe"].values)
            row.update({"auroc_a": dl["auc_a"], "auroc_b": dl["auc_b"],
                        "auroc_delta": dl["delta"], "delong_p": dl["p_value"]})
        rows.append(row)

    df = pd.DataFrame(rows)
    for column, label in [("mcnemar_p", "mcnemar"), ("delong_p", "delong")]:
        if not df.empty and column in df.columns and df[column].notna().all():
            corrected = holm_correction(df[column].values)
            df[f"{label}_p_holm"] = [c["p_holm"] for c in corrected]
            df[f"{label}_significant"] = [c["significant"] for c in corrected]
    return df


def base_rate_table(combined, models):
    rows = []
    for model in models:
        sub = combined[combined["model"] == model]
        y = to_labels(sub["ground_truth"].values)
        scores = sub["p_unsafe"].values if has_scores(sub) else (sub["prediction"] == "unsafe").astype(float).values
        point = rates_at(y, scores, 0.5)
        row = {"model": model, "tpr": point["tpr"], "fpr": point["fpr"],
               "balanced_accuracy": point["balanced_accuracy"]}
        for pi in BASE_RATES:
            row[f"pi_{pi}"] = accuracy_at_base_rate(point["tpr"], point["fpr"], pi)
        rows.append(row)
    return pd.DataFrame(rows)


def recalibration_table(combined, models):
    rows = []
    for model in models:
        sub = combined[combined["model"] == model]
        if not has_scores(sub):
            continue
        y = to_labels(sub["ground_truth"].values)
        for target in FPR_TARGETS:
            row = recalibrate(y, sub["p_unsafe"].values, target)
            row["model"] = model
            rows.append(row)
    return pd.DataFrame(rows)


def per_dataset_table(combined, models):
    rows = []
    for (model, dataset), group in combined.groupby(["model", "dataset"]):
        y = to_labels(group["ground_truth"].values)
        scores = group["p_unsafe"].values if has_scores(group) else (group["prediction"] == "unsafe").astype(float).values
        point = rates_at(y, scores, 0.5)
        rows.append({
            "model": model, "dataset": dataset, "n": len(group),
            "harmful_base_rate": float((group["ground_truth"] == "unsafe").mean()),
            "tpr": point["tpr"], "fpr": point["fpr"], "f1": point["f1"],
            "balanced_accuracy": point["balanced_accuracy"],
            "auroc": None if not has_scores(group) else calculate_threshold_free_metrics(group).get("auroc"),
        })
    return pd.DataFrame(rows)


def save(df, tables_dir, name):
    if isinstance(df, pd.DataFrame) and not df.empty:
        df.to_csv(os.path.join(tables_dir, f"{name}.csv"), index=False)


def plot_roc_overlay(combined, models, out_dir, zoom=False):
    plt.figure(figsize=(6, 5.5))
    for model in models:
        sub = combined[combined["model"] == model]
        if not has_scores(sub):
            continue
        fpr, tpr, _ = roc_curve(to_labels(sub["ground_truth"].values), sub["p_unsafe"].values)
        plt.plot(fpr, tpr, label=model, linewidth=1.3)
    if zoom:
        plt.xlim(0, 0.15)
    else:
        plt.plot([0, 1], [0, 1], "k--", linewidth=0.7)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC, low-FPR region" if zoom else "ROC by precision")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_zoom_low_fpr.png" if zoom else "roc_overlay.png"), dpi=160)
    plt.close()


def plot_threshold_sweep(combined, models, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for model in models:
        sub = combined[combined["model"] == model]
        if not has_scores(sub):
            continue
        sweep = pd.DataFrame(threshold_sweep(to_labels(sub["ground_truth"].values), sub["p_unsafe"].values))
        axes[0].plot(sweep["threshold"], sweep["f1"], label=model)
        axes[1].plot(sweep["tpr"], sweep["precision"], label=model)
        axes[2].plot(sweep["fpr"], sweep["fnr"], label=model)
    for ax, (xl, yl, title) in zip(axes, [("threshold", "F1", "F1 vs threshold"),
                                          ("recall", "precision", "Precision-recall"),
                                          ("FPR", "FNR", "DET")]):
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "threshold_sweep.png"), dpi=160)
    plt.close()


def plot_base_rate(base_df, out_dir):
    plt.figure(figsize=(7, 5))
    for _, row in base_df.iterrows():
        plt.plot(BASE_RATES, [row[f"pi_{pi}"] for pi in BASE_RATES], marker="o", label=row["model"])
    plt.xscale("log")
    plt.xlabel("Harmful base rate")
    plt.ylabel("Accuracy")
    plt.title("Ranking depends on base rate")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "base_rate_sensitivity.png"), dpi=160)
    plt.close()


def plot_margins(combined, models, out_dir):
    groups = [
        ("correct harmful", "unsafe", "unsafe"),
        ("missed harmful", "unsafe", "safe"),
        ("correct benign", "safe", "safe"),
        ("false alarm", "safe", "unsafe"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (title, truth, pred) in zip(axes.flat, groups):
        for model in models:
            sub = combined[(combined["model"] == model) &
                           (combined["ground_truth"] == truth) &
                           (combined["prediction"] == pred)]
            if "margin" not in sub.columns or len(sub) < 2:
                continue
            ax.hist(sub["margin"].dropna().values, bins=40, histtype="step", label=model, density=True)
        ax.set_title(title)
        ax.set_xlabel("logit_unsafe - logit_safe")
        ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "margin_distributions.png"), dpi=160)
    plt.close()


def plot_reliability(combined, models, out_dir):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.7)
    for model in models:
        sub = combined[combined["model"] == model]
        if not has_scores(sub):
            continue
        curve = pd.DataFrame(reliability_curve(to_labels(sub["ground_truth"].values), sub["p_unsafe"].values))
        if not curve.empty:
            plt.plot(curve["mean_predicted"], curve["observed_frequency"], marker="o", label=model)
    plt.xlabel("Mean predicted p(unsafe)")
    plt.ylabel("Observed frequency")
    plt.title("Reliability")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reliability.png"), dpi=160)
    plt.close()


def plot_flip_distance(distance_df, out_dir):
    if distance_df.empty:
        return
    plt.figure(figsize=(7, 5))
    for target, group in distance_df.groupby("target"):
        plt.plot(group["mean_distance"], group["flip_rate"], marker="o", label=target)
    plt.xlabel("Distance of reference p(unsafe) from threshold")
    plt.ylabel("Flip rate")
    plt.title("Flips concentrate near the boundary")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "flip_rate_vs_distance.png"), dpi=160)
    plt.close()


def plot_pareto(front, out_dir):
    if front.empty:
        return
    plt.figure(figsize=(7, 5))
    plt.scatter(front["memory_gb"], front["tpr_at_target_fpr"], c="steelblue")
    edge = front[front["on_pareto_front"]]
    plt.plot(edge["memory_gb"], edge["tpr_at_target_fpr"], "r--", linewidth=1)
    for _, row in front.iterrows():
        plt.annotate(str(row["model"]).split(":")[-1], (row["memory_gb"], row["tpr_at_target_fpr"]),
                     fontsize=6, xytext=(3, 3), textcoords="offset points")
    plt.xlabel("Memory (GB)")
    plt.ylabel("TPR @ FPR=0.05")
    plt.title("Safety per memory budget")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "memory_pareto.png"), dpi=160)
    plt.close()


def plot_cascade(cascade, out_dir):
    if cascade.empty:
        return
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(cascade["fraction_escalated"], cascade["tpr"], marker="o", color="steelblue", label="TPR")
    ax1.set_xlabel("Fraction escalated to reference model")
    ax1.set_ylabel("TPR", color="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(cascade["fraction_escalated"], cascade["fpr"], marker="s", color="indianred", label="FPR")
    ax2.set_ylabel("FPR", color="indianred")
    plt.title("Uncertainty-gated cascade")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "uncertainty_cascade.png"), dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Full analysis across all phases")
    parser.add_argument("--predictions-dir", default="results/predictions")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--reference-model", default=None)
    parser.add_argument("--fast-model", default=None)
    parser.add_argument("--layer-sweep", default="results/mechanism/layer_sensitivity.csv")
    parser.add_argument("--require-same-hardware", action="store_true",
                        help="abort if prediction files span more than one environment")
    parser.add_argument("--memory-col", default="total_memory_mb",
                        choices=["total_memory_mb", "vram_mb", "weights_mb"],
                        help="memory basis for the deployment tables")
    args = parser.parse_args()

    os.makedirs(args.tables_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    combined = load_predictions(args.predictions_dir)
    models = sort_keys(list(combined["model"].unique()))
    scored = has_scores(combined)
    reference = args.reference_model or models[0]

    print(f"Loaded {len(combined)} rows | {len(models)} models | scores={scored}")
    print(f"Datasets: {sorted(combined['dataset'].unique())}")
    print(f"Reference model: {reference}")

    # Efficiency metrics are only comparable across precisions when every
    # other element of the configuration was held fixed.  Verify that before
    # writing any table that divides by latency or memory.
    hardware = hardware_consistency(combined)
    efficiency_ok = hardware.get("efficiency_metrics_comparable", False)
    print(f"\nHardware: {hardware['status']} — {hardware['reason']}")
    for env in hardware.get("environments", []):
        print(f"  {env['env_hash']}  {env['backend']}  {env['gpu_name'] or 'cpu'}  "
              f"{env['n_rows']} rows  {len(env['models'])} models")
    if not efficiency_ok:
        print("  -> deployment tables will be written but flagged NOT COMPARABLE")
    save(pd.DataFrame(hardware.get("environments", [])), args.tables_dir, "environments")

    if args.require_same_hardware and not efficiency_ok:
        print("\nAborting: --require-same-hardware was set and the check did not pass.")
        sys.exit(4)

    summary = build_summary(combined, models)
    save(summary, args.tables_dir, "summary_metrics")

    pairs = pairwise_tests(combined, models)
    save(pairs, args.tables_dir, "pairwise_tests")

    base_df = base_rate_table(combined, models)
    save(base_df, args.tables_dir, "base_rate_sensitivity")

    save(per_dataset_table(combined, models), args.tables_dir, "per_dataset")
    save(error_decomposition(combined), args.tables_dir, "error_decomposition")
    save(severity_weighted_risk(combined), args.tables_dir, "severity_weighted_risk")
    save(expected_cost(combined), args.tables_dir, "expected_cost")

    cat_df = per_category(combined)
    save(cat_df, args.tables_dir, "per_category")
    save(category_degradation(cat_df, reference), args.tables_dir, "category_degradation")
    save(per_language(combined), args.tables_dir, "per_language")

    save(agreement_matrix(combined, models), args.tables_dir, "agreement_matrix")
    flip_sum, flips, flip_dist = flip_report(combined, models, reference)
    save(flip_sum, args.tables_dir, "flip_summary")
    save(flip_dist, args.tables_dir, "flip_rate_by_distance")
    save(borderline_subset(combined, models), args.tables_dir, "borderline_examples")

    recal = recalibration_table(combined, models) if scored else pd.DataFrame()
    save(recal, args.tables_dir, "recalibration")

    deploy = safety_per_gb(combined, memory_col=args.memory_col) if scored else pd.DataFrame()
    if not deploy.empty:
        deploy["efficiency_comparable"] = efficiency_ok
        deploy["memory_basis"] = args.memory_col
    save(deploy, args.tables_dir, "safety_per_gb")
    front = pareto_front(deploy) if not deploy.empty else pd.DataFrame()
    save(front, args.tables_dir, "memory_pareto")
    save(iso_memory_comparison(combined, BUDGETS_GB) if scored else pd.DataFrame(),
         args.tables_dir, "iso_memory")

    fast = args.fast_model or (models[-1] if len(models) > 1 else None)
    cascade = pd.DataFrame()
    if scored and fast and fast != reference:
        cascade = uncertainty_cascade(combined, fast, reference)
        save(cascade, args.tables_dir, "uncertainty_cascade")

    verdicts = {"gate_a": gate_a(pairs, summary), "gate_c": gate_c(summary),
                "gate_d": gate_d(recal) if not recal.empty else {"gate": "D", "status": "NOT_EVALUABLE"}}
    verdicts["hardware_consistency"] = hardware
    crossover = base_rate_crossover(base_df, BASE_RATES)

    sweep = pd.read_csv(args.layer_sweep) if os.path.exists(args.layer_sweep) else pd.DataFrame()
    from evaluation.gates import layer_concentration

    # Claim 2: at a fixed bit budget, does the algorithm change the decision?
    # Both axes are evaluated; the 3-bit one previously had no test at all.
    axis_tables, axis_verdicts = [], {}
    for axis_name, axis_precisions in (("4bit", ALGORITHM_AXIS_4BIT),
                                       ("3bit", ALGORITHM_AXIS_3BIT)):
        axis_keys = [f"{DEFAULT_FAMILY}:{p}" for p in axis_precisions]
        table, verdict = algorithm_axis_report(summary, pairs, axis_keys, axis_name)
        axis_verdicts[axis_name] = verdict
        if not table.empty:
            axis_tables.append(table)
    if axis_tables:
        save(pd.concat(axis_tables, ignore_index=True), args.tables_dir, "algorithm_axis")

    algo_keys = [f"{DEFAULT_FAMILY}:{p}" for p in ALGORITHM_AXIS_4BIT]
    evidence = {
        "gate_a": verdicts["gate_a"]["status"],
        "gate_c": verdicts["gate_c"]["status"],
        "gate_d": verdicts["gate_d"]["status"],
        "base_rate_crossover": crossover["crossover_observed"],
        "realistic_traffic_reversal": crossover["crossover_observed"]
        if "realistic_traffic" in set(combined["dataset"]) else None,
        "algorithm_axis_4bit": axis_verdicts["4bit"]["status"],
        "algorithm_axis_3bit": axis_verdicts["3bit"]["status"],
        "layer_concentration": layer_concentration(sweep).get("concentrated"),
        "imatrix_divergence": None,
        "mixed_precision_pareto": None,
    }
    claims = claims_table(evidence)
    save(claims, args.tables_dir, "claims_to_evidence")

    if scored:
        plot_roc_overlay(combined, models, args.figures_dir)
        plot_roc_overlay(combined, models, args.figures_dir, zoom=True)
        plot_threshold_sweep(combined, models, args.figures_dir)
        plot_margins(combined, models, args.figures_dir)
        plot_reliability(combined, models, args.figures_dir)
        plot_flip_distance(flip_dist, args.figures_dir)
        plot_pareto(front, args.figures_dir)
        plot_cascade(cascade, args.figures_dir)
    plot_base_rate(base_df, args.figures_dir)

    report = {"models": models, "datasets": sorted(combined["dataset"].unique()),
              "n_rows": len(combined), "scored": scored, "reference_model": reference,
              "efficiency_metrics_comparable": efficiency_ok,
              "memory_basis": args.memory_col,
              **verdicts, "algorithm_axis": axis_verdicts,
              "base_rate_crossover": crossover, "evidence": evidence}
    with open(os.path.join(args.tables_dir, "gates.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print("\n=== GATES ===")
    for key in ("gate_a", "gate_c", "gate_d"):
        v = verdicts[key]
        print(f"  {v['gate']}: {v['status']:<22} {v.get('reason','')}")
    print(f"  HW: {hardware['status']:<22} {hardware['reason']}")

    print("\n=== Algorithm axis (same bits, different method) ===")
    for axis_name, v in axis_verdicts.items():
        print(f"  {axis_name}: {v['status']:<16} {v['reason']}")
        if v["status"] != "NOT_TESTED":
            print(f"        {v['n_algorithms']} algorithms, {v['n_pairs_tested']} pairs | "
                  f"TPR spread {v.get('tpr_at_fpr_05_spread', v.get('safety_rate_spread', float('nan'))):.4f} | "
                  f"bits spread {v['bits_spread']:.2f}")

    print("\n=== Base-rate winners ===")
    for pi, winner in crossover["winners"].items():
        print(f"  base rate {pi:<5} -> {winner}")
    print(f"  crossover observed: {crossover['crossover_observed']}")

    if scored:
        cols = [c for c in ["model", "auroc", "auroc_ci_lo", "auroc_ci_hi",
                            "tpr_at_fpr_05", "flag_rate", "ece", "temperature"]
                if c in summary.columns]
        print("\n=== Threshold-free summary ===")
        print(summary[cols].to_string(index=False))

    print("\n=== Claims ===")
    print(claims[["claim", "status"]].to_string(index=False))

    print(f"\nTables  -> {args.tables_dir}")
    print(f"Figures -> {args.figures_dir}")


if __name__ == "__main__":
    main()
