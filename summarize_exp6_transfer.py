import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from updated_code.common import save_json

RESULTS_DIR = os.path.join(ROOT, "updated_code", "results")
DEFAULT_INPUTS = [
    os.path.join(RESULTS_DIR, "run_exp6_tetrapeptide_tps_seed42.json"),
    os.path.join(RESULTS_DIR, "run_exp6_tetrapeptide_tps_seed43.json"),
    os.path.join(RESULTS_DIR, "run_exp6_tetrapeptide_tps_seed44.json"),
]
DEFAULT_OUTPUT_JSON = os.path.join(RESULTS_DIR, "exp6_transfer_aggregate.json")
DEFAULT_PLOT_BASE = os.path.join(RESULTS_DIR, "exp6_transfer_aggregate")


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate Exp 6 transfer results across seeded held-out peptide panels.")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--plot-base", default=DEFAULT_PLOT_BASE)
    parser.add_argument(
        "--task-modes",
        nargs="+",
        default=None,
        help="Optional subset of task modes to aggregate, e.g. medium hard.",
    )
    return parser.parse_args()


def finite_values(records, key):
    values = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            values.append(float(value))
    return values


def summarize_values(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": int(arr.size),
    }


def maybe_float(value):
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None


def extract_panel_records(path, selected_modes=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transfer_setup = data["transfer_setup"]
    parameters = data["parameters"]
    panel_id = os.path.basename(path)

    task_modes = transfer_setup.get("state_pair_modes", [])
    if selected_modes is not None:
        task_modes = [mode for mode in task_modes if mode in selected_modes]

    records = []
    for peptide_name in transfer_setup.get("test_peptides", []):
        peptide_info = data["peptides"].get(peptide_name)
        if peptide_info is None:
            continue
        tasks = peptide_info.get("tasks", {})
        for task_mode in task_modes:
            task = tasks.get(task_mode)
            if task is None:
                continue
            gt = task["ground_truth"]
            for method, entry in task["methods"].items():
                log_z = entry.get("log_Z")
                ess = entry.get("ess")
                metrics = entry.get("metrics", {})
                log_z_mean = None if log_z is None else maybe_float(log_z.get("mean"))
                records.append(
                    {
                        "panel_id": panel_id,
                        "source_json": path,
                        "peptide": peptide_name,
                        "task_mode": task_mode,
                        "method": method,
                        "learned_gt": maybe_float(gt.get("learned_proposal_log_Z")),
                        "empirical_gt": maybe_float(gt.get("empirical_log_Z_from_reference_windows")),
                        "log_Z_mean": log_z_mean,
                        "log_Z_std": None if log_z is None else maybe_float(log_z.get("std")),
                        "ess_mean": None if ess is None else maybe_float(ess.get("mean")),
                        "valid_path_rate": maybe_float(metrics.get("valid_path_rate")),
                        "js_distance": maybe_float(metrics.get("js_distance")),
                        "js_divergence": maybe_float(metrics.get("js_divergence")),
                        "average_log_path_probability": maybe_float(metrics.get("average_log_path_probability")),
                    }
                )

    panel_summary = {
        "panel_id": panel_id,
        "source_json": path,
        "seed": data.get("seed", parameters.get("seed")),
        "test_peptides": transfer_setup.get("test_peptides", []),
        "task_modes": task_modes,
        "twist_variants": transfer_setup.get("twist_variants", []),
        "parameters": {
            "K": parameters.get("K"),
            "M": parameters.get("M"),
            "n_trials": parameters.get("n_trials"),
            "proposal_history_frames": parameters.get("proposal_history_frames"),
            "self_ref_windows": parameters.get("self_ref_windows"),
            "reward_mode": parameters.get("reward_mode"),
            "skip_oracle_twist": parameters.get("skip_oracle_twist"),
        },
    }
    return data, panel_summary, records


def panel_metric_summary(records, metric_key):
    panel_values = defaultdict(list)
    for record in records:
        value = record.get(metric_key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            panel_values[record["panel_id"]].append(float(value))
    summary = {}
    for panel_id, values in panel_values.items():
        summary[panel_id] = float(np.mean(np.asarray(values, dtype=np.float64)))
    return summary


def aggregate_records(records):
    task_modes = sorted({record["task_mode"] for record in records})
    methods = sorted({record["method"] for record in records})

    aggregated = {}
    for task_mode in task_modes:
        aggregated[task_mode] = {}
        task_records = [record for record in records if record["task_mode"] == task_mode]
        for method in methods:
            subset = [record for record in task_records if record["method"] == method]
            if not subset:
                continue

            learned_errors = [
                abs(record["log_Z_mean"] - record["learned_gt"])
                for record in subset
                if record["log_Z_mean"] is not None and record["learned_gt"] is not None
            ]
            empirical_errors = [
                abs(record["log_Z_mean"] - record["empirical_gt"])
                for record in subset
                if record["log_Z_mean"] is not None and record["empirical_gt"] is not None
            ]

            panel_summaries = {
                "log_Z_mean": panel_metric_summary(subset, "log_Z_mean"),
                "abs_log_Z_error_learned": panel_metric_summary(
                    [
                        dict(record, abs_log_Z_error_learned=abs(record["log_Z_mean"] - record["learned_gt"]))
                        for record in subset
                        if record["log_Z_mean"] is not None and record["learned_gt"] is not None
                    ],
                    "abs_log_Z_error_learned",
                ),
                "abs_log_Z_error_empirical": panel_metric_summary(
                    [
                        dict(record, abs_log_Z_error_empirical=abs(record["log_Z_mean"] - record["empirical_gt"]))
                        for record in subset
                        if record["log_Z_mean"] is not None and record["empirical_gt"] is not None
                    ],
                    "abs_log_Z_error_empirical",
                ),
                "valid_path_rate": panel_metric_summary(subset, "valid_path_rate"),
                "js_distance": panel_metric_summary(subset, "js_distance"),
            }

            aggregated[task_mode][method] = {
                "instance_count": int(len(subset)),
                "panel_count": int(len({record["panel_id"] for record in subset})),
                "unique_peptides": sorted({record["peptide"] for record in subset}),
                "log_Z_mean": summarize_values(finite_values(subset, "log_Z_mean")),
                "log_Z_std": summarize_values(finite_values(subset, "log_Z_std")),
                "ess_mean": summarize_values(finite_values(subset, "ess_mean")),
                "abs_log_Z_error_learned": summarize_values(learned_errors),
                "abs_log_Z_error_empirical": summarize_values(empirical_errors),
                "valid_path_rate": summarize_values(finite_values(subset, "valid_path_rate")),
                "js_distance": summarize_values(finite_values(subset, "js_distance")),
                "js_divergence": summarize_values(finite_values(subset, "js_divergence")),
                "average_log_path_probability": summarize_values(finite_values(subset, "average_log_path_probability")),
                "panel_means": {
                    metric: summarize_values(list(panel_values.values()))
                    for metric, panel_values in panel_summaries.items()
                },
                "panel_values": panel_summaries,
            }

    return aggregated


def build_comparisons(aggregated):
    comparisons = {}
    for task_mode, methods in aggregated.items():
        if "Bootstrap" not in methods or "Take3-SelfTwist" not in methods:
            continue
        bootstrap = methods["Bootstrap"]
        self_twist = methods["Take3-SelfTwist"]
        comparisons[task_mode] = {
            "selftwist_minus_bootstrap": {
                "abs_log_Z_error_learned_mean": (
                    None
                    if bootstrap["abs_log_Z_error_learned"] is None or self_twist["abs_log_Z_error_learned"] is None
                    else float(
                        self_twist["abs_log_Z_error_learned"]["mean"]
                        - bootstrap["abs_log_Z_error_learned"]["mean"]
                    )
                ),
                "abs_log_Z_error_empirical_mean": (
                    None
                    if bootstrap["abs_log_Z_error_empirical"] is None or self_twist["abs_log_Z_error_empirical"] is None
                    else float(
                        self_twist["abs_log_Z_error_empirical"]["mean"]
                        - bootstrap["abs_log_Z_error_empirical"]["mean"]
                    )
                ),
                "valid_path_rate_mean": (
                    None
                    if bootstrap["valid_path_rate"] is None or self_twist["valid_path_rate"] is None
                    else float(
                        self_twist["valid_path_rate"]["mean"]
                        - bootstrap["valid_path_rate"]["mean"]
                    )
                ),
                "js_distance_mean": (
                    None
                    if bootstrap["js_distance"] is None or self_twist["js_distance"] is None
                    else float(
                        self_twist["js_distance"]["mean"]
                        - bootstrap["js_distance"]["mean"]
                    )
                ),
            }
        }
    return comparisons


def plot_aggregate(plot_base, aggregated):
    os.makedirs(os.path.dirname(plot_base), exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(RESULTS_DIR, ".mplconfig"))
    import matplotlib.pyplot as plt

    metrics = [
        ("abs_log_Z_error_learned", "Abs err vs learned GT"),
        ("abs_log_Z_error_empirical", "Abs err vs empirical GT"),
        ("valid_path_rate", "Valid path rate"),
        ("js_distance", "Path JSD"),
    ]
    preferred_order = ["NaiveDiffusion", "Bootstrap", "Take3-SelfTwist"]
    colors = {
        "NaiveDiffusion": "#7f7f7f",
        "Bootstrap": "#1f77b4",
        "Take3-SelfTwist": "#ff7f0e",
    }

    task_modes = list(aggregated.keys())
    fig, axes = plt.subplots(
        len(task_modes),
        len(metrics),
        figsize=(4.0 * len(metrics), 3.5 * len(task_modes)),
        constrained_layout=True,
    )
    if len(task_modes) == 1:
        axes = np.asarray([axes])

    for row_idx, task_mode in enumerate(task_modes):
        methods = [method for method in preferred_order if method in aggregated[task_mode]]
        for col_idx, (metric_key, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for method_idx, method in enumerate(methods):
                method_summary = aggregated[task_mode][method]
                metric_summary = method_summary.get(metric_key)
                if metric_summary is None:
                    continue
                panel_metric = method_summary["panel_means"].get(metric_key)
                panel_values = method_summary["panel_values"].get(metric_key, {})
                mean = metric_summary["mean"]
                err = 0.0 if panel_metric is None else panel_metric["std"]
                ax.errorbar(
                    [method_idx],
                    [mean],
                    yerr=[err],
                    fmt="o",
                    color=colors.get(method, "#333333"),
                    capsize=4,
                    markersize=6,
                    linewidth=1.5,
                )
                for panel_value in panel_values.values():
                    ax.scatter(
                        [method_idx],
                        [panel_value],
                        color=colors.get(method, "#333333"),
                        alpha=0.35,
                        s=22,
                    )

            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=30, ha="right")
            ax.grid(alpha=0.25, axis="y")
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.set_ylabel(f"{task_mode.capitalize()} tasks")

    fig.savefig(plot_base + ".png", dpi=180, bbox_inches="tight")
    fig.savefig(plot_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    selected_modes = None if args.task_modes is None else set(args.task_modes)

    panel_summaries = []
    all_records = []
    consistency = defaultdict(set)
    used_inputs = []

    for path in args.inputs:
        if not os.path.exists(path):
            print(f"Skipping missing input: {path}")
            continue
        _, panel_summary, records = extract_panel_records(path, selected_modes=selected_modes)
        panel_summaries.append(panel_summary)
        all_records.extend(records)
        used_inputs.append(path)
        for key, value in panel_summary["parameters"].items():
            consistency[key].add(json.dumps(value, sort_keys=True))

    if not all_records:
        raise SystemExit("No records found for the selected Exp 6 inputs.")

    aggregated = aggregate_records(all_records)
    comparisons = build_comparisons(aggregated)
    consistency_report = {
        key: {
            "values": [json.loads(item) for item in sorted(values)],
            "consistent": len(values) == 1,
        }
        for key, values in consistency.items()
    }

    output = {
        "experiment": "exp6_transfer_aggregate",
        "input_files": used_inputs,
        "task_modes": sorted(aggregated.keys()),
        "panel_summaries": panel_summaries,
        "consistency_report": consistency_report,
        "records": all_records,
        "aggregated": aggregated,
        "comparisons": comparisons,
    }
    save_json(output, args.output_json)
    plot_aggregate(args.plot_base, aggregated)
    print(f"Saved Exp 6 aggregate JSON to {args.output_json}")
    print(f"Saved Exp 6 aggregate plots to {args.plot_base}.png/.pdf")


if __name__ == "__main__":
    main()
