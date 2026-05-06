"""
Preset sweep for Experiment 5 (alanine dipeptide).

This runner wraps `run_exp5_alanine.py` with a small family of increasingly
rare midpoint+endpoint tasks.
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from updated_code.common import save_json  # noqa: E402


DEFAULT_DATA_PATH = os.path.join(
    ROOT,
    "updated_code",
    "alanine-dipeptide-3x250ns-backbone-dihedrals.npz",
)

PRESETS = {
    "baseline": {
        "window_size": 25,
        "forward_horizon": 25,
        "reward_lambda": 6.0,
        "midpoint_lambda": 4.0,
        "midpoint_radius_deg": 35.0,
        "target_radius_deg": 30.0,
    },
    "rare_midpoint": {
        "window_size": 25,
        "forward_horizon": 25,
        "reward_lambda": 7.0,
        "midpoint_lambda": 6.0,
        "midpoint_radius_deg": 28.0,
        "target_radius_deg": 28.0,
    },
    "rare_endpoint": {
        "window_size": 25,
        "forward_horizon": 25,
        "reward_lambda": 8.0,
        "midpoint_lambda": 4.5,
        "midpoint_radius_deg": 35.0,
        "target_radius_deg": 22.0,
    },
    "balanced_bottleneck": {
        "window_size": 30,
        "forward_horizon": 30,
        "reward_lambda": 9.0,
        "midpoint_lambda": 7.0,
        "midpoint_radius_deg": 26.0,
        "target_radius_deg": 23.5,
    },
    "long_horizon": {
        "window_size": 30,
        "forward_horizon": 30,
        "reward_lambda": 8.0,
        "midpoint_lambda": 6.0,
        "midpoint_radius_deg": 30.0,
        "target_radius_deg": 25.0,
    },
    "bottleneck": {
        "window_size": 30,
        "forward_horizon": 30,
        "reward_lambda": 10.0,
        "midpoint_lambda": 8.0,
        "midpoint_radius_deg": 24.0,
        "target_radius_deg": 22.0,
    },
}

FULL_BUDGET = {
    "score_epochs": 10000,
    "twist_epochs": 10000,
    "hidden_dim": 384,
    "n_layers": 5,
    "max_pairs": 200000,
    "max_ref_windows": 50000,
    "max_forward_paths": 500,
    "max_transition_ref": 30000,
    "gt_samples": 10000,
    "n_trials": 8,
    "K": 192,
    "M": 16,
}

QUICK_BUDGET = {
    "score_epochs": 1200,
    "twist_epochs": 1200,
    "hidden_dim": 192,
    "n_layers": 3,
    "max_pairs": 25000,
    "max_ref_windows": 3000,
    "max_forward_paths": 120,
    "max_transition_ref": 800,
    "gt_samples": 1000,
    "n_trials": 2,
    "K": 64,
    "M": 8,
}


def parse_levels(spec):
    if isinstance(spec, (list, tuple)):
        raw_tokens = []
        for item in spec:
            raw_tokens.extend(str(item).split(","))
    else:
        raw_tokens = str(spec).split(",")

    levels = []
    for token in raw_tokens:
        token = token.strip().lower()
        if token:
            if token not in PRESETS:
                raise ValueError(f"Unknown Exp 5 sweep preset {token!r}.")
            levels.append(token)
    return levels


def summarize_child_result(payload):
    ground_truth = payload["endpoint_conditioning"]["ground_truth"]
    empirical_log_Z = float(ground_truth["empirical_log_Z_from_test_windows"])
    learned_log_Z = float(ground_truth["learned_proposal_log_Z"])
    counts = payload["dataset_counts"]
    methods = payload["endpoint_conditioning"]["methods"]

    out = {
        "task_support": {
            "test_start_windows": int(counts["test_start_windows"]),
            "test_transition_windows": int(counts["test_transition_windows"]),
            "transition_rate_given_start": float(
                counts["test_transition_windows"] / max(counts["test_start_windows"], 1)
            ),
        },
        "ground_truth": {
            "empirical_log_Z": empirical_log_Z,
            "learned_proposal_log_Z": learned_log_Z,
            "proposal_gap_nats": float(abs(empirical_log_Z - learned_log_Z)),
        },
        "methods": {},
    }

    take3_method_names = [
        name
        for name in ["Take3 (MC)", "Take3 (KL)", "Take3 (TD)", "Take3"]
        if name in methods
    ]
    method_names = ["NaiveDiffusion", "Bootstrap"] + take3_method_names

    for method_name in method_names:
        method_payload = methods[method_name]
        summary = method_payload["summary"]
        method_out = {
            "success_rate": float(summary["success_rate"]),
            "midpoint_success_rate": float(summary.get("midpoint_success_rate", 0.0)),
            "terminal_success_rate": float(summary.get("terminal_success_rate", 0.0)),
            "tica_jsd": None if summary["tica_fes"] is None else float(summary["tica_fes"]["js_distance"]),
            "ramachandran_jsd": None if summary["ramachandran"] is None else float(summary["ramachandran"]["js_distance"]),
        }
        if method_payload["log_Z"] is not None:
            log_Z_mean = float(method_payload["log_Z"]["mean"])
            method_out["log_Z_mean"] = log_Z_mean
            method_out["abs_logZ_error_empirical"] = float(abs(log_Z_mean - empirical_log_Z))
            method_out["abs_logZ_error_learned"] = float(abs(log_Z_mean - learned_log_Z))
        if method_payload["ess"] is not None:
            method_out["ess_mean"] = float(method_payload["ess"]["mean"])
        out["methods"][method_name] = method_out

    bootstrap = out["methods"]["Bootstrap"]
    primary_take3_name = "Take3 (MC)" if "Take3 (MC)" in out["methods"] else "Take3"
    if primary_take3_name not in out["methods"] and take3_method_names:
        primary_take3_name = take3_method_names[0]
    out["primary_take3_method"] = primary_take3_name
    take3 = out["methods"][primary_take3_name]
    out["take3_vs_bootstrap"] = {
        "empirical_logZ_error_improvement": float(
            bootstrap["abs_logZ_error_empirical"] - take3["abs_logZ_error_empirical"]
        ),
        "learned_logZ_error_improvement": float(
            bootstrap["abs_logZ_error_learned"] - take3["abs_logZ_error_learned"]
        ),
        "success_rate_improvement": float(take3["success_rate"] - bootstrap["success_rate"]),
        "tica_jsd_improvement": None
        if bootstrap["tica_jsd"] is None or take3["tica_jsd"] is None
        else float(bootstrap["tica_jsd"] - take3["tica_jsd"]),
        "ramachandran_jsd_improvement": None
        if bootstrap["ramachandran_jsd"] is None or take3["ramachandran_jsd"] is None
        else float(bootstrap["ramachandran_jsd"] - take3["ramachandran_jsd"]),
    }
    return out


def rank_presets(summaries, min_support_rate):
    ranking = []
    for preset_name, summary in summaries.items():
        support_rate = summary["task_support"]["transition_rate_given_start"]
        transition_count = int(summary["task_support"]["test_transition_windows"])
        gains = summary["take3_vs_bootstrap"]
        support_penalty = 0.0 if support_rate >= min_support_rate else (min_support_rate - support_rate) * 25.0
        transition_count_penalty = 0.08 * abs(transition_count - 8)
        if transition_count < 6:
            transition_count_penalty += 0.15 * (6 - transition_count)
        score = (
            gains["empirical_logZ_error_improvement"]
            + 0.5 * (gains["tica_jsd_improvement"] or 0.0)
            + 0.25 * gains["success_rate_improvement"]
            - support_penalty
            - transition_count_penalty
        )
        ranking.append(
            {
                "preset": preset_name,
                "score": float(score),
                "support_rate": float(support_rate),
                "test_transition_windows": transition_count,
                "proposal_gap_nats": float(summary["ground_truth"]["proposal_gap_nats"]),
                "take3_empirical_logZ_improvement": float(gains["empirical_logZ_error_improvement"]),
                "take3_tica_jsd_improvement": None
                if gains["tica_jsd_improvement"] is None
                else float(gains["tica_jsd_improvement"]),
                "take3_success_rate_improvement": float(gains["success_rate_improvement"]),
            }
        )
    ranking.sort(key=lambda item: item["score"], reverse=True)
    return ranking


def plot_sweep(plot_base, summaries):
    os.makedirs(os.path.dirname(plot_base), exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, "updated_code", "results", ".mplconfig"))
    import matplotlib.pyplot as plt

    preset_names = list(summaries.keys())
    support = [summaries[name]["task_support"]["transition_rate_given_start"] for name in preset_names]
    bootstrap = [summaries[name]["methods"]["Bootstrap"] for name in preset_names]
    take3 = [
        summaries[name]["methods"][summaries[name].get("primary_take3_method", "Take3")]
        for name in preset_names
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    panels = [
        ("abs_logZ_error_empirical", "Empirical |log Z error|"),
        ("success_rate", "Joint success rate"),
        ("tica_jsd", "TICA JSD"),
    ]
    colors = {"Bootstrap": "#1f77b4", "Take3": "#ff7f0e"}

    for ax, (metric, ylabel) in zip(axes, panels):
        y_boot = [entry[metric] for entry in bootstrap]
        y_take3 = [entry[metric] for entry in take3]
        ax.scatter(support, y_boot, s=70, color=colors["Bootstrap"], label="Bootstrap")
        ax.scatter(support, y_take3, s=70, color=colors["Take3"], label="Take3")
        for idx, preset_name in enumerate(preset_names):
            ax.annotate(preset_name, (support[idx], y_take3[idx]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Held-out transition rate")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    axes[0].legend(frameon=False, fontsize=9)
    fig.savefig(plot_base + ".png", dpi=180, bbox_inches="tight")
    fig.savefig(plot_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Rarity sweep for Exp 5 alanine tasks.")
    parser.add_argument(
        "--runner",
        default=os.path.join(ROOT, "updated_code", "run_exp5_alanine.py"),
        help="Path to the base Exp 5 runner.",
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Path to alanine trajectory data.",
    )
    parser.add_argument(
        "--output-path",
        default=os.path.join(ROOT, "updated_code", "results", "exp5_alanine_rarity_sweep.json"),
    )
    parser.add_argument(
        "--plot-base",
        default=os.path.join(ROOT, "updated_code", "results", "exp5_alanine_rarity_sweep"),
    )
    parser.add_argument(
        "--run-dir",
        default=os.path.join(ROOT, "updated_code", "results", "exp5_alanine_rarity_sweep_runs"),
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["baseline", "rare_midpoint", "rare_endpoint", "balanced_bottleneck", "long_horizon", "bottleneck"],
        help="Preset names to run; accepts space-separated values or comma-separated tokens.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Optional device override forwarded to the child runner.")
    parser.add_argument(
        "--take3-objectives",
        nargs="+",
        choices=["mc", "kl", "td"],
        default=["mc", "kl", "td"],
        help="Take 3 objectives forwarded to the child Exp 5 runner.",
    )
    parser.add_argument(
        "--twist-loss-space",
        choices=["linear", "log"],
        default="linear",
        help="MC/TD twist regression loss space forwarded to the child runner.",
    )
    parser.add_argument("--min-support-rate", type=float, default=0.005)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    levels = parse_levels(args.levels)
    budget = QUICK_BUDGET if args.quick else FULL_BUDGET

    os.makedirs(args.run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    summaries = {}
    for level in levels:
        preset = PRESETS[level].copy()
        child_json = os.path.join(args.run_dir, f"{level}.json")
        child_plot_dir = os.path.join(args.run_dir, level)
        cmd = [
            sys.executable,
            args.runner,
            "--data-path",
            args.data_path,
            "--output-json",
            child_json,
            "--plots-dir",
            child_plot_dir,
            "--task-mode",
            "midpoint_endpoint",
            "--window-size",
            str(preset["window_size"]),
            "--forward-horizon",
            str(preset["forward_horizon"]),
            "--reward-lambda",
            str(preset["reward_lambda"]),
            "--midpoint-lambda",
            str(preset["midpoint_lambda"]),
            "--midpoint-radius",
            str(preset["midpoint_radius_deg"] * 3.141592653589793 / 180.0),
            "--target-radius",
            str(preset["target_radius_deg"] * 3.141592653589793 / 180.0),
            "--score-epochs",
            str(budget["score_epochs"]),
            "--twist-epochs",
            str(budget["twist_epochs"]),
            "--hidden-dim",
            str(budget["hidden_dim"]),
            "--n-layers",
            str(budget["n_layers"]),
            "--max-pairs",
            str(budget["max_pairs"]),
            "--max-ref-windows",
            str(budget["max_ref_windows"]),
            "--max-forward-paths",
            str(budget["max_forward_paths"]),
            "--max-transition-ref",
            str(budget["max_transition_ref"]),
            "--gt-samples",
            str(budget["gt_samples"]),
            "--n-trials",
            str(budget["n_trials"]),
            "--K",
            str(budget["K"]),
            "--M",
            str(budget["M"]),
            "--take3-objectives",
            *args.take3_objectives,
            "--twist-loss-space",
            args.twist_loss_space,
            "--seed",
            str(args.seed),
        ]
        if args.device:
            cmd.extend(["--device", args.device])
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", os.path.join(args.run_dir, ".mplconfig"))

        if args.force or not os.path.exists(child_json):
            print(f"Running Exp 5 preset {level}: {' '.join(cmd)}")
            t0 = time.time()
            subprocess.run(cmd, check=True, cwd=ROOT, env=env)
            wall_clock_sec = time.time() - t0
        else:
            wall_clock_sec = None
            print(f"Reusing cached Exp 5 preset {level}: {child_json}")

        with open(child_json, "r", encoding="utf-8") as f:
            payload = json.load(f)

        summaries[level] = summarize_child_result(payload)
        summaries[level]["child_json"] = child_json
        summaries[level]["child_plot_dir"] = child_plot_dir
        summaries[level]["external_wall_clock_sec"] = wall_clock_sec
        summaries[level]["preset"] = preset
        summaries[level]["budget"] = budget

    ranking = rank_presets(summaries, min_support_rate=args.min_support_rate)
    plot_sweep(args.plot_base, summaries)

    output = {
        "experiment": "run_exp5_alanine_rarity_sweep",
        "runner": args.runner,
        "data_path": args.data_path,
        "quick": bool(args.quick),
        "min_support_rate": float(args.min_support_rate),
        "levels": levels,
        "ranking": ranking,
        "recommended_preset": None if not ranking else ranking[0]["preset"],
        "summaries": summaries,
    }
    save_json(output, args.output_path)
    print(f"Saved Exp 5 rarity sweep summary to {args.output_path}")


if __name__ == "__main__":
    main()
