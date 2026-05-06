"""
Alanine dipeptide experiment scaffold inspired by the MDGen evaluation style.

This runner assumes precomputed trajectories saved as `.npz` or `.npy`.
Supported trajectory payloads:
  - `angles`: (n_traj, n_frames, 2) with phi/psi in radians or degrees
  - `phi` and `psi`: (n_traj, n_frames)
  - `features`: (n_traj, n_frames, 4) storing cos/sin embeddings
  - `positions`: (n_traj, n_frames, n_atoms, 3) plus `phi_indices`, `psi_indices`

Outputs:
  - JSON summary with metrics and configuration
  - Ramachandran and TICA free-energy comparison plots
"""

import argparse
import os
import sys
import time
import numpy as np
from scipy.special import logsumexp
try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "run_exp5_alanine.py requires PyTorch. Install torch to train the alanine score and twist models."
    ) from exc

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXPERIMENTS_DIR = os.path.join(ROOT, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

from updated_code.learned_scores import (
    train_conditional_score,
    NNConditionalScore,
)

from updated_code.common import (
    systematic_resample,
    compute_ess,
    build_future_reward_targets,
    sample_row_indices,
    mean_std,
    save_json,
)
from updated_code.fixed_twist import (
    train_positive_twist_kl,
    train_positive_twist_mc,
    train_positive_twist_td,
    PositiveNNTwist,
)
from updated_code.alanine import (
    DEFAULT_ALPHA_R,
    DEFAULT_C7AX,
    DEFAULT_C7EQ,
    angles_to_features,
    basin_mask,
    circular_mean,
    compute_free_energy_surface,
    features_to_angles,
    fit_tica,
    histogram_js_distance_2d,
    load_alanine_trajectories,
    make_windows,
    normalize_torsion_features,
    plot_free_energy_comparison,
    plot_histogram_comparison,
    split_train_test_temporal,
    squared_angular_distance,
    subsample_rows,
    tica_transform,
)


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_OUTPUT_PATH = os.path.join(RESULTS_DIR, "run_exp5_alanine.json")
DEFAULT_PLOTS_DIR = os.path.join(RESULTS_DIR, "alanine")


def parse_args():
    parser = argparse.ArgumentParser(description="Alanine dipeptide endpoint-conditioning experiment.")
    parser.add_argument("--data-path", required=True, help="Path to precomputed alanine trajectory data (.npz/.npy).")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_PATH, help="Path to JSON results.")
    parser.add_argument("--plots-dir", default=DEFAULT_PLOTS_DIR, help="Directory for plots.")
    parser.add_argument("--device", default=DEVICE, help="Torch device.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--window-size", type=int, default=25, help="Endpoint-conditioning horizon T.")
    parser.add_argument("--forward-horizon", type=int, default=None, help="Short rollout horizon for forward evaluation; defaults to window-size.")
    parser.add_argument("--window-stride", type=int, default=5)
    parser.add_argument("--tica-lag", type=int, default=10)
    parser.add_argument("--score-epochs", type=int, default=5000)
    parser.add_argument("--twist-epochs", type=int, default=5000)
    parser.add_argument(
        "--take3-objectives",
        nargs="+",
        choices=["mc", "kl", "td"],
        default=["mc", "kl", "td"],
        help="Take 3 twist objectives to train and evaluate.",
    )
    parser.add_argument(
        "--twist-loss-space",
        choices=["linear", "log"],
        default="linear",
        help="Regression loss space for MC and TD twist objectives.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument("--max-ref-windows", type=int, default=10000)
    parser.add_argument("--max-forward-paths", type=int, default=400)
    parser.add_argument("--max-transition-ref", type=int, default=2000)
    parser.add_argument("--gt-samples", type=int, default=5000)
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--S-max", type=float, default=1.5)
    parser.add_argument("--n-diff-steps", type=int, default=40)
    parser.add_argument("--reward-lambda", type=float, default=6.0)
    parser.add_argument("--midpoint-lambda", type=float, default=4.0)
    parser.add_argument("--start-radius", type=float, default=np.deg2rad(35.0))
    parser.add_argument("--midpoint-radius", type=float, default=np.deg2rad(35.0))
    parser.add_argument("--target-radius", type=float, default=np.deg2rad(30.0))
    parser.add_argument("--task-mode", choices=["endpoint", "midpoint_endpoint"], default="midpoint_endpoint")
    parser.add_argument("--start-center-deg", nargs=2, type=float, default=None)
    parser.add_argument("--midpoint-center-deg", nargs=2, type=float, default=None)
    parser.add_argument("--target-center-deg", nargs=2, type=float, default=None)
    return parser.parse_args()


def resolve_centers(args):
    start_center = DEFAULT_C7EQ if args.start_center_deg is None else np.deg2rad(np.asarray(args.start_center_deg, dtype=np.float64))
    midpoint_center = DEFAULT_ALPHA_R if args.midpoint_center_deg is None else np.deg2rad(np.asarray(args.midpoint_center_deg, dtype=np.float64))
    target_center = DEFAULT_ALPHA_R if args.target_center_deg is None else np.deg2rad(np.asarray(args.target_center_deg, dtype=np.float64))
    return start_center, midpoint_center, target_center


def conditional_score_eval(o_score, x_batch, s, x_cond_batch):
    x_arr = np.asarray(x_batch, dtype=np.float64).reshape(-1, 4)
    x_cond_arr = np.asarray(x_cond_batch, dtype=np.float64)
    if x_cond_arr.ndim == 1:
        x_cond_arr = x_cond_arr.reshape(1, 4)
    else:
        x_cond_arr = x_cond_arr.reshape(-1, 4)
    if len(x_cond_arr) == 1 and len(x_arr) > 1:
        x_cond_arr = np.tile(x_cond_arr, (len(x_arr), 1))
    elif len(x_cond_arr) != len(x_arr):
        raise ValueError("x_batch and x_cond_batch must have matching batch sizes.")

    with torch.no_grad():
        inp = torch.tensor(
            np.concatenate([x_arr, x_cond_arr], axis=-1),
            dtype=torch.float32,
            device=o_score.device,
        )
        st = torch.full((len(x_arr),), float(s), dtype=torch.float32, device=o_score.device)
        out = o_score.model(inp, st).cpu().numpy()
    return np.clip(out, -o_score.clip, o_score.clip)


def rollout_reverse_step(o_score, x_cond, n_samples, s_max, n_steps):
    dim = 4
    x_cond_arr = np.asarray(x_cond, dtype=np.float64)
    if x_cond_arr.ndim == 1:
        x_cond_arr = x_cond_arr.reshape(1, dim)
    else:
        x_cond_arr = x_cond_arr.reshape(-1, dim)
    n_cond = len(x_cond_arr)
    x_cond_rep = np.repeat(x_cond_arr, n_samples, axis=0)
    ds = s_max / n_steps
    x = np.random.randn(len(x_cond_rep), dim)
    x = normalize_torsion_features(x)
    for step in range(n_steps):
        s = max(s_max - step * ds, 1e-6)
        score = conditional_score_eval(o_score, x, s, x_cond_rep).reshape(len(x_cond_rep), dim)
        score = np.clip(np.nan_to_num(score, nan=0.0), -12.0, 12.0)
        drift = np.clip(0.5 * x + score, -12.0, 12.0)
        x = x + drift * ds + np.random.randn(len(x_cond_rep), dim) * np.sqrt(ds)
        x = normalize_torsion_features(x)
        x = np.clip(x, -1.5, 1.5)
    x = normalize_torsion_features(x)
    if n_cond == 1:
        return x.reshape(n_samples, dim)
    return x.reshape(n_cond, n_samples, dim)


def sample_forward_paths(initial_features, horizon, reverse_step_fn):
    n_paths = len(initial_features)
    traj = np.zeros((n_paths, horizon + 1, 4), dtype=np.float64)
    traj[:, 0] = initial_features
    for t in range(horizon):
        traj[:, t + 1] = reverse_step_fn(traj[:, t], 1).reshape(n_paths, 4)
    return traj


def compute_eval_range(samples_a, samples_b, margin=0.05):
    stacked = np.concatenate([samples_a, samples_b], axis=0)
    mins = np.min(stacked, axis=0)
    maxs = np.max(stacked, axis=0)
    width = np.maximum(maxs - mins, 1e-6)
    mins = mins - margin * width
    maxs = maxs + margin * width
    return [[float(mins[0]), float(maxs[0])], [float(mins[1]), float(maxs[1])]]


def summarize_forward_model(reference_paths, model_paths, tica_model, plots_dir, prefix):
    ref_angles = features_to_angles(reference_paths.reshape(-1, 4))
    model_angles = features_to_angles(model_paths.reshape(-1, 4))

    rama_range = [[-np.pi, np.pi], [-np.pi, np.pi]]
    rama_stats = histogram_js_distance_2d(ref_angles, model_angles, bins=64, value_range=rama_range)
    ref_rama = compute_free_energy_surface(ref_angles, bins=64, value_range=rama_range)
    model_rama = compute_free_energy_surface(model_angles, bins=64, value_range=rama_range)
    plot_histogram_comparison(
        ref_rama,
        model_rama,
        os.path.join(plots_dir, f"{prefix}_ramachandran"),
        f"{prefix.replace('_', ' ').title()} Ramachandran",
        axis_labels=(r"$\phi$", r"$\psi$"),
    )

    ref_tica = tica_transform(reference_paths.reshape(-1, 4), tica_model)
    model_tica = tica_transform(model_paths.reshape(-1, 4), tica_model)
    tica_range = compute_eval_range(ref_tica[:, :2], model_tica[:, :2])
    tica_stats = histogram_js_distance_2d(ref_tica[:, :2], model_tica[:, :2], bins=50, value_range=tica_range)
    ref_tica_fes = compute_free_energy_surface(ref_tica[:, :2], bins=50, value_range=tica_range)
    model_tica_fes = compute_free_energy_surface(model_tica[:, :2], bins=50, value_range=tica_range)
    plot_free_energy_comparison(
        ref_tica_fes,
        model_tica_fes,
        os.path.join(plots_dir, f"{prefix}_tica"),
        f"{prefix.replace('_', ' ').title()} TICA FES",
        axis_labels=("TIC-0", "TIC-1"),
    )

    return {
        "ramachandran": {
            "js_distance": rama_stats["js_distance"],
            "js_divergence": rama_stats["js_divergence"],
        },
        "tica_fes": {
            "js_distance": tica_stats["js_distance"],
            "js_divergence": tica_stats["js_divergence"],
        },
    }


def checkpoint_log_G_fn(checkpoints):
    checkpoint_map = {int(spec["time"]): spec for spec in checkpoints}

    def checkpoint_log_G(features, t):
        features = np.asarray(features, dtype=np.float64).reshape(-1, 4)
        spec = checkpoint_map.get(int(t))
        if spec is None:
            return np.zeros(len(features), dtype=np.float64)
        angles = features_to_angles(features)
        dist2 = squared_angular_distance(angles, spec["center"])
        return -float(spec["lambda"]) * dist2

    return checkpoint_log_G


def path_log_reward(paths, checkpoint_log_G):
    paths = np.asarray(paths, dtype=np.float64)
    horizon = paths.shape[1] - 1
    total = np.zeros(len(paths), dtype=np.float64)
    for t in range(1, horizon + 1):
        total += checkpoint_log_G(paths[:, t], t)
    return total


def estimate_log_Z_from_chain(start_features, horizon, reverse_step_fn, endpoint_log_G, n_samples, batch_size, rng):
    start_features = np.asarray(start_features, dtype=np.float64).reshape(-1, 4)
    log_rewards = np.zeros(n_samples, dtype=np.float64)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = end - start
        idx = rng.choice(len(start_features), size=batch, replace=True)
        paths = sample_forward_paths(start_features[idx], horizon, reverse_step_fn)
        log_rewards[start:end] = path_log_reward(paths, endpoint_log_G)
    return float(logsumexp(log_rewards) - np.log(n_samples))


def run_bootstrap(start_features, horizon, K, reverse_step_fn, endpoint_log_G):
    start_features = np.asarray(start_features, dtype=np.float64)
    if start_features.ndim == 1:
        start_features = np.tile(start_features, (K, 1))
    else:
        start_features = start_features.reshape(K, 4)
    traj = np.zeros((K, horizon + 1, 4), dtype=np.float64)
    traj[:, 0] = start_features
    ess_per_step = []
    log_Z = 0.0
    for t in range(1, horizon + 1):
        traj[:, t] = reverse_step_fn(traj[:, t - 1], 1).reshape(K, 4)
        log_w = endpoint_log_G(traj[:, t], t)
        log_Z += logsumexp(log_w) - np.log(K)
        ess_per_step.append(compute_ess(log_w))
        indices = systematic_resample(log_w)
        traj = traj[indices]
    return {
        "log_Z": float(log_Z),
        "ess_per_step": ess_per_step,
        "paths": traj,
    }


def run_take3(start_features, horizon, K, M, reverse_step_fn, endpoint_log_G, twist_fn):
    start_features = np.asarray(start_features, dtype=np.float64)
    if start_features.ndim == 1:
        start_features = np.tile(start_features, (K, 1))
    else:
        start_features = start_features.reshape(K, 4)
    traj = np.zeros((K, horizon + 1, 4), dtype=np.float64)
    traj[:, 0] = start_features
    log_psi_prev = np.asarray(twist_fn(start_features, 0), dtype=np.float64).reshape(K)
    log_Z = float(logsumexp(log_psi_prev) - np.log(K))
    ess_per_step = []

    for t in range(1, horizon + 1):
        proposals = reverse_step_fn(traj[:, t - 1], M)
        log_G = endpoint_log_G(proposals.reshape(-1, 4), t).reshape(K, M)
        if t < horizon:
            log_twist = np.asarray(twist_fn(proposals.reshape(-1, 4), t), dtype=np.float64).reshape(K, M)
        else:
            log_twist = np.zeros((K, M), dtype=np.float64)
        log_v = log_G + log_twist
        j_sel, log_norm = sample_row_indices(log_v, n_draws=1)
        traj[:, t] = proposals[np.arange(K), j_sel]
        log_weights = log_norm - np.log(M) - log_psi_prev
        log_psi_next = log_twist[np.arange(K), j_sel] if t < horizon else np.zeros(K, dtype=np.float64)
        log_Z += logsumexp(log_weights) - np.log(K)
        ess_per_step.append(compute_ess(log_weights))
        indices = systematic_resample(log_weights)
        traj = traj[indices]
        if t < horizon:
            log_psi_prev = log_psi_next[indices]

    return {
        "log_Z": float(log_Z),
        "ess_per_step": ess_per_step,
        "paths": traj,
    }


def summarize_conditioned_paths(reference_paths, sample_paths, tica_model, checkpoints, plots_dir, prefix):
    sample_angles = features_to_angles(sample_paths.reshape(-1, 4))
    checkpoint_success_masks = []
    summary = {
        "num_paths": int(len(sample_paths)),
    }

    for spec in checkpoints:
        time_idx = int(spec["time"])
        cp_angles = features_to_angles(sample_paths[:, time_idx])
        cp_mask = basin_mask(cp_angles, spec["center"], spec["radius"])
        checkpoint_success_masks.append(cp_mask)
        summary[f"{spec['name']}_success_rate"] = float(np.mean(cp_mask))
        summary[f"{spec['name']}_mean_distance_rad"] = float(
            np.mean(np.sqrt(squared_angular_distance(cp_angles, spec["center"])))
        )

    if checkpoint_success_masks:
        joint_success = checkpoint_success_masks[0].copy()
        for mask in checkpoint_success_masks[1:]:
            joint_success &= mask
        summary["success_rate"] = float(np.mean(joint_success))
    else:
        summary["success_rate"] = 0.0

    if reference_paths is None or len(reference_paths) == 0:
        summary["reference_num_paths"] = 0
        summary["ramachandran"] = None
        summary["tica_fes"] = None
        return summary

    ref_angles = features_to_angles(reference_paths.reshape(-1, 4))
    rama_range = [[-np.pi, np.pi], [-np.pi, np.pi]]
    rama_stats = histogram_js_distance_2d(ref_angles, sample_angles, bins=64, value_range=rama_range)
    ref_rama = compute_free_energy_surface(ref_angles, bins=64, value_range=rama_range)
    sample_rama = compute_free_energy_surface(sample_angles, bins=64, value_range=rama_range)
    plot_histogram_comparison(
        ref_rama,
        sample_rama,
        os.path.join(plots_dir, f"{prefix}_ramachandran"),
        f"{prefix.replace('_', ' ').title()} Ramachandran",
        axis_labels=(r"$\phi$", r"$\psi$"),
    )
    ref_tica = tica_transform(reference_paths.reshape(-1, 4), tica_model)
    sample_tica = tica_transform(sample_paths.reshape(-1, 4), tica_model)
    tica_range = compute_eval_range(ref_tica[:, :2], sample_tica[:, :2])
    tica_stats = histogram_js_distance_2d(ref_tica[:, :2], sample_tica[:, :2], bins=50, value_range=tica_range)
    ref_tica_fes = compute_free_energy_surface(ref_tica[:, :2], bins=50, value_range=tica_range)
    sample_tica_fes = compute_free_energy_surface(sample_tica[:, :2], bins=50, value_range=tica_range)
    plot_free_energy_comparison(
        ref_tica_fes,
        sample_tica_fes,
        os.path.join(plots_dir, f"{prefix}_tica"),
        f"{prefix.replace('_', ' ').title()} TICA FES",
        axis_labels=("TIC-0", "TIC-1"),
    )

    summary["reference_num_paths"] = int(len(reference_paths))
    summary["ramachandran"] = {
        "js_distance": rama_stats["js_distance"],
        "js_divergence": rama_stats["js_divergence"],
    }
    summary["tica_fes"] = {
        "js_distance": tica_stats["js_distance"],
        "js_divergence": tica_stats["js_divergence"],
    }
    return summary


def main():
    args = parse_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if args.plots_dir:
        os.makedirs(args.plots_dir, exist_ok=True)

    start_center, midpoint_center, target_center = resolve_centers(args)
    print("=" * 72)
    print("UPDATED EXPERIMENT 5: Alanine Dipeptide")
    print("=" * 72)

    print("Loading trajectories...")
    angles, data_meta = load_alanine_trajectories(args.data_path)
    print(f"  Loaded {angles.shape[0]} trajectories x {angles.shape[1]} frames")

    split_gap = max(args.window_size, args.forward_horizon or args.window_size, args.tica_lag)
    train_angles, test_angles, split_info = split_train_test_temporal(
        angles,
        train_fraction=args.train_fraction,
        gap=split_gap,
    )
    if len(test_angles) == 0:
        raise ValueError("Need at least one held-out trajectory for evaluation.")

    train_features = angles_to_features(train_angles)
    test_features = angles_to_features(test_angles)
    dim = train_features.shape[-1]

    all_pairs_x = train_features[:, :-1, :].reshape(-1, dim)
    all_pairs_y = train_features[:, 1:, :].reshape(-1, dim)
    if args.max_pairs is not None and len(all_pairs_x) > args.max_pairs:
        pair_idx = rng.choice(len(all_pairs_x), size=args.max_pairs, replace=False)
        train_pairs_x = all_pairs_x[pair_idx]
        train_pairs_y = all_pairs_y[pair_idx]
    else:
        train_pairs_x = all_pairs_x
        train_pairs_y = all_pairs_y

    print("Building endpoint-conditioning windows...")
    train_windows_angles, _ = make_windows(train_angles, args.window_size, stride=args.window_stride)
    test_windows_angles, _ = make_windows(test_angles, args.window_size, stride=args.window_stride)
    if len(train_windows_angles) == 0 or len(test_windows_angles) == 0:
        raise ValueError("Not enough frames to build alanine windows at the requested horizon.")

    train_start_mask = basin_mask(train_windows_angles[:, 0], start_center, args.start_radius)
    test_start_mask = basin_mask(test_windows_angles[:, 0], start_center, args.start_radius)
    reward_checkpoints = []
    eval_checkpoints = []
    if args.task_mode == "midpoint_endpoint":
        mid_t = args.window_size // 2
        reward_checkpoints.append({
            "name": "midpoint",
            "time": mid_t,
            "center": midpoint_center,
            "radius": float(args.midpoint_radius),
            "lambda": float(args.midpoint_lambda),
        })
        eval_checkpoints.append({
            "name": "midpoint",
            "time": mid_t,
            "center": midpoint_center,
            "radius": float(args.midpoint_radius),
        })

    reward_checkpoints.append({
        "name": "terminal",
        "time": args.window_size,
        "center": target_center,
        "radius": float(args.target_radius),
        "lambda": float(args.reward_lambda),
    })
    eval_checkpoints.append({
        "name": "terminal",
        "time": args.window_size,
        "center": target_center,
        "radius": float(args.target_radius),
    })

    test_transition_mask = test_start_mask.copy()
    for spec in eval_checkpoints:
        test_transition_mask &= basin_mask(
            test_windows_angles[:, spec["time"]],
            spec["center"],
            spec["radius"],
        )

    train_start_windows = train_windows_angles[train_start_mask]
    test_start_windows = test_windows_angles[test_start_mask]
    test_transition_windows = test_windows_angles[test_transition_mask]

    if len(train_start_windows) == 0:
        raise ValueError("No training windows start in the specified C7eq basin.")
    if len(test_start_windows) == 0:
        raise ValueError("No held-out windows start in the specified C7eq basin.")

    if len(test_transition_windows) == 0:
        print("  Warning: no held-out windows satisfy all checkpoint constraints; conditioned metrics will use success rates only.")

    ref_windows_angles = subsample_rows(train_start_windows, args.max_ref_windows, rng)
    ref_windows_features = angles_to_features(ref_windows_angles)
    endpoint_log_G = checkpoint_log_G_fn(reward_checkpoints)
    future_targets = build_future_reward_targets(ref_windows_features, args.window_size, endpoint_log_G)

    start_feature_pool = angles_to_features(test_start_windows[:, 0])
    representative_start_feature = angles_to_features(circular_mean(train_start_windows[:, 0]).reshape(1, 2))[0]

    print("Training scores...")
    t0 = time.time()
    o_model, _ = train_conditional_score(
        train_pairs_x,
        train_pairs_y,
        dim=dim,
        n_epochs=args.score_epochs,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        device=args.device,
    )
    o_score = NNConditionalScore(o_model, dim=dim, device=args.device)
    print(f"  Score training time: {time.time() - t0:.1f}s")

    objective_labels = {"mc": "Take3 (MC)", "kl": "Take3 (KL)", "td": "Take3 (TD)"}
    if not args.take3_objectives:
        raise ValueError("At least one Take 3 objective must be requested.")

    print("Training Take 3 twists...")
    twist_fns = {}
    twist_training = {}
    for objective in args.take3_objectives:
        method_name = objective_labels[objective]
        print(f"  Training {method_name} twist...")
        t0 = time.time()
        common_kwargs = {
            "dim": dim,
            "T": args.window_size,
            "n_epochs": args.twist_epochs,
            "hidden_dim": args.hidden_dim,
            "n_layers": max(args.n_layers - 1, 2),
            "device": args.device,
        }
        if objective == "mc":
            twist_model, losses = train_positive_twist_mc(
                ref_windows_features,
                future_targets,
                loss_space=args.twist_loss_space,
                **common_kwargs,
            )
        elif objective == "kl":
            twist_model, losses = train_positive_twist_kl(
                ref_windows_features,
                future_targets,
                **common_kwargs,
            )
        elif objective == "td":
            twist_model, losses = train_positive_twist_td(
                ref_windows_features,
                log_G_fn=endpoint_log_G,
                loss_space=args.twist_loss_space,
                **common_kwargs,
            )
        else:
            raise ValueError(f"Unsupported Take 3 objective: {objective}")

        elapsed = time.time() - t0
        twist_fns[method_name] = PositiveNNTwist(twist_model, dim=dim, device=args.device)
        twist_training[method_name] = {
            "objective": objective.upper(),
            "time_seconds": float(elapsed),
            "final_loss": None if len(losses) == 0 else float(losses[-1]),
            "num_finite_steps": int(len(losses)),
        }
        print(f"  {method_name} twist training time: {elapsed:.1f}s")

    def reverse_step(x_cond, n_samples):
        return rollout_reverse_step(
            o_score=o_score,
            x_cond=x_cond,
            n_samples=n_samples,
            s_max=args.S_max,
            n_steps=args.n_diff_steps,
        )

    print("Fitting TICA on training trajectories...")
    tica_model = fit_tica(train_features, lag=args.tica_lag, n_components=2)

    print("Running forward-simulation evaluation...")
    forward_horizon = args.forward_horizon or args.window_size
    forward_windows_angles, _ = make_windows(test_angles, forward_horizon, stride=args.window_stride)
    if len(forward_windows_angles) == 0:
        raise ValueError("Not enough held-out frames to build forward-evaluation windows.")
    forward_ref = angles_to_features(subsample_rows(forward_windows_angles, args.max_forward_paths, rng))
    forward_model = sample_forward_paths(
        initial_features=forward_ref[:, 0],
        horizon=forward_horizon,
        reverse_step_fn=reverse_step,
    )
    forward_metrics = summarize_forward_model(
        reference_paths=forward_ref,
        model_paths=forward_model,
        tica_model=tica_model,
        plots_dir=args.plots_dir,
        prefix="forward_simulation",
    )

    print("Running endpoint-conditioning comparison...")
    empirical_log_Z = float(
        logsumexp(path_log_reward(angles_to_features(test_start_windows), endpoint_log_G))
        - np.log(len(test_start_windows))
    )
    learned_log_Z = estimate_log_Z_from_chain(
        start_features=start_feature_pool,
        horizon=args.window_size,
        reverse_step_fn=reverse_step,
        endpoint_log_G=endpoint_log_G,
        n_samples=args.gt_samples,
        batch_size=min(256, args.gt_samples),
        rng=np.random.default_rng(args.seed + 999),
    )

    endpoint_results = {}
    reference_transition_paths = (
        angles_to_features(subsample_rows(test_transition_windows, args.max_transition_ref, rng))
        if len(test_transition_windows) > 0
        else None
    )

    naive_paths = []
    for trial in range(args.n_trials):
        np.random.seed(args.seed + 31 * trial)
        trial_rng = np.random.default_rng(args.seed + 31 * trial)
        start_idx = trial_rng.choice(len(start_feature_pool), size=args.K, replace=True)
        naive_paths.append(
            sample_forward_paths(
                initial_features=start_feature_pool[start_idx],
                horizon=args.window_size,
                reverse_step_fn=reverse_step,
            )
        )
    naive_paths = np.concatenate(naive_paths, axis=0)
    endpoint_results["NaiveDiffusion"] = {
        "summary": summarize_conditioned_paths(
            reference_paths=reference_transition_paths,
            sample_paths=naive_paths,
            tica_model=tica_model,
            checkpoints=eval_checkpoints,
            plots_dir=args.plots_dir,
            prefix="endpoint_naive_diffusion",
        ),
        "log_Z": None,
        "ess": None,
    }

    bootstrap_trials = []
    bootstrap_paths = []
    for trial in range(args.n_trials):
        np.random.seed(args.seed + 101 * trial)
        trial_rng = np.random.default_rng(args.seed + 101 * trial)
        start_idx = trial_rng.choice(len(start_feature_pool), size=args.K, replace=True)
        out = run_bootstrap(
            start_features=start_feature_pool[start_idx],
            horizon=args.window_size,
            K=args.K,
            reverse_step_fn=reverse_step,
            endpoint_log_G=endpoint_log_G,
        )
        bootstrap_trials.append(out)
        bootstrap_paths.append(out["paths"])
    bootstrap_paths = np.concatenate(bootstrap_paths, axis=0)
    endpoint_results["Bootstrap"] = {
        "log_Z": mean_std([x["log_Z"] for x in bootstrap_trials]),
        "ess": mean_std([np.mean(x["ess_per_step"]) for x in bootstrap_trials]),
        "summary": summarize_conditioned_paths(
            reference_paths=reference_transition_paths,
            sample_paths=bootstrap_paths,
            tica_model=tica_model,
            checkpoints=eval_checkpoints,
            plots_dir=args.plots_dir,
            prefix="endpoint_bootstrap",
        ),
    }

    take3_plot_prefixes = {
        "Take3 (MC)": "endpoint_take3",
        "Take3 (KL)": "endpoint_take3_kl",
        "Take3 (TD)": "endpoint_take3_td",
    }
    for method_name, method_twist_fn in twist_fns.items():
        print(f"  Running {method_name}...")
        take3_trials = []
        take3_paths = []
        for trial in range(args.n_trials):
            np.random.seed(args.seed + 211 * trial)
            trial_rng = np.random.default_rng(args.seed + 211 * trial)
            start_idx = trial_rng.choice(len(start_feature_pool), size=args.K, replace=True)
            out = run_take3(
                start_features=start_feature_pool[start_idx],
                horizon=args.window_size,
                K=args.K,
                M=args.M,
                reverse_step_fn=reverse_step,
                endpoint_log_G=endpoint_log_G,
                twist_fn=method_twist_fn,
            )
            take3_trials.append(out)
            take3_paths.append(out["paths"])
        take3_paths = np.concatenate(take3_paths, axis=0)
        endpoint_results[method_name] = {
            "log_Z": mean_std([x["log_Z"] for x in take3_trials]),
            "ess": mean_std([np.mean(x["ess_per_step"]) for x in take3_trials]),
            "training": twist_training[method_name],
            "summary": summarize_conditioned_paths(
                reference_paths=reference_transition_paths,
                sample_paths=take3_paths,
                tica_model=tica_model,
                checkpoints=eval_checkpoints,
                plots_dir=args.plots_dir,
                prefix=take3_plot_prefixes.get(method_name, method_name.lower().replace(" ", "_")),
            ),
        }

    output = {
        "experiment": "run_exp5_alanine",
        "data_path": args.data_path,
        "output_json": args.output_json,
        "plots_dir": args.plots_dir,
        "device": args.device,
        "seed": args.seed,
        "data_metadata": data_meta,
        "split": split_info,
        "parameters": {
            "window_size": args.window_size,
            "forward_horizon": forward_horizon,
            "window_stride": args.window_stride,
            "tica_lag": args.tica_lag,
            "K": args.K,
            "M": args.M,
            "n_trials": args.n_trials,
            "S_max": args.S_max,
            "n_diff_steps": args.n_diff_steps,
            "gt_samples": args.gt_samples,
            "take3_objectives": args.take3_objectives,
            "twist_loss_space": args.twist_loss_space,
            "task_mode": args.task_mode,
            "reward_lambda": args.reward_lambda,
            "midpoint_lambda": args.midpoint_lambda,
            "start_radius_rad": args.start_radius,
            "midpoint_radius_rad": args.midpoint_radius,
            "target_radius_rad": args.target_radius,
            "start_center_rad": start_center,
            "midpoint_center_rad": midpoint_center,
            "target_center_rad": target_center,
            "reward_checkpoints": reward_checkpoints,
            "default_target_basin": "alpha_R" if args.target_center_deg is None else "custom",
            "representative_start_feature": representative_start_feature,
        },
        "dataset_counts": {
            "train_frames": int(train_features.shape[0] * train_features.shape[1]),
            "test_frames": int(test_features.shape[0] * test_features.shape[1]),
            "train_start_windows": int(len(train_start_windows)),
            "test_start_windows": int(len(test_start_windows)),
            "test_transition_windows": int(len(test_transition_windows)),
            "reference_checkpoint_windows": int(0 if reference_transition_paths is None else len(reference_transition_paths)),
        },
        "forward_simulation": forward_metrics,
        "take3_training": twist_training,
        "endpoint_conditioning": {
            "ground_truth": {
                "empirical_log_Z_from_test_windows": empirical_log_Z,
                "learned_proposal_log_Z": learned_log_Z,
            },
            "methods": endpoint_results,
        },
    }

    save_json(output, args.output_json)
    print(f"Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
