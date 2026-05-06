"""
Diagnostic plots for FKC paper.

1. ESS-over-time: shows per-step ESS for all methods
2. M-sweep: log Z bias and std vs M
3. Variance decomposition: R_t vs D_t/M stacked bars

All plots use matplotlib and can read from JSON results or run inline.
"""

import json
import os
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), "results", ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logsumexp

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _save_figure(fig, output_path):
    fig.savefig(output_path + ".pdf", dpi=150, bbox_inches="tight")
    fig.savefig(output_path + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}.pdf/.png")


def _estimate_variance_components(log_v, log_psi_cached, M):
    """
    Practical per-step diagnostics from the variance appendix remarks.

    Given row-wise inner log-weights
      log_v = log G_t + log psi_t
    and cached outer twist values log psi_{t-1}, estimate

      R_t    from the empirical variance of mean-one normalized outer weights
      D_t    from the inner-ESS proxy chi^2(q_t^* || p_ref) ≈ M / ESS_inner - 1.
    """
    log_v = np.asarray(log_v, dtype=np.float64)
    log_psi_cached = np.asarray(log_psi_cached, dtype=np.float64).reshape(-1)
    if log_v.ndim != 2:
        raise ValueError("log_v must be a 2D array with shape (K, M)")
    if log_v.shape[0] != len(log_psi_cached):
        raise ValueError("log_v rows and log_psi_cached must have matching length")

    log_norm = logsumexp(log_v, axis=1, keepdims=True)
    inner_w = np.exp(log_v - log_norm)
    invalid = (~np.isfinite(inner_w)).any(axis=1) | (np.sum(inner_w, axis=1) < 1e-300)
    if np.any(invalid):
        inner_w[invalid] = 1.0 / log_v.shape[1]

    ess_inner = 1.0 / np.sum(inner_w ** 2, axis=1)
    inner_chi = np.maximum(M / np.maximum(ess_inner, 1e-12) - 1.0, 0.0)

    log_outer = log_norm.reshape(-1) - np.log(M) - log_psi_cached
    log_h = log_outer - (logsumexp(log_outer) - np.log(len(log_outer)))
    h_hat = np.exp(log_h)

    return {
        "R_t": float(np.var(h_hat)),
        "D_t": float(np.mean((h_hat ** 2) * inner_chi)),
        "inner_ess_mean": float(np.mean(ess_inner)),
        "log_outer": log_outer,
    }


# ============================================================
# 1. ESS-over-time plot
# ============================================================

def run_ess_over_time_exp1(output_path=None, force=False):
    """
    Run Exp 1 methods and record per-step ESS.
    Saves plot + JSON data.
    """
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "ess_over_time_exp1")
    json_path = output_path + ".json"
    if os.path.exists(json_path) and not force:
        print(f"Using existing ESS JSON: {json_path}")
        plot_ess_from_json(json_path, output_path=output_path)
        with open(json_path) as f:
            return json.load(f)["ess_data"]

    import sys
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from updated_code.double_well_core import (
        langevin_step, log_reward, sample_equilibrium,
    )
    from updated_code.learned_scores import (
        train_conditional_score,
        NNConditionalScore,
    )
    from updated_code.common import (
        systematic_resample, compute_ess, sample_row_indices,
        build_future_reward_targets, save_json,
    )
    from updated_code.fixed_twist import (
        train_positive_twist_mc, PositiveNNTwist,
    )

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    np.random.seed(42)

    dim = 1; dt = 0.05; T = 10; x_0 = -1.0; x_target = 1.0; lam = 3.0
    K = 50; M = 5; S_MAX = 2.0; N_DIFF = 25; n_trials = 10
    alpha_tweedie = np.exp(-S_MAX / 2.0)
    sigma2_tweedie = 1.0 - np.exp(-S_MAX)

    def endpoint_log_G(x, t):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if t == T:
            return log_reward(x, x_target, lam)
        return np.zeros_like(x)

    def conditional_score_eval_1d(x_batch, s, x_cond_batch):
        x_arr = np.asarray(x_batch, dtype=np.float64).reshape(-1, 1)
        x_cond_arr = np.asarray(x_cond_batch, dtype=np.float64)
        if x_cond_arr.ndim == 0:
            x_cond_arr = np.full((len(x_arr), 1), float(x_cond_arr), dtype=np.float64)
        else:
            x_cond_arr = x_cond_arr.reshape(-1, 1)
            if len(x_cond_arr) == 1 and len(x_arr) > 1:
                x_cond_arr = np.tile(x_cond_arr, (len(x_arr), 1))
            elif len(x_cond_arr) != len(x_arr):
                raise ValueError("x_batch and x_cond_batch must have matching batch sizes")

        with torch.no_grad():
            inp = torch.tensor(
                np.concatenate([x_arr, x_cond_arr], axis=-1),
                dtype=torch.float32,
                device=device,
            )
            st = torch.full((len(x_arr),), float(s), dtype=torch.float32, device=device)
            out = o_model(inp, st).cpu().numpy().reshape(-1)
        return np.clip(out, -o_score.clip, o_score.clip)

    def reverse_sde_sample(x_cond, n_samples=1):
        x_cond_arr = np.asarray(x_cond, dtype=np.float64).reshape(-1)
        n_cond = len(x_cond_arr)
        x_cond_rep = np.repeat(x_cond_arr, n_samples)
        ds = S_MAX / N_DIFF
        x = np.random.randn(len(x_cond_rep))
        for step in range(N_DIFF):
            s = max(S_MAX - step * ds, 1e-6)
            score = conditional_score_eval_1d(x, s, x_cond_rep)
            x = x + (0.5 * x + score) * ds + np.random.randn(len(x_cond_rep)) * np.sqrt(ds)
        if n_cond == 1:
            return x.reshape(n_samples)
        return x.reshape(n_cond, n_samples)

    def exact_transition_sample(x_cond, n_samples=1):
        x_cond_arr = np.asarray(x_cond, dtype=np.float64).reshape(-1)
        n_cond = len(x_cond_arr)
        if n_cond == 1:
            return np.asarray(
                langevin_step(float(x_cond_arr[0]), dt, n_samples=n_samples),
                dtype=np.float64,
            ).reshape(n_samples)
        x_rep = np.repeat(x_cond_arr, n_samples)
        return np.asarray(
            langevin_step(x_rep, dt, n_samples=len(x_rep)),
            dtype=np.float64,
        ).reshape(n_cond, n_samples)

    def sample_reference_trajectories(n_trajectories):
        trajs = np.zeros((n_trajectories, T + 1), dtype=np.float64)
        trajs[:, 0] = x_0
        x = np.full(n_trajectories, x_0, dtype=np.float64)
        for t_idx in range(T):
            x = exact_transition_sample(x, n_samples=1).reshape(-1)
            trajs[:, t_idx + 1] = x
        return trajs

    def tweedie_twist_batch_fast_1d(x_proposals, t, n_rollouts=1, clip=10.0):
        x_proposals = np.asarray(x_proposals, dtype=np.float64).reshape(-1)
        if t >= T:
            return np.zeros(len(x_proposals), dtype=np.float64)
        log_rewards = np.zeros((n_rollouts, len(x_proposals)), dtype=np.float64)
        for r in range(n_rollouts):
            x_hat = x_proposals.copy()
            for _ in range(t, T):
                z = np.random.randn(len(x_hat))
                score = conditional_score_eval_1d(z, S_MAX, x_hat)
                x_hat = np.clip((z + sigma2_tweedie * score) / alpha_tweedie, -clip, clip)
            log_rewards[r] = -lam * (x_hat - x_target) ** 2
        return logsumexp(log_rewards, axis=0) - np.log(n_rollouts)

    print("Collecting training data...")
    eq_samples = sample_equilibrium(10000, dt=0.01, n_burnin=10000)
    n_pairs = 10000
    xt = np.zeros(n_pairs); xtp1 = np.zeros(n_pairs)
    eq_idx = np.random.randint(len(eq_samples), size=n_pairs // 2)
    xt[: n_pairs // 2] = eq_samples[eq_idx]
    xtp1[: n_pairs // 2] = np.asarray(
        langevin_step(xt[: n_pairs // 2], dt, n_samples=n_pairs // 2),
        dtype=np.float64,
    ).reshape(-1)
    for i in range(n_pairs // 2, n_pairs):
        x = x_0
        for _ in range(np.random.randint(0, T)):
            x = langevin_step(x, dt)
        xt[i] = x
        xtp1[i] = langevin_step(x, dt)

    n_ref = 5000
    ref_trajs = sample_reference_trajectories(n_ref)
    future_targets = build_future_reward_targets(ref_trajs, T, endpoint_log_G)

    print("Training scores...")
    o_model, _ = train_conditional_score(xt.reshape(-1, 1), xtp1.reshape(-1, 1), dim=1, n_epochs=3000, hidden_dim=64, n_layers=3, device=device, verbose=False)
    o_score = NNConditionalScore(o_model, dim=1, device=device)

    print("Training twist...")
    twist_model, _ = train_positive_twist_mc(ref_trajs, future_targets, dim=1, T=T, n_epochs=3000, hidden_dim=64, device=device, verbose=False)
    twist_mc = PositiveNNTwist(twist_model, dim=1, device=device)

    ess_data = {}

    print("Running Bootstrap...")
    bootstrap_ess = np.zeros((n_trials, T))
    for trial in range(n_trials):
        np.random.seed(trial * 31)
        traj = np.zeros((K, T + 1), dtype=np.float64)
        traj[:, 0] = x_0
        for t in range(1, T + 1):
            traj[:, t] = np.asarray(reverse_sde_sample(traj[:, t - 1], n_samples=1)).reshape(K)
            log_weights = endpoint_log_G(traj[:, t], t)
            bootstrap_ess[trial, t - 1] = compute_ess(log_weights)
            idx = systematic_resample(log_weights)
            traj = traj[idx]
    ess_data["Bootstrap"] = {
        "mean": bootstrap_ess.mean(axis=0).tolist(),
        "std": bootstrap_ess.std(axis=0).tolist(),
    }

    print("Running Take3 (MC)...")
    take3_ess = np.zeros((n_trials, T))
    for trial in range(n_trials):
        np.random.seed(trial * 31)
        traj = np.zeros((K, T + 1), dtype=np.float64)
        traj[:, 0] = x_0
        log_psi_cached = np.full(K, float(twist_mc(np.array([x_0]), 0)))
        for t in range(1, T + 1):
            proposals = np.asarray(reverse_sde_sample(traj[:, t - 1], n_samples=M)).reshape(K, M)
            log_G = endpoint_log_G(proposals.reshape(-1), t).reshape(K, M)
            if t < T:
                log_twist = np.asarray(twist_mc(proposals.reshape(-1, 1), t)).reshape(K, M)
            else:
                log_twist = np.zeros((K, M))
            log_v = log_G + log_twist
            j_sel, log_norm = sample_row_indices(log_v, n_draws=1)
            traj[:, t] = proposals[np.arange(K), j_sel]
            log_weights = log_norm - np.log(M) - log_psi_cached
            take3_ess[trial, t - 1] = compute_ess(log_weights)
            idx = systematic_resample(log_weights)
            traj = traj[idx]
            log_psi_cached = (log_twist[np.arange(K), j_sel] if t < T else np.zeros(K))[idx]
    ess_data["Take3 (MC)"] = {
        "mean": take3_ess.mean(axis=0).tolist(),
        "std": take3_ess.std(axis=0).tolist(),
    }

    print("Running Take1 (Tweedie)...")
    take1_ess = np.zeros((n_trials, T))
    for trial in range(n_trials):
        np.random.seed(trial * 31)
        traj = np.zeros((K, T + 1), dtype=np.float64)
        traj[:, 0] = x_0
        log_psi_init = float(tweedie_twist_batch_fast_1d(np.array([x_0]), 0, n_rollouts=3)[0])
        log_psi_cached = np.full(K, log_psi_init)
        for t in range(1, T + 1):
            proposals = np.asarray(reverse_sde_sample(traj[:, t - 1], n_samples=M)).reshape(K, M)
            log_G = endpoint_log_G(proposals.reshape(-1), t).reshape(K, M)
            if t < T:
                log_twist = tweedie_twist_batch_fast_1d(proposals.reshape(-1), t, n_rollouts=1).reshape(K, M)
            else:
                log_twist = np.zeros((K, M))
            log_v = log_G + log_twist
            j_sel, log_norm = sample_row_indices(log_v, n_draws=1)
            traj[:, t] = proposals[np.arange(K), j_sel]
            log_weights = log_norm - np.log(M) - log_psi_cached
            take1_ess[trial, t - 1] = compute_ess(log_weights)
            idx = systematic_resample(log_weights)
            traj = traj[idx]
            log_psi_cached = (log_twist[np.arange(K), j_sel] if t < T else np.zeros(K))[idx]
    ess_data["Take1 (Tweedie)"] = {
        "mean": take1_ess.mean(axis=0).tolist(),
        "std": take1_ess.std(axis=0).tolist(),
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    times = np.arange(1, T + 1)
    colors = {"Bootstrap": "#2196F3", "Take1 (Tweedie)": "#FF9800", "Take3 (MC)": "#4CAF50"}
    for name in ["Bootstrap", "Take1 (Tweedie)", "Take3 (MC)"]:
        mean = np.array(ess_data[name]["mean"])
        std = np.array(ess_data[name]["std"])
        ax.plot(times, mean, label=name, color=colors[name], linewidth=2)
        ax.fill_between(times, mean - std, mean + std, alpha=0.2, color=colors[name])

    ax.set_xlabel("Time step $t$", fontsize=12)
    ax.set_ylabel("ESS", fontsize=12)
    ax.set_title("Per-step ESS — 1D Double-Well (K=50)", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(1, T)
    ax.set_ylim(0, K + 5)
    ax.axhline(y=K, color="gray", linestyle="--", alpha=0.3, label=f"K={K}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_figure(fig, output_path)

    save_json({"parameters": {"K": K, "M": M, "T": T, "n_trials": n_trials},
               "ess_data": ess_data}, json_path)
    print(f"Saved: {json_path}")
    return ess_data


# ============================================================
# 2. Plot from existing JSON results (no re-run needed)
# ============================================================

def plot_ess_from_json(json_path, output_path=None):
    """Plot ESS-over-time from a previously saved JSON."""
    with open(json_path) as f:
        data = json.load(f)

    ess_data = data["ess_data"]
    T = data["parameters"]["T"]
    K = data["parameters"]["K"]
    times = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"Bootstrap": "#2196F3", "Take1 (Tweedie)": "#FF9800",
              "Take3 (MC)": "#4CAF50", "Take3 (TD)": "#9C27B0"}
    for name, vals in ess_data.items():
        mean = np.array(vals["mean"])
        std = np.array(vals["std"])
        c = colors.get(name, "#607D8B")
        ax.plot(times, mean, label=name, color=c, linewidth=2)
        ax.fill_between(times, mean - std, mean + std, alpha=0.2, color=c)

    ax.set_xlabel("Time step $t$", fontsize=12)
    ax.set_ylabel("ESS", fontsize=12)
    ax.set_title(f"Per-step ESS (K={K})", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(1, T)
    ax.set_ylim(0, K + 5)
    ax.axhline(y=K, color="gray", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path is None:
        output_path = json_path.replace(".json", "")
    _save_figure(fig, output_path)


# ============================================================
# 3. M-sweep plot from Exp 4 results
# ============================================================

def plot_m_sweep(exp4_json=None, output_path=None):
    """
    Plot log Z bias and std vs M from Exp 4 Part A data.
    """
    if exp4_json is None:
        exp4_json = os.path.join(RESULTS_DIR, "run_exp4.json")
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "m_sweep_part_a")

    with open(exp4_json) as f:
        data = json.load(f)

    part_a = data.get("part_a_highd", data["part_a_1d"])
    gt = part_a["ground_truth_log_Z"]
    params = part_a.get("parameters", {})
    system_name = params.get("system_name", "Part A system")
    K_val = params.get("K", "?")

    M_vals = []
    ss_bias, ss_std = [], []
    ts_bias, ts_std = [], []

    for key in sorted(part_a["results_by_M"].keys(), key=lambda k: int(k.split("_")[1])):
        M_val = int(key.split("_")[1])
        M_vals.append(M_val)
        ss = part_a["results_by_M"][key]["single_stage"]["summary"]
        ts = part_a["results_by_M"][key]["two_stage"]["summary"]
        ss_bias.append(ss["log_Z"]["mean"] - gt)
        ss_std.append(ss["log_Z"]["std"])
        ts_bias.append(ts["log_Z"]["mean"] - gt)
        ts_std.append(ts["log_Z"]["std"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(M_vals, np.abs(ss_bias), "o-", label="Single-stage", color="#E91E63", linewidth=2, markersize=8)
    ax1.plot(M_vals, np.abs(ts_bias), "s-", label="Two-stage", color="#3F51B5", linewidth=2, markersize=8)
    ax1.set_xlabel("M (inner proposals)", fontsize=12)
    ax1.set_ylabel("|Bias| on log Z", fontsize=12)
    ax1.set_title(f"Bias vs M — {system_name} (K={K_val})", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.set_xscale("log")
    ax1.set_xticks(M_vals)
    ax1.set_xticklabels(M_vals)
    ax1.grid(True, alpha=0.3)

    ax2.plot(M_vals, ss_std, "o-", label="Single-stage", color="#E91E63", linewidth=2, markersize=8)
    ax2.plot(M_vals, ts_std, "s-", label="Two-stage", color="#3F51B5", linewidth=2, markersize=8)
    ax2.set_xlabel("M (inner proposals)", fontsize=12)
    ax2.set_ylabel("Std of log Ẑ", fontsize=12)
    ax2.set_title(f"Variance vs M — {system_name} (K={K_val})", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.set_xscale("log")
    ax2.set_xticks(M_vals)
    ax2.set_xticklabels(M_vals)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, output_path)


# ============================================================
# 4. Summary comparison bar chart
# ============================================================

def plot_method_comparison(exp1_json=None, output_path=None):
    """Bar chart comparing all methods on Exp 1."""
    if exp1_json is None:
        exp1_json = os.path.join(RESULTS_DIR, "run_exp1.json")
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "method_comparison_exp1")

    with open(exp1_json) as f:
        data = json.load(f)

    gt_diff = data["ground_truth"]["diffusion_log_Z"]
    methods = ["Bootstrap", "Take1 (Tweedie)", "Take3 (MC)", "Take3 (TD)", "Take2 (two-stage)"]
    means = [data["summary"][m]["log_Z"]["mean"] for m in methods]
    stds = [data["summary"][m]["log_Z"]["std"] for m in methods]
    labels = ["Bootstrap", "Take 1\n(Tweedie)", "Take 3\n(MC)", "Take 3\n(TD)", "Take 2\n(two-stage)"]

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=5, edgecolor="white", linewidth=1.5)
    ax.axhline(y=gt_diff, color="black", linestyle="--", linewidth=1.5, label=f"GT (diffusion) = {gt_diff:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("log Ẑ", fontsize=12)
    ax.set_title("1D Double-Well: Method Comparison (K=50, M=5)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    _save_figure(fig, output_path)


# ============================================================
# 5. Variance decomposition plot (Theorem 1 / Proposition)
# ============================================================

def run_variance_decomposition_exp1(output_path=None, force=False):
    """
    Practical variance diagnostics for Take 3 on Exp 1.

    This estimator follows the variance appendix remarks more faithfully than the
    previous heuristic:

    - R_t is estimated from the empirical variance of the mean-one normalized
      outer weights h_t^k within each step.
    - D_t is estimated via the inner ESS approximation
        chi^2(q_t^* || p_ref) ≈ M / ESS_inner - 1
      and then averaged as E[h_t^2 * chi^2].

    To match the appendix setup, the inner proposals are sampled from the exact
    reference transition p_ref rather than the learned reverse-SDE proposal.
    """
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "variance_decomposition_exp1")
    json_path = output_path + ".json"
    if os.path.exists(json_path) and not force:
        print(f"Using existing variance JSON: {json_path}")
        plot_variance_decomposition_from_json(json_path, output_path=output_path)
        with open(json_path) as f:
            return json.load(f)["decomposition"]

    import sys
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from updated_code.double_well_core import (
        langevin_step, log_reward,
    )
    from updated_code.common import (
        systematic_resample, sample_row_indices,
        build_future_reward_targets, save_json,
    )
    from updated_code.fixed_twist import (
        train_positive_twist_mc, PositiveNNTwist,
    )

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    np.random.seed(42)

    dim = 1; dt = 0.05; T = 10; x_0 = -1.0; x_target = 1.0; lam = 3.0
    K = 50; M = 5
    n_trials = 50

    def endpoint_log_G(x, t):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if t == T:
            return log_reward(x, x_target, lam)
        return np.zeros_like(x)

    def exact_transition_sample(x_cond, n_samples=1):
        x_cond_arr = np.asarray(x_cond, dtype=np.float64).reshape(-1)
        n_cond = len(x_cond_arr)
        if n_cond == 1:
            return np.asarray(
                langevin_step(float(x_cond_arr[0]), dt, n_samples=n_samples),
                dtype=np.float64,
            ).reshape(n_samples)
        x_rep = np.repeat(x_cond_arr, n_samples)
        return np.asarray(
            langevin_step(x_rep, dt, n_samples=len(x_rep)),
            dtype=np.float64,
        ).reshape(n_cond, n_samples)

    def sample_reference_trajectories(n_trajectories):
        trajs = np.zeros((n_trajectories, T + 1), dtype=np.float64)
        trajs[:, 0] = x_0
        x = np.full(n_trajectories, x_0, dtype=np.float64)
        for t_idx in range(T):
            x = exact_transition_sample(x, n_samples=1).reshape(-1)
            trajs[:, t_idx + 1] = x
        return trajs

    print("Collecting reference trajectories...")
    n_ref = 5000
    ref_trajs = sample_reference_trajectories(n_ref)
    future_targets = build_future_reward_targets(ref_trajs, T, endpoint_log_G)

    print("Training twist...")
    twist_model, _ = train_positive_twist_mc(ref_trajs, future_targets, dim=1, T=T, n_epochs=3000, hidden_dim=64, device=device, verbose=False)
    twist_mc = PositiveNNTwist(twist_model, dim=1, device=device)

    print(f"Running Take 3 MC ({n_trials} trials for variance diagnostics)...")
    r_trials = np.zeros((n_trials, T))
    d_trials = np.zeros((n_trials, T))
    inner_ess_trials = np.zeros((n_trials, T))
    for trial in range(n_trials):
        np.random.seed(trial * 31)
        traj = np.zeros((K, T + 1), dtype=np.float64)
        traj[:, 0] = x_0
        log_psi_cached = np.full(K, float(twist_mc(np.array([x_0]), 0)))

        for t in range(1, T + 1):
            proposals = np.asarray(exact_transition_sample(traj[:, t - 1], n_samples=M)).reshape(K, M)
            log_G = endpoint_log_G(proposals.reshape(-1), t).reshape(K, M)
            if t < T:
                log_twist = np.asarray(twist_mc(proposals.reshape(-1, 1), t)).reshape(K, M)
            else:
                log_twist = np.zeros((K, M))
            log_v = log_G + log_twist

            var_metrics = _estimate_variance_components(log_v, log_psi_cached, M)
            j_sel, _ = sample_row_indices(log_v, n_draws=1)
            selected = proposals[np.arange(K), j_sel]
            r_trials[trial, t - 1] = var_metrics["R_t"]
            d_trials[trial, t - 1] = var_metrics["D_t"]
            inner_ess_trials[trial, t - 1] = var_metrics["inner_ess_mean"]

            idx = systematic_resample(var_metrics["log_outer"])
            traj[:, t] = selected
            traj = traj[idx]
            log_psi_cached = (log_twist[np.arange(K), j_sel] if t < T else np.zeros(K))[idx]

        if (trial + 1) % 10 == 0:
            print(f"  trial {trial + 1}/{n_trials}")

    R_t = np.mean(r_trials, axis=0)
    D_t = np.mean(d_trials, axis=0)
    D_t_over_M = D_t / M

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    times = np.arange(1, T + 1)
    bar_width = 0.35

    ax.bar(times, R_t, width=bar_width, color="#4CAF50", alpha=0.85, label="$R_t$ estimate")
    ax.bar(times, D_t_over_M, width=bar_width, bottom=R_t,
           color="#FF9800", alpha=0.85, label="$D_t/M$ estimate")
    ax.set_xlabel("Time step $t$", fontsize=12)
    ax.set_ylabel("Variance contribution", fontsize=12)
    ax.set_title("Take 3 (MC, exact $p^{\\mathrm{ref}}$) — Variance diagnostics", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xticks(times)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    _save_figure(fig, output_path)

    save_json({
        "parameters": {"K": K, "M": M, "T": T, "n_trials": n_trials},
        "notes": {
            "proposal": "exact_p_ref",
            "estimator": "practical_diagnostics_from_variance_appendix",
        },
        "decomposition": {
            "Take3 (MC)": {
                "R_t": R_t.tolist(),
                "D_t": D_t.tolist(),
                "D_t_over_M": D_t_over_M.tolist(),
                "inner_ess_mean": np.mean(inner_ess_trials, axis=0).tolist(),
                "R_t_trials": r_trials.tolist(),
                "D_t_trials": d_trials.tolist(),
            }
        },
    }, json_path)
    print(f"Saved: {json_path}")
    return {
        "Take3 (MC)": {
            "R_t": R_t.tolist(),
            "D_t": D_t.tolist(),
            "D_t_over_M": D_t_over_M.tolist(),
        }
    }


def run_variance_decomposition_exp8(output_path=None, force=False, quick=False):
    """
    Variance diagnostics for the path-dependent Exp 8 benchmark.

    This compares
      - a trivial-twist baseline ("Bootstrap / trivial twist"), and
      - Take 3 (KL)
    under the *same* exact-p_ref nested diagnostic wrapper. This keeps the
    appendix decomposition aligned with the theory by holding the inner proposal
    budget M fixed across methods and changing only the twist.
    """
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "variance_decomposition_exp8")
    json_path = output_path + ".json"
    if os.path.exists(json_path) and not force:
        print(f"Using existing variance JSON: {json_path}")
        plot_variance_decomposition_from_json(json_path, output_path=output_path)
        with open(json_path) as f:
            return json.load(f)["decomposition"]

    import sys
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    EXPERIMENTS_DIR = os.path.join(ROOT, "experiments")
    if EXPERIMENTS_DIR not in sys.path:
        sys.path.insert(0, EXPERIMENTS_DIR)

    from updated_code.common import (
        systematic_resample, sample_row_indices,
        build_future_reward_targets, save_json,
    )
    from updated_code.fixed_twist import (
        train_positive_twist_kl, PositiveNNTwist,
    )
    from updated_code.run_exp8_nonmarkov_route import (
        DEVICE,
        DT_DYNAMICS,
        FirstHitUpperReward,
        WELL_L,
        WELL_R,
        langevin_step,
        sample_x_trajectories,
        sanitize_log_values,
    )

    import torch

    np.random.seed(42)
    torch.manual_seed(42)

    T = 36
    dt = DT_DYNAMICS
    K = 24 if quick else 64
    M = 6 if quick else 12
    n_trials = 4 if quick else 40
    n_ref = 512 if quick else 12000
    twist_epochs = 150 if quick else 3500
    twist_hidden_dim = 192 if quick else 256
    twist_layers = 3 if quick else 4

    reward_model = FirstHitUpperReward(
        T=T,
        target=WELL_R,
        x_gate_halfwidth=0.35,
        y_gate_threshold=0.45,
        beta_upper=0.04,
        beta_lower=0.04,
        lam_endpoint=10.0,
        endpoint_radius=0.5,
    )
    x0 = WELL_L.copy()

    def exact_transition_sample(x_cond, n_samples=1):
        x_cond = np.asarray(x_cond, dtype=np.float64)
        if x_cond.ndim == 1:
            x_batch = np.tile(x_cond, (n_samples, 1))
            return np.asarray(langevin_step(x_batch, dt), dtype=np.float64).reshape(n_samples, 2)
        x_batch = np.repeat(x_cond.reshape(-1, 2), n_samples, axis=0)
        return np.asarray(langevin_step(x_batch, dt), dtype=np.float64).reshape(len(x_cond), n_samples, 2)

    print("Collecting exact-backend reference trajectories for Exp 8 variance diagnostics...")
    ref_x_trajs = sample_x_trajectories(exact_transition_sample, x0, T, n_ref, batch_size=256)
    ref_aug_trajs, _ = reward_model.augment_trajectories(ref_x_trajs)
    future_targets = build_future_reward_targets(ref_aug_trajs, T, reward_model.log_G)

    print("Training Take 3 KL twist for Exp 8 variance diagnostics...")
    twist_model, kl_losses = train_positive_twist_kl(
        ref_aug_trajs,
        future_targets,
        dim=4,
        T=T,
        n_epochs=twist_epochs,
        lr=1e-3,
        hidden_dim=twist_hidden_dim,
        n_layers=twist_layers,
        device=DEVICE,
        verbose=not quick,
    )
    twist_kl = PositiveNNTwist(twist_model, dim=4, device=DEVICE)

    def run_method(method_name, twist=None):
        print(f"Running {method_name} ({n_trials} trials)...")
        r_trials = np.zeros((n_trials, T))
        d_trials = np.zeros((n_trials, T))
        inner_ess_trials = np.zeros((n_trials, T))

        init_flags = np.zeros(1, dtype=np.int64)
        init_aug = reward_model.augment_states(x0.reshape(1, 2), init_flags)
        if twist is None:
            init_log_psi = 0.0
        else:
            init_log_psi = float(sanitize_log_values(twist(init_aug, 0)).reshape(-1)[0])

        for trial in range(n_trials):
            np.random.seed(1000 + 37 * trial)
            particles = np.tile(x0, (K, 1))
            flags = reward_model.initial_flags(K)
            log_psi_cached = np.full(K, init_log_psi, dtype=np.float64)

            for t in range(1, T + 1):
                proposals_x = np.asarray(exact_transition_sample(particles, n_samples=M), dtype=np.float64).reshape(K, M, 2)
                proposal_flags = np.repeat(flags[:, None], M, axis=1).reshape(-1)
                proposal_flags = reward_model.advance_flags(
                    proposals_x.reshape(-1, 2),
                    proposal_flags,
                ).reshape(K, M)
                proposals_aug = reward_model.augment_states(
                    proposals_x.reshape(-1, 2),
                    proposal_flags.reshape(-1),
                ).reshape(K, M, 4)
                log_G = reward_model.log_G(proposals_aug.reshape(-1, 4), t).reshape(K, M)
                if t < T and twist is not None:
                    log_twist = sanitize_log_values(twist(proposals_aug.reshape(-1, 4), t)).reshape(K, M)
                else:
                    log_twist = np.zeros((K, M), dtype=np.float64)
                log_v = sanitize_log_values(log_G + log_twist)

                var_metrics = _estimate_variance_components(log_v, log_psi_cached, M)
                r_trials[trial, t - 1] = var_metrics["R_t"]
                d_trials[trial, t - 1] = var_metrics["D_t"]
                inner_ess_trials[trial, t - 1] = var_metrics["inner_ess_mean"]

                j_sel, _ = sample_row_indices(log_v, n_draws=1)
                chosen_x = proposals_x[np.arange(K), j_sel]
                chosen_flags = proposal_flags[np.arange(K), j_sel]
                if t < T and twist is not None:
                    log_psi_next = log_twist[np.arange(K), j_sel]
                else:
                    log_psi_next = np.zeros(K, dtype=np.float64)

                ancestors = systematic_resample(var_metrics["log_outer"])
                particles = chosen_x[ancestors]
                flags = chosen_flags[ancestors]
                log_psi_cached = log_psi_next[ancestors]

            if (trial + 1) % 10 == 0 or trial + 1 == n_trials:
                print(f"  {method_name}: trial {trial + 1}/{n_trials}")

        return {
            "R_t": np.mean(r_trials, axis=0).tolist(),
            "D_t": np.mean(d_trials, axis=0).tolist(),
            "D_t_over_M": (np.mean(d_trials, axis=0) / M).tolist(),
            "inner_ess_mean": np.mean(inner_ess_trials, axis=0).tolist(),
            "R_t_trials": r_trials.tolist(),
            "D_t_trials": d_trials.tolist(),
        }

    decomposition = {
        "Bootstrap / trivial twist": run_method("Bootstrap / trivial twist", twist=None),
        "Take3 (KL)": run_method("Take3 (KL)", twist=twist_kl),
    }

    save_json({
        "parameters": {
            "benchmark": "exp8_nonmarkov_route",
            "proposal_backend": "exact",
            "K": K,
            "M": M,
            "T": T,
            "dt": dt,
            "n_trials": n_trials,
            "n_ref": n_ref,
            "twist_epochs": twist_epochs,
            "quick": bool(quick),
        },
        "notes": {
            "title": "Experiment 8 exact-backend variance diagnostics",
            "proposal": "exact_p_ref",
            "estimator": "practical_diagnostics_from_variance_appendix",
            "comparison": (
                "Both methods are evaluated under the same nested-SMC diagnostic wrapper. "
                "The baseline uses the trivial twist, so this isolates how the KL twist "
                "moves variance from the resampling term R_t toward the inner term D_t/M."
            ),
            "kl_final_loss": float(kl_losses[-1]) if kl_losses else None,
        },
        "decomposition": decomposition,
    }, json_path)
    print(f"Saved: {json_path}")
    plot_variance_decomposition_from_json(json_path, output_path=output_path)
    return decomposition


def plot_variance_decomposition_from_json(json_path, output_path=None):
    """Plot variance decomposition from a previously saved JSON."""
    with open(json_path) as f:
        data = json.load(f)

    decomp = data["decomposition"]
    T = data["parameters"]["T"]
    times = np.arange(1, T + 1)
    method_names = list(decomp.keys())
    nonterminal_times = times[:-1]

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(12.8, 6.6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3.0, 1.35], hspace=0.35, wspace=0.18)
    top_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    bottom_ax = fig.add_subplot(gs[1, :])

    legend_handles = None
    legend_labels = None
    terminal_totals = []
    terminal_R = []
    terminal_D = []

    for idx, method_name in enumerate(method_names):
        ax = top_axes[idx]
        R_t = np.array(decomp[method_name]["R_t"], dtype=np.float64)
        D_t_M = np.array(decomp[method_name]["D_t_over_M"], dtype=np.float64)
        total = R_t + D_t_M

        R_nonterminal = R_t[:-1]
        D_nonterminal = D_t_M[:-1]
        total_nonterminal = total[:-1]

        ax.bar(nonterminal_times, R_nonterminal, width=0.72, color="#66BB6A", alpha=0.92, label="$R_t$ estimate")
        ax.bar(
            nonterminal_times, D_nonterminal, width=0.72, bottom=R_nonterminal,
            color="#FFA726", alpha=0.92, label="$D_t/M$ estimate",
        )
        ax.plot(nonterminal_times, total_nonterminal, color="black", linewidth=1.5, linestyle="--", label="Total")
        ax.scatter(nonterminal_times, total_nonterminal, color="black", s=9, zorder=3)
        ax.set_title(
            method_name + "\n" +
            f"non-terminal sum R = {np.sum(R_nonterminal):.2f}, sum D/M = {np.sum(D_nonterminal):.2f}",
            fontsize=12,
            pad=8,
        )
        ax.set_xlim(0.4, T - 0.4)
        ax.set_xticks(nonterminal_times[:: max(1, len(nonterminal_times) // 8)])
        ax.set_xlabel("Time step $t < T$", fontsize=12)
        ax.grid(True, alpha=0.25, axis="y")
        ax.grid(False, axis="x")
        ax.set_ylim(0.0, max(1e-12, 1.10 * float(np.max(total_nonterminal))))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        if idx == 0:
            ax.set_ylabel("Non-terminal variance", fontsize=12)
        legend_handles, legend_labels = ax.get_legend_handles_labels()

        terminal_R.append(float(R_t[-1]))
        terminal_D.append(float(D_t_M[-1]))
        terminal_totals.append(float(total[-1]))

    x = np.arange(len(method_names), dtype=np.float64)
    bottom_ax.bar(x, terminal_R, width=0.60, color="#66BB6A", alpha=0.92, label="$R_T$ estimate")
    bottom_ax.bar(x, terminal_D, width=0.60, bottom=terminal_R, color="#FFA726", alpha=0.92, label="$D_T/M$ estimate")
    bottom_ax.plot(x, terminal_totals, color="black", linewidth=1.5, linestyle="--", marker="o", markersize=5, label="Terminal total")
    bottom_ax.set_xticks(x)
    bottom_ax.set_xticklabels(method_names, fontsize=11)
    bottom_ax.set_ylabel("Terminal-step variance", fontsize=12)
    bottom_ax.set_title("Terminal step $t=T$", fontsize=12, pad=8)
    bottom_ax.grid(True, alpha=0.25, axis="y")
    bottom_ax.grid(False, axis="x")
    bottom_ax.set_ylim(0.0, max(1e-12, 1.14 * max(terminal_totals)))

    for idx, total_val in enumerate(terminal_totals):
        bottom_ax.text(
            x[idx], total_val + 0.02 * max(terminal_totals),
            f"{total_val:.2f}",
            ha="center", va="bottom", fontsize=10,
        )

    title = data.get("notes", {}).get("title", "Variance diagnostics")
    fig.suptitle(title, fontsize=15, y=0.985)
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center", ncol=3, frameon=False,
            bbox_to_anchor=(0.5, 0.935),
        )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.11, top=0.86, hspace=0.36, wspace=0.18)
    if output_path is None:
        output_path = json_path.replace(".json", "")
    _save_figure(fig, output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ess", action="store_true", help="Run ESS-over-time diagnostic (requires torch)")
    parser.add_argument("--var-decomp", action="store_true", help="Run Exp 1 variance decomposition (requires torch)")
    parser.add_argument("--var-decomp-exp8", action="store_true", help="Run Exp 8 exact-backend variance decomposition (requires torch)")
    parser.add_argument("--m-sweep", action="store_true", help="Plot M-sweep from Exp 4 JSON")
    parser.add_argument("--comparison", action="store_true", help="Plot method comparison bar chart")
    parser.add_argument("--all-plots", action="store_true", help="Generate all plots from existing JSON (no torch needed)")
    parser.add_argument("--all-run", action="store_true", help="Run all diagnostics that need torch + all plots")
    parser.add_argument("--quick", action="store_true", help="Use reduced settings for heavy diagnostics")
    parser.add_argument("--force", action="store_true", help="Recompute heavy diagnostics even if cached JSON exists")
    args = parser.parse_args()

    if args.ess or args.all_run:
        run_ess_over_time_exp1(force=args.force)
    if args.var_decomp or args.all_run:
        run_variance_decomposition_exp1(force=args.force)
    if args.var_decomp_exp8 or args.all_run:
        run_variance_decomposition_exp8(force=args.force, quick=args.quick)
    if args.m_sweep or args.all_plots or args.all_run:
        plot_m_sweep()
    if args.comparison or args.all_plots or args.all_run:
        plot_method_comparison()
    if not any(vars(args).values()):
        print("Usage:")
        print("  python diagnostics.py --ess            # Run ESS diagnostic (needs torch)")
        print("  python diagnostics.py --var-decomp      # Run Exp 1 variance decomposition (needs torch)")
        print("  python diagnostics.py --var-decomp-exp8 # Run Exp 8 variance decomposition (needs torch)")
        print("  python diagnostics.py --m-sweep         # Plot M-sweep from Exp 4 JSON")
        print("  python diagnostics.py --comparison      # Plot method comparison")
        print("  python diagnostics.py --all-plots       # All plots from existing JSON (no torch)")
        print("  python diagnostics.py --all-run         # Run all diagnostics + all plots")
        print("  python diagnostics.py --ess --force     # Recompute ESS even if cached JSON exists")
        print("  python diagnostics.py --var-decomp-exp8 --quick --force")
