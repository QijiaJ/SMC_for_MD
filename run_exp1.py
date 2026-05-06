"""
Updated Experiment 1: 1D double-well with corrected twist handling.

"""

import os
import sys
import time
import numpy as np
from scipy.special import logsumexp
import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from updated_code.double_well_core import (
    langevin_step,
    log_reward,
    sample_trajectory,
    sample_equilibrium,
)
from updated_code.learned_scores import (
    train_marginal_score,
    train_conditional_score,
    NNMarginalScore,
    NNConditionalScore,
)

from updated_code.common import (
    systematic_resample,
    compute_ess,
    build_future_reward_targets,
    mean_std,
    save_json,
)
from updated_code.fixed_twist import (
    train_positive_twist_mc,
    train_positive_twist_td,
    PositiveNNTwist,
)

np.random.seed(42)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "results", "run_exp1.json")

dim = 1
dt = 0.05
T = 10
x_0 = -1.0
x_target = 1.0
lam = 3.0
K = 50
M = 5
S_MAX = 2.0
N_DIFF_STEPS = 25
n_trials = 15
n_trials_t2 = 10
MARGINAL_SCORE_CONFIG = {
    "n_eq_samples": 15000,
    "hidden_dim": 64,
    "n_layers": 3,
    "n_epochs": 4000,
}
CONDITIONAL_SCORE_CONFIG = {
    "n_transition_pairs": 6000,
    "hidden_dim": 64,
    "n_layers": 3,
    "n_epochs": 2000,
}

print("=" * 70)
print("UPDATED EXPERIMENT 1: 1D Double-Well")
print("=" * 70)
print(f"dt={dt}, T={T}, x0={x_0}, target={x_target}, K={K}, M={M}")
print(f"Reverse SDE: S={S_MAX}, n_steps={N_DIFF_STEPS}\n")
print(
    "Marginal score config: "
    f"samples={MARGINAL_SCORE_CONFIG['n_eq_samples']}, "
    f"hidden={MARGINAL_SCORE_CONFIG['hidden_dim']}, "
    f"layers={MARGINAL_SCORE_CONFIG['n_layers']}, "
    f"epochs={MARGINAL_SCORE_CONFIG['n_epochs']}"
)
print(
    "Conditional score config: "
    f"pairs={CONDITIONAL_SCORE_CONFIG['n_transition_pairs']}, "
    f"hidden={CONDITIONAL_SCORE_CONFIG['hidden_dim']}, "
    f"layers={CONDITIONAL_SCORE_CONFIG['n_layers']}, "
    f"epochs={CONDITIONAL_SCORE_CONFIG['n_epochs']}\n"
)


def endpoint_log_G(x, t):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if t == T:
        return log_reward(x, x_target, lam)
    return np.zeros_like(x)


def reward_fn_1d(x_T):
    x_T = float(np.squeeze(x_T))
    return -lam * (x_T - x_target) ** 2


print("--- Phase 1: Collecting training data ---\n")
print(f"Collecting {MARGINAL_SCORE_CONFIG['n_eq_samples']} equilibrium samples...")
eq_samples = sample_equilibrium(
    MARGINAL_SCORE_CONFIG["n_eq_samples"],
    dt=0.01,
    n_burnin=10000,
)
print(f"  Range: [{eq_samples.min():.2f}, {eq_samples.max():.2f}]")

print(f"Collecting {CONDITIONAL_SCORE_CONFIG['n_transition_pairs']} transition pairs (x_t, x_t+1)...")
n_pairs = CONDITIONAL_SCORE_CONFIG["n_transition_pairs"]
x_t_data = np.zeros(n_pairs)
x_tp1_data = np.zeros(n_pairs)
for i in range(n_pairs // 2):
    x_t_data[i] = eq_samples[np.random.randint(len(eq_samples))]
    x_tp1_data[i] = langevin_step(x_t_data[i], dt)
for i in range(n_pairs // 2, n_pairs):
    x = x_0
    for _ in range(np.random.randint(0, T)):
        x = langevin_step(x, dt)
    x_t_data[i] = x
    x_tp1_data[i] = langevin_step(x, dt)
print(f"  x_t range: [{x_t_data.min():.2f}, {x_t_data.max():.2f}]")


print("--- Phase 2: Training NN scores ---\n")
print("Training marginal score j_s (NN, DSM)...")
t0 = time.time()
j_model, _ = train_marginal_score(
    eq_samples.reshape(-1, 1),
    dim=1,
    n_epochs=MARGINAL_SCORE_CONFIG["n_epochs"],
    hidden_dim=MARGINAL_SCORE_CONFIG["hidden_dim"],
    n_layers=MARGINAL_SCORE_CONFIG["n_layers"],
    device=DEVICE,
)
j_score = NNMarginalScore(j_model, dim=1, device=DEVICE)
print(f"  Time: {time.time() - t0:.1f}s\n")

print("Training conditional score o_s (NN, DSM)...")
t0 = time.time()
o_model, _ = train_conditional_score(
    x_t_data.reshape(-1, 1),
    x_tp1_data.reshape(-1, 1),
    dim=1,
    n_epochs=CONDITIONAL_SCORE_CONFIG["n_epochs"],
    hidden_dim=CONDITIONAL_SCORE_CONFIG["hidden_dim"],
    n_layers=CONDITIONAL_SCORE_CONFIG["n_layers"],
    device=DEVICE,
)
o_score = NNConditionalScore(o_model, dim=1, device=DEVICE)
print(f"  Time: {time.time() - t0:.1f}s\n")


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
            device=DEVICE,
        )
        st = torch.full((len(x_arr),), float(s), dtype=torch.float32, device=DEVICE)
        out = o_model(inp, st).cpu().numpy().reshape(-1)
    return np.clip(out, -o_score.clip, o_score.clip)


def reverse_sde_sample(x_cond, n_samples=1, S=S_MAX, n_steps=N_DIFF_STEPS):
    x_cond_arr = np.asarray(x_cond, dtype=np.float64).reshape(-1)
    n_cond = len(x_cond_arr)
    x_cond_rep = np.repeat(x_cond_arr, n_samples)
    ds = S / n_steps
    x = np.random.randn(len(x_cond_rep))
    for step in range(n_steps):
        s = max(S - step * ds, 1e-6)
        score = conditional_score_eval_1d(x, s, x_cond_rep)
        x = x + (0.5 * x + score) * ds + np.random.randn(len(x_cond_rep)) * np.sqrt(ds)
    if n_cond == 1:
        return x.reshape(n_samples)
    return x.reshape(n_cond, n_samples)


def exact_transition_sample(x_cond, n_samples=1):
    x_cond_arr = np.asarray(x_cond, dtype=np.float64).reshape(-1)
    n_cond = len(x_cond_arr)
    if n_cond == 1:
        x_scalar = float(np.squeeze(x_cond_arr[0]))
        return np.asarray(langevin_step(x_scalar, dt, n_samples=n_samples), dtype=np.float64).reshape(n_samples)
    x_rep = np.repeat(x_cond_arr, n_samples)
    return np.asarray(
        langevin_step(x_rep, dt, n_samples=len(x_rep)),
        dtype=np.float64,
    ).reshape(n_cond, n_samples)


def composition_reverse_sde(x_cond, n_samples=1, S=S_MAX, n_steps=N_DIFF_STEPS):
    x_cond_arr = np.asarray(x_cond, dtype=np.float64).reshape(-1)
    n_cond = len(x_cond_arr)
    x_cond_rep = np.repeat(x_cond_arr, n_samples)
    ds = S / n_steps
    x = np.random.randn(len(x_cond_rep))
    log_w = np.zeros(len(x_cond_rep))
    clip = 15.0

    for step in range(n_steps):
        s = max(S - step * ds, 1e-6)
        j_s = np.clip(j_score.score(x.reshape(-1, 1), s), -clip, clip).ravel()
        o_s = np.clip(conditional_score_eval_1d(x, s, x_cond_rep), -clip, clip).ravel()
        j_0 = np.clip(j_score.boltzmann_score(x.reshape(-1, 1)), -clip, clip).ravel()

        drift = np.clip(o_s - j_s + 0.5 * j_0 + 0.5 * x, -clip, clip)
        fkc_inc = 0.5 * dim + j_s * (j_s - o_s) + 0.5 * j_0 * (x + o_s - j_s)
        fkc_inc = np.clip(fkc_inc, -20 / S, 20 / S)
        log_w += fkc_inc * ds
        x = x + drift * ds + np.random.randn(len(x_cond_rep)) * np.sqrt(ds)

    log_w = np.nan_to_num(log_w, nan=0.0, posinf=20.0, neginf=-20.0)
    if n_cond == 1:
        return x.reshape(n_samples), log_w.reshape(n_samples)
    return x.reshape(n_cond, n_samples), log_w.reshape(n_cond, n_samples)


def sample_row_indices(log_weights, n_draws=1):
    log_weights = np.asarray(log_weights, dtype=np.float64)
    log_norm = logsumexp(log_weights, axis=1, keepdims=True)
    weights = np.exp(log_weights - log_norm)
    invalid = (~np.isfinite(weights)).any(axis=1) | (np.sum(weights, axis=1) < 1e-300)
    if np.any(invalid):
        weights[invalid] = 1.0 / weights.shape[1]
    cum = np.cumsum(weights, axis=1)
    u = np.random.rand(weights.shape[0], n_draws)
    idx = np.sum(u[..., None] > cum[:, None, :], axis=-1)
    idx = np.minimum(idx, weights.shape[1] - 1)
    if n_draws == 1:
        idx = idx.reshape(-1)
    return idx, log_norm.reshape(-1)


ALPHA_TWEEDIE = np.exp(-S_MAX / 2.0)
SIGMA2_TWEEDIE = 1.0 - np.exp(-S_MAX)


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
            x_hat = np.clip((z + SIGMA2_TWEEDIE * score) / ALPHA_TWEEDIE, -clip, clip)
        log_rewards[r] = -lam * (x_hat - x_target) ** 2
    return logsumexp(log_rewards, axis=0) - np.log(n_rollouts)


def sample_reference_trajectories(n_trajectories):
    trajs = np.zeros((n_trajectories, T + 1), dtype=np.float64)
    trajs[:, 0] = x_0
    x = np.full(n_trajectories, x_0, dtype=np.float64)
    for t_idx in range(T):
        x = exact_transition_sample(x, n_samples=1).reshape(-1)
        trajs[:, t_idx + 1] = x
    return trajs


def estimate_exact_log_Z(n_samples, batch_size=5000):
    rewards = np.zeros(n_samples, dtype=np.float64)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = end - start
        x = np.full(batch, x_0, dtype=np.float64)
        for _ in range(T):
            x = exact_transition_sample(x, n_samples=1).reshape(-1)
        rewards[start:end] = np.exp(-lam * (x - x_target) ** 2)
    return float(np.log(np.mean(rewards)))


print("Collecting 5000 reference trajectories...")
n_ref = 5000
ref_trajs = sample_reference_trajectories(n_ref)
future_targets = build_future_reward_targets(ref_trajs, T, endpoint_log_G)
print(f"  Mean future target at t=0: {future_targets[:, 0].mean():.6f}")
print(f"  log Z (MC estimate): {np.log(future_targets[:, 0].mean() + 1e-300):.4f}\n")


print("--- Phase 3: Ground truth ---\n")
print("Ground truth Z via 100k exact MC trajectories...")
n_gt = 100000
log_Z_gt = estimate_exact_log_Z(n_gt)
print(f"  log Z_gt (exact Langevin) = {log_Z_gt:.4f}")

print("Diffusion ground truth via 20k trajectories...")
t0 = time.time()
n_gt_diff = 20000
diff_final_x = np.full(n_gt_diff, x_0)
for t_step in range(T):
    diff_final_x = reverse_sde_sample(diff_final_x, n_samples=1).reshape(-1)
    if (t_step + 1) % 5 == 0:
        print(f"  step {t_step + 1}/{T} ({time.time() - t0:.1f}s)")
diff_rewards = np.exp(-lam * (diff_final_x - x_target) ** 2)
log_Z_gt_diff = np.log(np.mean(diff_rewards))
print(f"  log Z_gt (diffusion) = {log_Z_gt_diff:.4f}")
print(f"  Gap: {abs(log_Z_gt - log_Z_gt_diff):.4f} nats\n")


print("--- Phase 4: Training twists ---\n")
print("Training Take 3 twist (MC regression, positive psi)...")
t0 = time.time()
twist_mc_model, _ = train_positive_twist_mc(
    ref_trajs,
    future_targets,
    dim=1,
    T=T,
    n_epochs=3000,
    hidden_dim=64,
    device=DEVICE,
)
twist_mc = PositiveNNTwist(twist_mc_model, dim=1, device=DEVICE)
print(f"  Time: {time.time() - t0:.1f}s\n")

print("Training Take 3 twist (TD, positive psi)...")
t0 = time.time()
twist_td_model, _ = train_positive_twist_td(
    ref_trajs,
    dim=1,
    T=T,
    log_G_fn=endpoint_log_G,
    terminal_targets=np.ones(n_ref),
    n_epochs=3000,
    hidden_dim=64,
    device=DEVICE,
)
twist_td = PositiveNNTwist(twist_td_model, dim=1, device=DEVICE)
print(f"  Time: {time.time() - t0:.1f}s\n")


print("--- Phase 5: Running SMC methods ---\n")
results = {}

def run_bootstrap(label, step_sampler):
    print(f"Running {label}...")
    data = {"log_Z": [], "ess": [], "time": []}
    for trial in range(n_trials):
        np.random.seed(trial * 31)
        t0 = time.time()
        traj = np.zeros((K, T + 1))
        traj[:, 0] = x_0
        log_Z = 0.0
        ess_list = []
        for t in range(1, T + 1):
            traj[:, t] = np.asarray(step_sampler(traj[:, t - 1], n_samples=1)).reshape(K)
            log_w = endpoint_log_G(traj[:, t], t)
            log_Z += logsumexp(log_w) - np.log(K)
            ess_list.append(compute_ess(log_w))
            idx = systematic_resample(log_w)
            traj = traj[idx]
        data["time"].append(time.time() - t0)
        data["log_Z"].append(log_Z)
        data["ess"].append(np.mean(ess_list))
    print(f"  log Z = {np.mean(data['log_Z']):.4f} ± {np.std(data['log_Z']):.4f}, ESS = {np.mean(data['ess']):.1f}")
    return data


results["Bootstrap"] = run_bootstrap("Bootstrap", reverse_sde_sample)


def run_take1_tweedie(label, step_sampler):
    print(f"\nRunning {label}...")
    data = {"log_Z": [], "ess": [], "time": []}
    for trial in range(n_trials):
        np.random.seed(trial * 31)
        t0 = time.time()
        traj = np.zeros((K, T + 1))
        traj[:, 0] = x_0
        ess_list = []

        log_psi_init = float(tweedie_twist_batch_fast_1d(np.array([x_0]), 0, n_rollouts=3)[0])
        log_Z = log_psi_init
        log_psi_cached = np.full(K, log_psi_init)

        for t in range(1, T + 1):
            proposals = np.asarray(step_sampler(traj[:, t - 1], n_samples=M)).reshape(K, M)
            log_G = endpoint_log_G(proposals.reshape(-1), t).reshape(K, M)
            if t < T:
                log_twist = tweedie_twist_batch_fast_1d(proposals.reshape(-1), t, n_rollouts=1).reshape(K, M)
            else:
                log_twist = np.zeros((K, M))
            log_v = log_G + log_twist
            j_sel, log_norm = sample_row_indices(log_v, n_draws=1)
            traj[:, t] = proposals[np.arange(K), j_sel]
            log_weights = log_norm - np.log(M) - log_psi_cached
            log_psi_next = log_twist[np.arange(K), j_sel] if t < T else np.zeros(K)
            log_Z += logsumexp(log_weights) - np.log(K)
            ess_list.append(compute_ess(log_weights))
            idx = systematic_resample(log_weights)
            traj = traj[idx]
            log_psi_cached = log_psi_next[idx]

        data["time"].append(time.time() - t0)
        data["log_Z"].append(log_Z)
        data["ess"].append(np.mean(ess_list))
    print(f"  log Z = {np.mean(data['log_Z']):.4f} ± {np.std(data['log_Z']):.4f}, ESS = {np.mean(data['ess']):.1f}")
    return data


results["Take1 (Tweedie)"] = run_take1_tweedie("Take1 (Tweedie)", reverse_sde_sample)


def run_take3(label, twist_model, step_sampler=reverse_sde_sample, target_dict=results):
    print(f"\nRunning {label}...")
    data = {"log_Z": [], "ess": [], "time": []}
    for trial in range(n_trials):
        np.random.seed(trial * 31)
        t0 = time.time()
        traj = np.zeros((K, T + 1))
        traj[:, 0] = x_0
        ess_list = []
        log_psi_cached = np.full(K, float(twist_model(np.array([x_0]), 0)))
        log_Z = log_psi_cached[0]

        for t in range(1, T + 1):
            proposals = np.asarray(step_sampler(traj[:, t - 1], n_samples=M)).reshape(K, M)
            log_G = endpoint_log_G(proposals.reshape(-1), t).reshape(K, M)
            if t < T:
                log_twist = np.asarray(twist_model(proposals.reshape(-1, 1), t)).reshape(K, M)
            else:
                log_twist = np.zeros((K, M))
            log_v = log_G + log_twist
            j_sel, log_norm = sample_row_indices(log_v, n_draws=1)
            traj[:, t] = proposals[np.arange(K), j_sel]
            log_weights = log_norm - np.log(M) - log_psi_cached
            log_psi_next = log_twist[np.arange(K), j_sel] if t < T else np.zeros(K)
            log_Z += logsumexp(log_weights) - np.log(K)
            ess_list.append(compute_ess(log_weights))
            idx = systematic_resample(log_weights)
            traj = traj[idx]
            log_psi_cached = log_psi_next[idx]

        data["time"].append(time.time() - t0)
        data["log_Z"].append(log_Z)
        data["ess"].append(np.mean(ess_list))

    target_dict[label] = data
    print(f"  log Z = {np.mean(data['log_Z']):.4f} ± {np.std(data['log_Z']):.4f}, ESS = {np.mean(data['ess']):.1f}")


run_take3("Take3 (MC)", twist_mc)
run_take3("Take3 (TD)", twist_td)


print(f"\nRunning Take 2 (two-stage, {n_trials_t2} trials)...")
data = {"log_Z": [], "ess": [], "time": []}
for trial in range(n_trials_t2):
    np.random.seed(trial * 31)
    t0 = time.time()
    traj = np.zeros((K, T + 1))
    traj[:, 0] = x_0
    ess_list = []

    log_psi_init = float(tweedie_twist_batch_fast_1d(np.array([x_0]), 0, n_rollouts=3)[0])
    log_Z = log_psi_init
    log_psi_cached = np.full(K, log_psi_init)

    for t in range(1, T + 1):
        x_comp, log_fkc = composition_reverse_sde(traj[:, t - 1], n_samples=M)
        fkc_idx, _ = sample_row_indices(log_fkc, n_draws=M)
        x_resampled = np.take_along_axis(x_comp, fkc_idx, axis=1)
        log_G = endpoint_log_G(x_resampled.reshape(-1), t).reshape(K, M)
        if t < T:
            log_twist = tweedie_twist_batch_fast_1d(x_resampled.reshape(-1), t, n_rollouts=1).reshape(K, M)
        else:
            log_twist = np.zeros((K, M))
        log_v = log_G + log_twist
        j_sel, log_norm = sample_row_indices(log_v, n_draws=1)
        traj[:, t] = x_resampled[np.arange(K), j_sel]
        log_weights = log_norm - np.log(M) - log_psi_cached
        log_psi_next = log_twist[np.arange(K), j_sel] if t < T else np.zeros(K)
        log_Z += logsumexp(log_weights) - np.log(K)
        ess_list.append(compute_ess(log_weights))
        idx = systematic_resample(log_weights)
        traj = traj[idx]
        log_psi_cached = log_psi_next[idx]

    data["time"].append(time.time() - t0)
    data["log_Z"].append(log_Z)
    data["ess"].append(np.mean(ess_list))
results["Take2 (two-stage)"] = data
print(f"  log Z = {np.mean(data['log_Z']):.4f} ± {np.std(data['log_Z']):.4f}, ESS = {np.mean(data['ess']):.1f}")


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Ground truth (exact Langevin): {log_Z_gt:.4f}")
print(f"Ground truth (diffusion):      {log_Z_gt_diff:.4f}")
for key in ["Bootstrap", "Take1 (Tweedie)", "Take2 (two-stage)", "Take3 (MC)", "Take3 (TD)"]:
    vals = results[key]
    print(f"{key:<20} log Z = {np.mean(vals['log_Z']):.4f} ± {np.std(vals['log_Z']):.4f}, "
          f"ESS = {np.mean(vals['ess']):.1f}")

exact_results = {}
exact_results["Bootstrap"] = run_bootstrap("Bootstrap (exact drift)", exact_transition_sample)
exact_results["Take1 (Tweedie)"] = run_take1_tweedie("Take1 (Tweedie, exact drift)", exact_transition_sample)
run_take3("Take3 (MC)", twist_mc, step_sampler=exact_transition_sample, target_dict=exact_results)
run_take3("Take3 (TD)", twist_td, step_sampler=exact_transition_sample, target_dict=exact_results)

output = {
    "experiment": "run_exp1",
    "output_path": OUTPUT_PATH,
    "parameters": {
        "dim": dim,
        "dt": dt,
        "T": T,
        "x_0": x_0,
        "x_target": x_target,
        "lam": lam,
        "K": K,
        "M": M,
        "S_MAX": S_MAX,
        "N_DIFF_STEPS": N_DIFF_STEPS,
        "n_trials": n_trials,
        "n_trials_t2": n_trials_t2,
        "marginal_score_config": MARGINAL_SCORE_CONFIG,
        "conditional_score_config": CONDITIONAL_SCORE_CONFIG,
    },
    "ground_truth": {
        "exact_langevin_log_Z": log_Z_gt,
        "diffusion_log_Z": log_Z_gt_diff,
        "discretization_gap_nats": abs(log_Z_gt - log_Z_gt_diff),
    },
    "methods": results,
    "exact_drift_methods": exact_results,
    "summary": {
        key: {
            "log_Z": mean_std(vals["log_Z"]),
            "ess": mean_std(vals["ess"]),
            "time": mean_std(vals["time"]),
        }
        for key, vals in results.items()
    },
    "exact_drift_summary": {
        key: {
            "log_Z": mean_std(vals["log_Z"]),
            "ess": mean_std(vals["ess"]),
            "time": mean_std(vals["time"]),
        }
        for key, vals in exact_results.items()
    },
}
save_json(output, OUTPUT_PATH)
print(f"\nSaved JSON results to {OUTPUT_PATH}")
