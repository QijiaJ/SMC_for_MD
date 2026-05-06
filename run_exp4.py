"""
Updated Experiment 4: single-stage vs two-stage Take 2.

Fixes relative to the original script:
- Learned twists are trained as positive psi_t and only logged at evaluation.
- The 2D upsampling twist uses future-only observation rewards, consistent with
  the separate log_G + log_twist decomposition in SMC.
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

from updated_code.learned_scores import (
    train_marginal_score,
    train_conditional_score,
    NNMarginalScore,
    NNConditionalScore,
)

from updated_code.common import (
    systematic_resample,
    resample_multinomial,
    compute_ess,
    build_future_reward_targets,
    sample_row_indices,
    mean_std,
    save_json,
)
from updated_code.fixed_twist import train_positive_twist_mc, PositiveNNTwist

np.random.seed(42)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "results", "run_exp4.json")


print("=" * 70)
print("UPDATED EXPERIMENT 4: Single-stage vs Two-stage Take 2")
print("=" * 70)

PART_A_SCORE_CONFIG = {
    "marginal": {
        "n_eq_samples": 16000,
        "hidden_dim": 160,
        "n_layers": 4,
        "n_epochs": 4000,
    },
    "conditional": {
        "n_transition_pairs": 8000,
        "hidden_dim": 128,
        "n_layers": 4,
        "n_epochs": 2500,
    },
}
PART_B_SCORE_CONFIG = {
    "marginal": {
        "n_eq_per_well": 3000,
        "hidden_dim": 160,
        "n_layers": 4,
        "n_epochs": 4000,
    },
    "conditional": {
        "n_transition_pairs": 3000,
        "hidden_dim": 128,
        "n_layers": 4,
        "n_epochs": 2000,
    },
}


def systematic_resample_local(log_weights):
    return systematic_resample(log_weights)


def resample_multinomial_local(log_weights, n_samples):
    return resample_multinomial(log_weights, n_samples)


def conditional_score_eval(score_wrapper, x_batch, s, x_cond_batch, dim):
    x_arr = np.asarray(x_batch, dtype=np.float64).reshape(-1, dim)
    x_cond_arr = np.asarray(x_cond_batch, dtype=np.float64)
    if x_cond_arr.ndim == 1:
        x_cond_arr = x_cond_arr.reshape(1, dim)
    else:
        x_cond_arr = x_cond_arr.reshape(-1, dim)
    if len(x_cond_arr) == 1 and len(x_arr) > 1:
        x_cond_arr = np.tile(x_cond_arr, (len(x_arr), 1))
    elif len(x_cond_arr) != len(x_arr):
        raise ValueError("x_batch and x_cond_batch must have matching batch sizes")

    with torch.no_grad():
        inp = torch.tensor(
            np.concatenate([x_arr, x_cond_arr], axis=-1),
            dtype=torch.float32,
            device=score_wrapper.device,
        )
        st = torch.full((len(x_arr),), float(s), dtype=torch.float32, device=score_wrapper.device)
        out = score_wrapper.model(inp, st).cpu().numpy()
    return np.clip(out, -score_wrapper.clip, score_wrapper.clip)


print("\nPART A: 6D Coupled Double-Well")

D_hd = 6
COUPLING_HD = 0.2
dt_hd = 0.05
T_hd = 10
x_0_hd = -np.ones(D_hd)
x_target_hd = np.ones(D_hd)
lam_hd = 0.35
K_hd = 100
S_MAX_hd = 2.0
N_DIFF_hd = 25
n_trials_hd = 20
SYSTEM_NAME_HD = "6D coupled double-well"


def U_hd(x):
    x = np.asarray(x, dtype=np.float64)
    base = np.sum((x ** 2 - 1.0) ** 2, axis=-1)
    coupling = COUPLING_HD * np.sum(x[..., :-1] * x[..., 1:], axis=-1)
    return base + coupling


def grad_U_hd(x):
    x = np.asarray(x, dtype=np.float64)
    grad = 4.0 * x * (x ** 2 - 1.0)
    grad[..., :-1] += COUPLING_HD * x[..., 1:]
    grad[..., 1:] += COUPLING_HD * x[..., :-1]
    return grad


def langevin_step_hd(x, dt=None, grad_clip=12.0):
    if dt is None:
        dt = dt_hd
    x = np.asarray(x, dtype=np.float64)
    scalar = x.ndim == 1
    if scalar:
        x = x[None, :]
    grad = np.clip(grad_U_hd(x), -grad_clip, grad_clip)
    noise = np.random.randn(*x.shape) * np.sqrt(2.0 * dt)
    x_new = x - dt * grad + noise
    return x_new[0] if scalar else x_new


def sample_trajectory_hd(x_0, T, dt=None):
    if dt is None:
        dt = dt_hd
    traj = np.zeros((T + 1, D_hd))
    traj[0] = np.asarray(x_0, dtype=np.float64)
    for t in range(T):
        traj[t + 1] = langevin_step_hd(traj[t], dt)
    return traj


def sample_equilibrium_hd(n_samples, dt=0.01, n_burnin=12000, thin=40):
    x = x_0_hd.copy() + 0.1 * np.random.randn(D_hd)
    for _ in range(n_burnin):
        x = langevin_step_hd(x, dt)
    samples = np.zeros((n_samples, D_hd))
    for i in range(n_samples):
        for _ in range(thin):
            x = langevin_step_hd(x, dt)
        samples[i] = x
    return samples


def endpoint_log_G_hd(x, t):
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if t == T_hd:
        return -lam_hd * np.sum((x - x_target_hd) ** 2, axis=-1)
    return np.zeros(x.shape[0])


def composition_sde_hd(x_cond, n_samples, S=S_MAX_hd, n_steps=N_DIFF_hd):
    x_cond_arr = np.asarray(x_cond, dtype=np.float64)
    if x_cond_arr.ndim == 1:
        x_cond_arr = x_cond_arr.reshape(1, D_hd)
    else:
        x_cond_arr = x_cond_arr.reshape(-1, D_hd)
    n_cond = len(x_cond_arr)
    x_cond_rep = np.repeat(x_cond_arr, n_samples, axis=0)
    ds = S / n_steps
    x = np.random.randn(len(x_cond_rep), D_hd)
    log_w = np.zeros(len(x_cond_rep))
    clip = 12.0

    for step in range(n_steps):
        s = max(S - step * ds, 1e-6)
        x = np.clip(x, -5, 5)
        j_s = np.clip(np.atleast_2d(j_score_hd.score(x, s)), -clip, clip)
        o_s = np.clip(conditional_score_eval(o_score_hd, x, s, x_cond_rep, D_hd), -clip, clip)
        j_0 = np.clip(np.atleast_2d(j_score_hd.boltzmann_score(x)), -clip, clip)
        j_s = np.nan_to_num(j_s, nan=0.0)
        o_s = np.nan_to_num(o_s, nan=0.0)
        j_0 = np.nan_to_num(j_0, nan=0.0)

        drift = np.clip(o_s - j_s + 0.5 * j_0 + 0.5 * x, -clip, clip)
        fkc_inc = (
            0.5 * D_hd
            + np.sum(j_s * (j_s - o_s), axis=-1)
            + 0.5 * np.sum(j_0 * (x + o_s - j_s), axis=-1)
        )
        fkc_inc = np.clip(fkc_inc, -20 / S, 20 / S)
        log_w += fkc_inc * ds
        x = x + drift * ds + np.random.randn(len(x_cond_rep), D_hd) * np.sqrt(ds)

    log_w = np.nan_to_num(log_w, nan=0.0, posinf=20.0, neginf=-20.0)
    if n_cond == 1:
        return x.reshape(n_samples, D_hd), log_w.reshape(n_samples)
    return x.reshape(n_cond, n_samples, D_hd), log_w.reshape(n_cond, n_samples)


def sample_fkc_resampled_step_hd(x_cond_batch, inner_M):
    x_cond_arr = np.asarray(x_cond_batch, dtype=np.float64)
    scalar = x_cond_arr.ndim == 1
    if scalar:
        x_cond_arr = x_cond_arr.reshape(1, D_hd)
    x_props, log_fkc = composition_sde_hd(x_cond_arr, inner_M)
    row_idx, _ = sample_row_indices(log_fkc, n_draws=1)
    x_next = x_props[np.arange(len(x_cond_arr)), row_idx]
    return x_next[0] if scalar else x_next


def estimate_log_Z_hd_from_step(step_fn, n_samples, batch_size=256):
    log_rewards = np.zeros(n_samples, dtype=np.float64)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = end - start
        x = np.tile(x_0_hd, (batch, 1))
        for _ in range(T_hd):
            x = np.asarray(step_fn(x)).reshape(batch, D_hd)
        log_rewards[start:end] = endpoint_log_G_hd(x, T_hd)
    return float(logsumexp(log_rewards) - np.log(n_samples))


def take2_hd_single_stage(x_0, T, K, M, twist_fn):
    traj = np.zeros((K, T + 1, D_hd))
    traj[:, 0] = x_0
    ess_per_step = []
    fkc_ess_per_step = []

    log_psi_prev = np.full(K, float(twist_fn(np.array([x_0]), 0)))
    log_Z = log_psi_prev[0]

    for t in range(1, T + 1):
        x_proposals, log_fkc = composition_sde_hd(traj[:, t - 1], M)
        fkc_ess_list = [compute_ess(row) for row in log_fkc]
        log_G = endpoint_log_G_hd(x_proposals.reshape(-1, D_hd), t).reshape(K, M)
        if t < T:
            log_twist = np.asarray(twist_fn(x_proposals.reshape(-1, D_hd), t)).reshape(K, M)
        else:
            log_twist = np.zeros((K, M))
        log_v = log_fkc + log_G + log_twist
        j_sel, log_norm = sample_row_indices(log_v, n_draws=1)
        traj[:, t] = x_proposals[np.arange(K), j_sel]
        log_weights = log_norm - np.log(M) - log_psi_prev
        log_psi_prev = np.array([float(twist_fn(traj[k, t:t + 1], t)) for k in range(K)]) if t < T else np.zeros(K)
        log_Z += logsumexp(log_weights) - np.log(K)
        ess_per_step.append(compute_ess(log_weights))
        fkc_ess_per_step.append(np.mean(fkc_ess_list))
        indices = systematic_resample_local(log_weights)
        traj = traj[indices]
        if t < T:
            log_psi_prev = log_psi_prev[indices]

    return {"log_Z": log_Z, "ess_per_step": ess_per_step, "fkc_ess": fkc_ess_per_step}


def take2_hd_two_stage(x_0, T, K, M, twist_fn):
    traj = np.zeros((K, T + 1, D_hd))
    traj[:, 0] = x_0
    ess_per_step = []
    fkc_ess_per_step = []

    log_psi_prev = np.full(K, float(twist_fn(np.array([x_0]), 0)))
    log_Z = log_psi_prev[0]

    for t in range(1, T + 1):
        x_proposals, log_fkc = composition_sde_hd(traj[:, t - 1], M)
        fkc_ess_list = [compute_ess(row) for row in log_fkc]
        resamp_idx, _ = sample_row_indices(log_fkc, n_draws=M)
        x_resampled = np.take_along_axis(x_proposals, resamp_idx[..., None], axis=1)
        log_G = endpoint_log_G_hd(x_resampled.reshape(-1, D_hd), t).reshape(K, M)
        if t < T:
            log_twist = np.asarray(twist_fn(x_resampled.reshape(-1, D_hd), t)).reshape(K, M)
        else:
            log_twist = np.zeros((K, M))
        log_v = log_G + log_twist
        j_sel, log_norm = sample_row_indices(log_v, n_draws=1)
        traj[:, t] = x_resampled[np.arange(K), j_sel]
        log_weights = log_norm - np.log(M) - log_psi_prev
        log_psi_prev = np.array([float(twist_fn(traj[k, t:t + 1], t)) for k in range(K)]) if t < T else np.zeros(K)
        log_Z += logsumexp(log_weights) - np.log(K)
        ess_per_step.append(compute_ess(log_weights))
        fkc_ess_per_step.append(np.mean(fkc_ess_list))
        indices = systematic_resample_local(log_weights)
        traj = traj[indices]
        if t < T:
            log_psi_prev = log_psi_prev[indices]

    return {"log_Z": log_Z, "ess_per_step": ess_per_step, "fkc_ess": fkc_ess_per_step}


print("Collecting high-dimensional training data...")
eq_samples_hd = sample_equilibrium_hd(PART_A_SCORE_CONFIG["marginal"]["n_eq_samples"])
n_pairs_hd = PART_A_SCORE_CONFIG["conditional"]["n_transition_pairs"]
xt_hd = np.zeros((n_pairs_hd, D_hd))
xtp1_hd = np.zeros((n_pairs_hd, D_hd))
for i in range(n_pairs_hd // 2):
    xt_hd[i] = eq_samples_hd[np.random.randint(len(eq_samples_hd))]
    xtp1_hd[i] = langevin_step_hd(xt_hd[i], dt_hd)
for i in range(n_pairs_hd // 2, n_pairs_hd):
    x = x_0_hd.copy()
    for _ in range(np.random.randint(0, T_hd)):
        x = langevin_step_hd(x, dt_hd)
    xt_hd[i] = x
    xtp1_hd[i] = langevin_step_hd(x, dt_hd)

n_ref_hd = 5000
ref_trajs_hd = np.zeros((n_ref_hd, T_hd + 1, D_hd))
ref_trajs_hd[:, 0] = x_0_hd
x_hd = np.tile(x_0_hd, (n_ref_hd, 1))
for t_idx in range(T_hd):
    x_hd = langevin_step_hd(x_hd, dt_hd)
    ref_trajs_hd[:, t_idx + 1] = x_hd
future_targets_hd = build_future_reward_targets(ref_trajs_hd, T_hd, endpoint_log_G_hd)

print("Training high-dimensional NN scores...")
t0 = time.time()
j_model_hd, _ = train_marginal_score(
    eq_samples_hd,
    dim=D_hd,
    n_epochs=PART_A_SCORE_CONFIG["marginal"]["n_epochs"],
    hidden_dim=PART_A_SCORE_CONFIG["marginal"]["hidden_dim"],
    n_layers=PART_A_SCORE_CONFIG["marginal"]["n_layers"],
    device=DEVICE,
)
j_score_hd = NNMarginalScore(j_model_hd, dim=D_hd, device=DEVICE)
o_model_hd, _ = train_conditional_score(
    xt_hd,
    xtp1_hd,
    dim=D_hd,
    n_epochs=PART_A_SCORE_CONFIG["conditional"]["n_epochs"],
    hidden_dim=PART_A_SCORE_CONFIG["conditional"]["hidden_dim"],
    n_layers=PART_A_SCORE_CONFIG["conditional"]["n_layers"],
    device=DEVICE,
)
o_score_hd = NNConditionalScore(o_model_hd, dim=D_hd, device=DEVICE)
print(f"  Done in {time.time() - t0:.1f}s")

print("Training high-dimensional twist...")
t0 = time.time()
twist_model_hd, _ = train_positive_twist_mc(
    ref_trajs_hd,
    future_targets_hd,
    dim=D_hd,
    T=T_hd,
    n_epochs=3000,
    hidden_dim=128,
    device=DEVICE,
)
twist_hd = PositiveNNTwist(twist_model_hd, dim=D_hd, device=DEVICE)
print(f"  Done in {time.time() - t0:.1f}s")

n_gt_hd = 20000
gt_rewards_hd = np.zeros(n_gt_hd)
x_hd = np.tile(x_0_hd, (n_gt_hd, 1))
for _ in range(T_hd):
    x_hd = langevin_step_hd(x_hd, dt_hd)
gt_rewards_hd = np.exp(endpoint_log_G_hd(x_hd, T_hd))
Z_gt_hd = np.mean(gt_rewards_hd)
log_Z_gt_hd = np.log(Z_gt_hd)
GT_PROPOSAL_M_HD = 32
n_gt_hd_learned = 4000
log_Z_gt_hd_learned = estimate_log_Z_hd_from_step(
    lambda x: sample_fkc_resampled_step_hd(x, GT_PROPOSAL_M_HD),
    n_gt_hd_learned,
)
Z_gt_hd_learned = float(np.exp(log_Z_gt_hd_learned))
print(f"  true-dynamics log Z     = {log_Z_gt_hd:.4f}")
print(f"  learned-proposal log Z = {log_Z_gt_hd_learned:.4f}")

part_a_results = {}
for M_val in [5, 10, 20]:
    print(f"\n--- 6D: M={M_val} ---")
    ss_logZ, ts_logZ = [], []
    ss_fkc, ts_fkc = [], []
    for trial in range(n_trials_hd):
        np.random.seed(trial * 31)
        res = take2_hd_single_stage(x_0_hd, T_hd, K_hd, M_val, twist_hd)
        ss_logZ.append(res["log_Z"])
        ss_fkc.append(np.mean(res["fkc_ess"]))
    for trial in range(n_trials_hd):
        np.random.seed(trial * 31)
        res = take2_hd_two_stage(x_0_hd, T_hd, K_hd, M_val, twist_hd)
        ts_logZ.append(res["log_Z"])
        ts_fkc.append(np.mean(res["fkc_ess"]))

    print(f"  Single-stage: log Z = {np.mean(ss_logZ):.4f} +/- {np.std(ss_logZ):.4f}, FKC ESS = {np.mean(ss_fkc):.2f}")
    print(f"  Two-stage:    log Z = {np.mean(ts_logZ):.4f} +/- {np.std(ts_logZ):.4f}, FKC ESS = {np.mean(ts_fkc):.2f}")
    print(
        f"  Bias (learned GT): SS={np.mean(np.exp(ss_logZ)) - Z_gt_hd_learned:.6f}, "
        f"TS={np.mean(np.exp(ts_logZ)) - Z_gt_hd_learned:.6f}"
    )
    part_a_results[f"M_{M_val}"] = {
        "single_stage": {
            "log_Z": ss_logZ,
            "fkc_ess": ss_fkc,
            "summary": {
                "log_Z": mean_std(ss_logZ),
                "fkc_ess": mean_std(ss_fkc),
                "bias_on_Z_scale": float(np.mean(np.exp(ss_logZ)) - Z_gt_hd_learned),
                "bias_on_Z_scale_true": float(np.mean(np.exp(ss_logZ)) - Z_gt_hd),
            },
        },
        "two_stage": {
            "log_Z": ts_logZ,
            "fkc_ess": ts_fkc,
            "summary": {
                "log_Z": mean_std(ts_logZ),
                "fkc_ess": mean_std(ts_fkc),
                "bias_on_Z_scale": float(np.mean(np.exp(ts_logZ)) - Z_gt_hd_learned),
                "bias_on_Z_scale_true": float(np.mean(np.exp(ts_logZ)) - Z_gt_hd),
            },
        },
    }


print("\nPART B: 2D Coupled Double-Well Upsampling")


def U_2d(x):
    x = np.asarray(x, dtype=np.float64)
    xx, yy = x[..., 0], x[..., 1]
    return (xx ** 2 - 1) ** 2 + yy ** 2 + 0.5 * xx * yy


def grad_U_2d(x):
    x = np.asarray(x, dtype=np.float64)
    xx, yy = x[..., 0], x[..., 1]
    dU_dx = 4 * xx * (xx ** 2 - 1) + 0.5 * yy
    dU_dy = 2 * yy + 0.5 * xx
    return np.stack([dU_dx, dU_dy], axis=-1)


WELL_R = np.array([1.0, -0.25])
WELL_L = np.array([-1.0, 0.25])
dt_2d = 0.02


def langevin_step_2d(x, dt=None, grad_clip=10.0):
    if dt is None:
        dt = dt_2d
    x = np.asarray(x, dtype=np.float64)
    scalar = x.ndim == 1
    if scalar:
        x = x[None, :]
    g = np.clip(grad_U_2d(x), -grad_clip, grad_clip)
    noise = np.random.randn(*x.shape) * np.sqrt(2.0 * dt)
    x_new = x - dt * g + noise
    return x_new[0] if scalar else x_new


def sample_trajectory_2d(x_0, T, dt=None):
    if dt is None:
        dt = dt_2d
    traj = np.zeros((T + 1, 2))
    traj[0] = np.asarray(x_0)
    for t in range(T):
        traj[t + 1] = langevin_step_2d(traj[t], dt)
    return traj


class UpsamplingObs:
    def __init__(self, gt_traj, obs_interval=2, lam=8.0):
        self.gt = gt_traj
        self.T = len(gt_traj) - 1
        self.interval = obs_interval
        self.lam = lam
        self.obs_times = set(range(0, self.T + 1, obs_interval))

    def log_G(self, x, t):
        x = np.atleast_2d(x)
        if t in self.obs_times:
            diff = x - self.gt[t]
            return -self.lam * np.sum(diff ** 2, axis=-1)
        return np.zeros(x.shape[0])


T_2d = 20
K_2d = 50
M_2d = 5
S_MAX_2d = 1.5
N_DIFF_2d = 20
n_trials_2d = 5

print("Collecting 2D training data...")
eq_2d = []
for start in [WELL_R, WELL_L]:
    x = start.copy() + 0.02 * np.random.randn(2)
    for _ in range(3000):
        x = langevin_step_2d(x, 0.005)
    for _ in range(PART_B_SCORE_CONFIG["marginal"]["n_eq_per_well"]):
        for _ in range(15):
            x = langevin_step_2d(x, 0.005)
        eq_2d.append(x.copy())
eq_2d = np.array(eq_2d)

n_walkers = 200
walkers_2d = np.zeros((n_walkers, 2))
for i in range(n_walkers):
    walkers_2d[i] = [WELL_R, WELL_L][i % 2].copy() + 0.05 * np.random.randn(2)
for _ in range(5000):
    walkers_2d = langevin_step_2d(walkers_2d, 0.005)
xt_2d_list, xtp1_2d_list = [], []
pairs_per_chunk = (PART_B_SCORE_CONFIG["conditional"]["n_transition_pairs"] + n_walkers - 1) // n_walkers
for _ in range(pairs_per_chunk):
    for _ in range(20):
        walkers_2d = langevin_step_2d(walkers_2d, 0.005)
    xt_2d_list.append(walkers_2d.copy())
    xtp1_2d_list.append(langevin_step_2d(walkers_2d, dt_2d))
xt_2d = np.concatenate(xt_2d_list, axis=0)[:PART_B_SCORE_CONFIG["conditional"]["n_transition_pairs"]]
xtp1_2d = np.concatenate(xtp1_2d_list, axis=0)[:PART_B_SCORE_CONFIG["conditional"]["n_transition_pairs"]]

print("Training 2D NN scores...")
t0 = time.time()
j_model_2d, _ = train_marginal_score(
    eq_2d,
    dim=2,
    n_epochs=PART_B_SCORE_CONFIG["marginal"]["n_epochs"],
    hidden_dim=PART_B_SCORE_CONFIG["marginal"]["hidden_dim"],
    n_layers=PART_B_SCORE_CONFIG["marginal"]["n_layers"],
    device=DEVICE,
)
j_score_2d = NNMarginalScore(j_model_2d, dim=2, device=DEVICE)
o_model_2d, _ = train_conditional_score(
    xt_2d,
    xtp1_2d,
    dim=2,
    n_epochs=PART_B_SCORE_CONFIG["conditional"]["n_epochs"],
    hidden_dim=PART_B_SCORE_CONFIG["conditional"]["hidden_dim"],
    n_layers=PART_B_SCORE_CONFIG["conditional"]["n_layers"],
    device=DEVICE,
)
o_score_2d = NNConditionalScore(o_model_2d, dim=2, device=DEVICE)
print(f"  Done in {time.time() - t0:.1f}s")

np.random.seed(42)
gt_traj = sample_trajectory_2d(WELL_R, T_2d)
obs_up = UpsamplingObs(gt_traj, obs_interval=2, lam=8.0)
missing_times = [t for t in range(T_2d + 1) if t not in obs_up.obs_times]

n_gt_2d = 5000
x = np.tile(gt_traj[0], (n_gt_2d, 1))
traj_batch_2d = np.zeros((n_gt_2d, T_2d + 1, 2), dtype=np.float64)
traj_batch_2d[:, 0] = x
for t_idx in range(T_2d):
    x = langevin_step_2d(x, dt_2d)
    traj_batch_2d[:, t_idx + 1] = x
gt_lr_2d = np.zeros(n_gt_2d, dtype=np.float64)
for t_idx in range(T_2d + 1):
    gt_lr_2d += obs_up.log_G(traj_batch_2d[:, t_idx], t_idx)
log_Z_gt_2d = logsumexp(gt_lr_2d) - np.log(n_gt_2d)
print(f"  true-dynamics log Z     = {log_Z_gt_2d:.4f}")

print("Training 2D twist...")
n_ref_2d = 5000
ref_trajs_2d = np.zeros((n_ref_2d, T_2d + 1, 2), dtype=np.float64)
x = np.tile(gt_traj[0], (n_ref_2d, 1))
ref_trajs_2d[:, 0] = x
for t_idx in range(T_2d):
    x = langevin_step_2d(x, dt_2d)
    ref_trajs_2d[:, t_idx + 1] = x
future_targets_2d = build_future_reward_targets(ref_trajs_2d, T_2d, obs_up.log_G)
t0 = time.time()
twist_model_2d, _ = train_positive_twist_mc(
    ref_trajs_2d,
    future_targets_2d,
    dim=2,
    T=T_2d,
    n_epochs=3000,
    hidden_dim=128,
    device=DEVICE,
)
twist_2d = PositiveNNTwist(twist_model_2d, dim=2, device=DEVICE)
print(f"  Done in {time.time() - t0:.1f}s")


def composition_sde_2d(x_cond, n_samples, S=S_MAX_2d, n_steps=N_DIFF_2d):
    x_cond_arr = np.asarray(x_cond, dtype=np.float64)
    if x_cond_arr.ndim == 1:
        x_cond_arr = x_cond_arr.reshape(1, 2)
    else:
        x_cond_arr = x_cond_arr.reshape(-1, 2)
    n_cond = len(x_cond_arr)
    x_cond_rep = np.repeat(x_cond_arr, n_samples, axis=0)
    ds = S / n_steps
    x = np.random.randn(len(x_cond_rep), 2)
    log_w = np.zeros(len(x_cond_rep))
    clip = 10.0

    for step in range(n_steps):
        s = max(S - step * ds, 1e-6)
        x = np.clip(x, -5, 5)
        j_s = np.clip(np.atleast_2d(j_score_2d.score(x, s)), -clip, clip)
        o_s = np.clip(conditional_score_eval(o_score_2d, x, s, x_cond_rep, 2), -clip, clip)
        j_0 = np.clip(np.atleast_2d(j_score_2d.boltzmann_score(x)), -clip, clip)
        j_s = np.nan_to_num(j_s, nan=0.0)
        o_s = np.nan_to_num(o_s, nan=0.0)
        j_0 = np.nan_to_num(j_0, nan=0.0)

        drift = np.clip(o_s - j_s + 0.5 * j_0 + 0.5 * x, -clip, clip)
        fkc_inc = 1.0 + np.sum(j_s * (j_s - o_s), axis=-1) + 0.5 * np.sum(j_0 * (x + o_s - j_s), axis=-1)
        fkc_inc = np.clip(fkc_inc, -20 / S, 20 / S)
        log_w += fkc_inc * ds
        x = x + drift * ds + np.random.randn(len(x_cond_rep), 2) * np.sqrt(ds)

    log_w = np.nan_to_num(log_w, nan=0.0, posinf=20.0, neginf=-20.0)
    if n_cond == 1:
        return x.reshape(n_samples, 2), log_w.reshape(n_samples)
    return x.reshape(n_cond, n_samples, 2), log_w.reshape(n_cond, n_samples)


def sample_fkc_resampled_step_2d(x_cond_batch, inner_M):
    x_cond_arr = np.asarray(x_cond_batch, dtype=np.float64)
    scalar = x_cond_arr.ndim == 1
    if scalar:
        x_cond_arr = x_cond_arr.reshape(1, 2)
    x_props, log_fkc = composition_sde_2d(x_cond_arr, inner_M)
    row_idx, _ = sample_row_indices(log_fkc, n_draws=1)
    x_next = x_props[np.arange(len(x_cond_arr)), row_idx]
    return x_next[0] if scalar else x_next


def estimate_log_Z_2d_from_step(obs_model, step_fn, n_samples, batch_size=256):
    log_rewards = np.zeros(n_samples, dtype=np.float64)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = end - start
        trajs = np.zeros((batch, T_2d + 1, 2), dtype=np.float64)
        x = np.tile(gt_traj[0], (batch, 1))
        trajs[:, 0] = x
        for t_idx in range(T_2d):
            x = np.asarray(step_fn(x)).reshape(batch, 2)
            trajs[:, t_idx + 1] = x
        step_logs = [obs_model.log_G(trajs[:, t_idx], t_idx) for t_idx in range(T_2d + 1)]
        log_rewards[start:end] = np.sum(np.stack(step_logs, axis=1), axis=1)
    return float(logsumexp(log_rewards) - np.log(n_samples))


GT_PROPOSAL_M_2D = 16
n_gt_2d_learned = 3000
log_Z_gt_2d_learned = estimate_log_Z_2d_from_step(
    obs_up,
    lambda x: sample_fkc_resampled_step_2d(x, GT_PROPOSAL_M_2D),
    n_gt_2d_learned,
)
print(f"  learned-proposal log Z = {log_Z_gt_2d_learned:.4f}")


def take2_2d_single_stage(x_0, T, K, M, obs_model, twist_fn):
    traj = np.zeros((K, T + 1, 2))
    traj[:, 0] = x_0
    ess_per_step = []
    fkc_ess_all = []

    log_psi_prev = np.array([float(twist_fn(x_0.reshape(1, 2), 0))] * K)
    log_Z = log_psi_prev[0]

    for t in range(1, T + 1):
        x_props, log_fkc = composition_sde_2d(traj[:, t - 1], M)
        fkc_ess_list = [compute_ess(row) for row in log_fkc]
        log_G = obs_model.log_G(x_props.reshape(-1, 2), t).reshape(K, M)
        if t < T:
            log_twist = np.asarray(twist_fn(x_props.reshape(-1, 2), t)).reshape(K, M)
        else:
            log_twist = np.zeros((K, M))
        log_v = log_fkc + log_G + log_twist
        j_sel, log_norm = sample_row_indices(log_v, n_draws=1)
        traj[:, t] = x_props[np.arange(K), j_sel]
        log_weights = log_norm - np.log(M) - log_psi_prev
        log_psi_prev = np.array([float(twist_fn(traj[k, t:t + 1], t)) for k in range(K)]) if t < T else np.zeros(K)
        log_Z += logsumexp(log_weights) - np.log(K)
        ess_per_step.append(compute_ess(log_weights))
        fkc_ess_all.append(np.mean(fkc_ess_list))
        indices = systematic_resample_local(log_weights)
        traj = traj[indices]
        if t < T:
            log_psi_prev = log_psi_prev[indices]

    return {"log_Z": log_Z, "ess_per_step": ess_per_step, "fkc_ess": fkc_ess_all, "trajectories": traj}


def take2_2d_two_stage(x_0, T, K, M, obs_model, twist_fn):
    traj = np.zeros((K, T + 1, 2))
    traj[:, 0] = x_0
    ess_per_step = []
    fkc_ess_all = []

    log_psi_prev = np.array([float(twist_fn(x_0.reshape(1, 2), 0))] * K)
    log_Z = log_psi_prev[0]

    for t in range(1, T + 1):
        x_props, log_fkc = composition_sde_2d(traj[:, t - 1], M)
        fkc_ess_list = [compute_ess(row) for row in log_fkc]
        resamp_idx, _ = sample_row_indices(log_fkc, n_draws=M)
        x_resampled = np.take_along_axis(x_props, resamp_idx[..., None], axis=1)
        log_G = obs_model.log_G(x_resampled.reshape(-1, 2), t).reshape(K, M)
        if t < T:
            log_twist = np.asarray(twist_fn(x_resampled.reshape(-1, 2), t)).reshape(K, M)
        else:
            log_twist = np.zeros((K, M))
        log_v = log_G + log_twist
        j_sel, log_norm = sample_row_indices(log_v, n_draws=1)
        traj[:, t] = x_resampled[np.arange(K), j_sel]
        log_weights = log_norm - np.log(M) - log_psi_prev
        log_psi_prev = np.array([float(twist_fn(traj[k, t:t + 1], t)) for k in range(K)]) if t < T else np.zeros(K)
        log_Z += logsumexp(log_weights) - np.log(K)
        ess_per_step.append(compute_ess(log_weights))
        fkc_ess_all.append(np.mean(fkc_ess_list))
        indices = systematic_resample_local(log_weights)
        traj = traj[indices]
        if t < T:
            log_psi_prev = log_psi_prev[indices]

    return {"log_Z": log_Z, "ess_per_step": ess_per_step, "fkc_ess": fkc_ess_all, "trajectories": traj}


print("\nRunning single-stage Take 2 (2D)...")
ss_2d = {"log_Z": [], "ess": [], "fkc_ess": []}
for trial in range(n_trials_2d):
    np.random.seed(trial * 41 + 99)
    res = take2_2d_single_stage(gt_traj[0], T_2d, K_2d, M_2d, obs_up, twist_2d)
    ss_2d["log_Z"].append(res["log_Z"])
    ss_2d["ess"].append(np.mean(res["ess_per_step"]))
    ss_2d["fkc_ess"].append(np.mean(res["fkc_ess"]))
    print(f"  Trial {trial + 1}: log Z = {res['log_Z']:.4f}")

print("\nRunning two-stage Take 2 (2D)...")
ts_2d = {"log_Z": [], "ess": [], "fkc_ess": []}
for trial in range(n_trials_2d):
    np.random.seed(trial * 41 + 99)
    res = take2_2d_two_stage(gt_traj[0], T_2d, K_2d, M_2d, obs_up, twist_2d)
    ts_2d["log_Z"].append(res["log_Z"])
    ts_2d["ess"].append(np.mean(res["ess_per_step"]))
    ts_2d["fkc_ess"].append(np.mean(res["fkc_ess"]))
    print(f"  Trial {trial + 1}: log Z = {res['log_Z']:.4f}")


def compute_mse(traj_samples, ground_truth, missing_times):
    mean_traj = np.mean(traj_samples, axis=0)
    errs = [np.sum((mean_traj[t] - ground_truth[t]) ** 2) for t in missing_times]
    return np.mean(errs)


np.random.seed(999)
res_ss = take2_2d_single_stage(gt_traj[0], T_2d, K_2d, M_2d, obs_up, twist_2d)
np.random.seed(999)
res_ts = take2_2d_two_stage(gt_traj[0], T_2d, K_2d, M_2d, obs_up, twist_2d)
mse_ss = compute_mse(res_ss["trajectories"], gt_traj, missing_times)
mse_ts = compute_mse(res_ts["trajectories"], gt_traj, missing_times)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(
    f"{SYSTEM_NAME_HD} GTs: true-dynamics = {log_Z_gt_hd:.4f}, "
    f"learned-proposal = {log_Z_gt_hd_learned:.4f}"
)
print(
    f"2D GTs: true-dynamics = {log_Z_gt_2d:.4f}, "
    f"learned-proposal = {log_Z_gt_2d_learned:.4f}"
)
print(f"2D single-stage: log Z = {np.mean(ss_2d['log_Z']):.4f} +/- {np.std(ss_2d['log_Z']):.4f}, ESS = {np.mean(ss_2d['ess']):.1f}, FKC ESS = {np.mean(ss_2d['fkc_ess']):.2f}")
print(f"2D two-stage:    log Z = {np.mean(ts_2d['log_Z']):.4f} +/- {np.std(ts_2d['log_Z']):.4f}, ESS = {np.mean(ts_2d['ess']):.1f}, FKC ESS = {np.mean(ts_2d['fkc_ess']):.2f}")
print(f"MSE (missing frames): single-stage = {mse_ss:.6f}, two-stage = {mse_ts:.6f}")

part_a_highd = {
    "parameters": {
        "system_name": SYSTEM_NAME_HD,
        "dim": D_hd,
        "dt": dt_hd,
        "T": T_hd,
        "x_0": x_0_hd.tolist(),
        "x_target": x_target_hd.tolist(),
        "lam": lam_hd,
        "K": K_hd,
        "S_MAX": S_MAX_hd,
        "N_DIFF": N_DIFF_hd,
        "n_trials": n_trials_hd,
        "coupling": COUPLING_HD,
        "score_config": PART_A_SCORE_CONFIG,
    },
    "ground_truth": {
        "true_dynamics_log_Z": float(log_Z_gt_hd),
        "learned_proposal_log_Z": float(log_Z_gt_hd_learned),
        "proposal_inner_M": GT_PROPOSAL_M_HD,
        "proposal_n_gt": n_gt_hd_learned,
    },
    "ground_truth_log_Z": float(log_Z_gt_hd_learned),
    "true_dynamics_log_Z": float(log_Z_gt_hd),
    "learned_proposal_log_Z": float(log_Z_gt_hd_learned),
    "results_by_M": part_a_results,
}

output = {
    "experiment": "run_exp4",
    "output_path": OUTPUT_PATH,
    "part_a_highd": part_a_highd,
    "part_a_1d": part_a_highd,
    "part_b_2d": {
        "parameters": {
            "T": T_2d,
            "K": K_2d,
            "M": M_2d,
            "S_MAX": S_MAX_2d,
            "N_DIFF": N_DIFF_2d,
            "n_trials": n_trials_2d,
            "score_config": PART_B_SCORE_CONFIG,
        },
        "ground_truth": {
            "true_dynamics_log_Z": float(log_Z_gt_2d),
            "learned_proposal_log_Z": float(log_Z_gt_2d_learned),
            "proposal_inner_M": GT_PROPOSAL_M_2D,
            "proposal_n_gt": n_gt_2d_learned,
        },
        "ground_truth_log_Z": float(log_Z_gt_2d_learned),
        "true_dynamics_log_Z": float(log_Z_gt_2d),
        "learned_proposal_log_Z": float(log_Z_gt_2d_learned),
        "single_stage": {
            "log_Z": ss_2d["log_Z"],
            "ess": ss_2d["ess"],
            "fkc_ess": ss_2d["fkc_ess"],
            "summary": {
                "log_Z": mean_std(ss_2d["log_Z"]),
                "ess": mean_std(ss_2d["ess"]),
                "fkc_ess": mean_std(ss_2d["fkc_ess"]),
                "mse_missing_frames": float(mse_ss),
            },
        },
        "two_stage": {
            "log_Z": ts_2d["log_Z"],
            "ess": ts_2d["ess"],
            "fkc_ess": ts_2d["fkc_ess"],
            "summary": {
                "log_Z": mean_std(ts_2d["log_Z"]),
                "ess": mean_std(ts_2d["ess"]),
                "fkc_ess": mean_std(ts_2d["fkc_ess"]),
                "mse_missing_frames": float(mse_ts),
            },
        },
    },
}
save_json(output, OUTPUT_PATH)
print(f"\nSaved JSON results to {OUTPUT_PATH}")
