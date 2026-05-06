"""
Canonical Experiment 3 runner: conditional generation on the 2D coupled double-well.

This consolidates the previous "run_exp3.py" and "run_exp3_improved.py" scripts into
one entry point. It reports:

- learned-score proposal variants (the current implementation setup)
- exact-drift proposal variants where the exact Langevin transition is available
- three Take 3 twist objectives: MC, TD, and Lemma-3 KL
- pooled-particle and per-trial reconstruction MSE

The learned-score backend still reflects the reverse-SDE approximation used in the
paper draft, so the JSON records both the true-dynamics and learned-proposal
normalizing-constant references to make the comparison explicit.
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
    compute_ess,
    build_future_reward_targets,
    make_future_trajectory_reward,
    sample_row_indices,
    mean_std,
    save_json,
)
from updated_code.fixed_twist import (
    train_positive_twist_mc,
    train_positive_twist_td,
    train_positive_twist_kl,
    PositiveNNTwist,
)

np.random.seed(42)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "results", "run_exp3.json")


def U(x):
    x = np.asarray(x, dtype=np.float64)
    xx, yy = x[..., 0], x[..., 1]
    return (xx ** 2 - 1) ** 2 + yy ** 2 + 0.5 * xx * yy


def grad_U(x):
    x = np.asarray(x, dtype=np.float64)
    xx, yy = x[..., 0], x[..., 1]
    dU_dx = 4 * xx * (xx ** 2 - 1) + 0.5 * yy
    dU_dy = 2 * yy + 0.5 * xx
    return np.stack([dU_dx, dU_dy], axis=-1)


WELL_R = np.array([1.0, -0.25])
WELL_L = np.array([-1.0, 0.25])
dt_dynamics = 0.02


def langevin_step(x, dt=None, grad_clip=10.0):
    if dt is None:
        dt = dt_dynamics
    x = np.asarray(x, dtype=np.float64)
    scalar = x.ndim == 1
    if scalar:
        x = x[None, :]
    g = np.clip(grad_U(x), -grad_clip, grad_clip)
    noise = np.random.randn(*x.shape) * np.sqrt(2.0 * dt)
    x_new = x - dt * g + noise
    return x_new[0] if scalar else x_new


def sample_trajectory(x_0, T, dt=None):
    if dt is None:
        dt = dt_dynamics
    traj = np.zeros((T + 1, 2))
    traj[0] = np.asarray(x_0)
    for t in range(T):
        traj[t + 1] = langevin_step(traj[t], dt)
    return traj


class UpsamplingObs:
    def __init__(self, ground_truth_traj, obs_interval=2, lam=8.0):
        self.gt = ground_truth_traj
        self.T = len(ground_truth_traj) - 1
        self.interval = obs_interval
        self.lam = lam
        self.obs_times = set(range(0, self.T + 1, obs_interval))

    def log_G(self, x, t):
        x = np.atleast_2d(x)
        if t in self.obs_times:
            diff = x - self.gt[t]
            return -self.lam * np.sum(diff ** 2, axis=-1)
        return np.zeros(x.shape[0])


class InpaintingObs:
    def __init__(self, ground_truth_traj, lam=15.0):
        self.gt = ground_truth_traj
        self.T = len(ground_truth_traj) - 1
        self.lam = lam

    def log_G(self, x, t):
        x = np.atleast_2d(x)
        diff_x = x[:, 0] - self.gt[t, 0]
        return -self.lam * diff_x ** 2


def compute_mse(traj_samples, ground_truth, missing_times=None, coord_idx=None):
    mean_traj = np.mean(traj_samples, axis=0)
    if missing_times is not None:
        errs = [np.sum((mean_traj[t] - ground_truth[t]) ** 2) for t in missing_times]
        return np.mean(errs)
    if coord_idx is not None:
        return np.mean((mean_traj[:, coord_idx] - ground_truth[:, coord_idx]) ** 2)
    return np.mean(np.sum((mean_traj - ground_truth) ** 2, axis=-1))


def summarize_mse(traj_trials, ground_truth, **mse_kwargs):
    traj_trials = [np.asarray(tr) for tr in traj_trials]
    trial_mse = [float(compute_mse(tr, ground_truth, **mse_kwargs)) for tr in traj_trials]
    pooled = float(compute_mse(np.concatenate(traj_trials, axis=0), ground_truth, **mse_kwargs))
    return {
        "per_trial": trial_mse,
        "trial_mean": mean_std(trial_mse),
        "pooled_particles": pooled,
    }


def exact_transition_sample(x_cond, n_samples=1):
    x_cond = np.asarray(x_cond, dtype=np.float64)
    if x_cond.ndim == 1:
        x_batch = np.tile(x_cond, (n_samples, 1))
        return np.asarray(langevin_step(x_batch, dt_dynamics), dtype=np.float64).reshape(n_samples, 2)
    else:
        x_batch = np.repeat(x_cond.reshape(-1, 2), n_samples, axis=0)
        return np.asarray(langevin_step(x_batch, dt_dynamics), dtype=np.float64).reshape(len(x_cond), n_samples, 2)


dim = 2
T = 20
K = 50
M = 8
S_MAX = 1.5
N_DIFF_STEPS = 20
n_trials = 5
LEARNED_GT_SAMPLES = 5000
TRUE_GT_SAMPLES = 5000
TWIST_CONFIG = {
    "n_ref": 20000,
    "hidden_dim": 256,
    "n_layers": 4,
    "n_epochs": 5000,
    "loss_space": "log",
}
TAKE1_EXACT_ROLLOUTS = 4


def conditional_score_eval_2d(o_score, x_batch, s, x_cond_batch):
    x_arr = np.asarray(x_batch, dtype=np.float64).reshape(-1, 2)
    x_cond_arr = np.asarray(x_cond_batch, dtype=np.float64)
    if x_cond_arr.ndim == 1:
        x_cond_arr = x_cond_arr.reshape(1, 2)
    else:
        x_cond_arr = x_cond_arr.reshape(-1, 2)
    if len(x_cond_arr) == 1 and len(x_arr) > 1:
        x_cond_arr = np.tile(x_cond_arr, (len(x_arr), 1))
    elif len(x_cond_arr) != len(x_arr):
        raise ValueError("x_batch and x_cond_batch must have matching batch sizes")

    with torch.no_grad():
        inp = torch.tensor(
            np.concatenate([x_arr, x_cond_arr], axis=-1),
            dtype=torch.float32,
            device=o_score.device,
        )
        st = torch.full((len(x_arr),), float(s), dtype=torch.float32, device=o_score.device)
        out = o_score.model(inp, st).cpu().numpy()
    return np.clip(out, -o_score.clip, o_score.clip)


ALPHA_TWEEDIE = np.exp(-S_MAX / 2.0)
SIGMA2_TWEEDIE = 1.0 - np.exp(-S_MAX)


def tweedie_twist_batch_fast(x_proposals, t, obs_model, o_score, n_rollouts=1, clip=10.0):
    x_proposals = np.asarray(x_proposals, dtype=np.float64).reshape(-1, 2)
    if t >= T:
        return np.zeros(len(x_proposals), dtype=np.float64)

    log_rewards = np.zeros((n_rollouts, len(x_proposals)), dtype=np.float64)
    for r in range(n_rollouts):
        x_hat = x_proposals.copy()
        total = np.zeros(len(x_proposals), dtype=np.float64)
        for t_future in range(t + 1, T + 1):
            z = np.random.randn(len(x_hat), 2)
            score = conditional_score_eval_2d(o_score, z, S_MAX, x_hat)
            x_hat = np.clip((z + SIGMA2_TWEEDIE * score) / ALPHA_TWEEDIE, -clip, clip)
            total += obs_model.log_G(x_hat, t_future)
        log_rewards[r] = total
    return logsumexp(log_rewards, axis=0) - np.log(n_rollouts)


def sample_chain_trajectories(x_0, T, step_sampler, n_samples):
    traj = np.zeros((n_samples, T + 1, 2), dtype=np.float64)
    traj[:, 0] = np.asarray(x_0, dtype=np.float64)
    x = np.tile(np.asarray(x_0, dtype=np.float64), (n_samples, 1))
    for t in range(T):
        x = np.asarray(step_sampler(x, n_samples=1)).reshape(n_samples, 2)
        traj[:, t + 1] = x
    return traj


def make_reverse_sde_sampler(o_score):
    def sampler(x_cond, n_samples=1, S=S_MAX, n_steps=N_DIFF_STEPS):
        x_cond_arr = np.asarray(x_cond, dtype=np.float64)
        if x_cond_arr.ndim == 1:
            x_cond_arr = x_cond_arr.reshape(1, 2)
        else:
            x_cond_arr = x_cond_arr.reshape(-1, 2)
        n_cond = len(x_cond_arr)
        x_cond_rep = np.repeat(x_cond_arr, n_samples, axis=0)
        ds = S / n_steps
        x = np.random.randn(len(x_cond_rep), 2)
        for step in range(n_steps):
            s = max(S - step * ds, 1e-6)
            score = conditional_score_eval_2d(o_score, x, s, x_cond_rep)
            x = x + (0.5 * x + score) * ds + np.random.randn(len(x_cond_rep), 2) * np.sqrt(ds)
        if n_cond == 1:
            return x.reshape(n_samples, 2)
        return x.reshape(n_cond, n_samples, 2)
    return sampler


def make_composition_sampler(j_score, o_score):
    def sampler(x_cond, n_samples=1, S=S_MAX, n_steps=N_DIFF_STEPS):
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

            j_s = np.clip(np.atleast_2d(j_score.score(x, s)), -clip, clip)
            o_s = np.clip(conditional_score_eval_2d(o_score, x, s, x_cond_rep), -clip, clip)
            j_0 = np.clip(np.atleast_2d(j_score.boltzmann_score(x)), -clip, clip)

            j_s = np.nan_to_num(j_s, nan=0.0)
            o_s = np.nan_to_num(o_s, nan=0.0)
            j_0 = np.nan_to_num(j_0, nan=0.0)

            drift = np.clip(o_s - j_s + 0.5 * j_0 + 0.5 * x, -clip, clip)
            fkc_inc = (
                0.5 * dim
                + np.sum(j_s * (j_s - o_s), axis=-1)
                + 0.5 * np.sum(j_0 * (x + o_s - j_s), axis=-1)
            )
            fkc_inc = np.clip(fkc_inc, -20 / S, 20 / S)
            log_w += fkc_inc * ds
            x = x + drift * ds + np.random.randn(len(x_cond_rep), 2) * np.sqrt(ds)

        log_w = np.nan_to_num(log_w, nan=0.0, posinf=20.0, neginf=-20.0)
        if n_cond == 1:
            return x.reshape(n_samples, 2), log_w.reshape(n_samples)
        return x.reshape(n_cond, n_samples, 2), log_w.reshape(n_cond, n_samples)
    return sampler


def sample_chain_trajectory(x_0, T, step_sampler):
    traj = np.zeros((T + 1, 2))
    traj[0] = np.asarray(x_0)
    for t in range(T):
        traj[t + 1] = step_sampler(traj[t], n_samples=1)[0]
    return traj


def estimate_log_Z_from_chain(x_0, T, obs_model, step_sampler, n_samples):
    traj = sample_chain_trajectories(x_0, T, step_sampler, n_samples)
    log_rewards = np.zeros(n_samples, dtype=np.float64)
    for t in range(T + 1):
        log_rewards += obs_model.log_G(traj[:, t], t)
    return float(logsumexp(log_rewards) - np.log(n_samples))


def train_twist_suite(ref_trajs, future_targets, obs_model):
    def final_loss(losses):
        return float(losses[-1]) if losses else None

    print(f"  Collecting twist suite with config: {TWIST_CONFIG}")
    t0 = time.time()
    mc_model, mc_losses = train_positive_twist_mc(
        ref_trajs,
        future_targets,
        dim=2,
        T=T,
        n_epochs=TWIST_CONFIG["n_epochs"],
        hidden_dim=TWIST_CONFIG["hidden_dim"],
        n_layers=TWIST_CONFIG["n_layers"],
        device=DEVICE,
        loss_space=TWIST_CONFIG["loss_space"],
    )
    mc_twist = PositiveNNTwist(mc_model, dim=2, device=DEVICE)
    print(f"    MC ready in {time.time() - t0:.1f}s")

    t0 = time.time()
    td_model, td_losses = train_positive_twist_td(
        ref_trajs,
        dim=2,
        T=T,
        log_G_fn=obs_model.log_G,
        terminal_targets=np.ones(len(ref_trajs)),
        n_epochs=TWIST_CONFIG["n_epochs"],
        hidden_dim=TWIST_CONFIG["hidden_dim"],
        n_layers=TWIST_CONFIG["n_layers"],
        device=DEVICE,
        loss_space=TWIST_CONFIG["loss_space"],
    )
    td_twist = PositiveNNTwist(td_model, dim=2, device=DEVICE)
    print(f"    TD ready in {time.time() - t0:.1f}s")

    t0 = time.time()
    kl_model, kl_losses = train_positive_twist_kl(
        ref_trajs,
        future_targets,
        dim=2,
        T=T,
        n_epochs=TWIST_CONFIG["n_epochs"],
        hidden_dim=TWIST_CONFIG["hidden_dim"],
        n_layers=TWIST_CONFIG["n_layers"],
        device=DEVICE,
    )
    kl_twist = PositiveNNTwist(kl_model, dim=2, device=DEVICE)
    print(f"    KL ready in {time.time() - t0:.1f}s")

    return {
        "Take3 (MC)": {"twist": mc_twist, "final_loss": final_loss(mc_losses), "objective": "MC"},
        "Take3 (TD)": {"twist": td_twist, "final_loss": final_loss(td_losses), "objective": "TD"},
        "Take3 (KL)": {"twist": kl_twist, "final_loss": final_loss(kl_losses), "objective": "KL"},
    }


def run_bootstrap(label, x0, obs_model, step_sampler):
    print(f"\n  {label}...")
    data = {"log_Z": [], "ess": [], "traj_trials": []}
    for trial in range(n_trials):
        np.random.seed(trial * 41)
        traj = np.zeros((K, T + 1, 2))
        traj[:, 0] = x0
        log_Z = 0.0
        ess_list = []

        for t in range(1, T + 1):
            traj[:, t] = np.asarray(step_sampler(traj[:, t - 1], n_samples=1)).reshape(K, 2)
            log_w = obs_model.log_G(traj[:, t], t)
            log_Z += logsumexp(log_w) - np.log(K)
            ess_list.append(compute_ess(log_w))
            indices = systematic_resample(log_w)
            traj = traj[indices]

        data["log_Z"].append(log_Z)
        data["ess"].append(np.mean(ess_list))
        data["traj_trials"].append(traj.copy())
    return data


def run_take3(label, x0, obs_model, step_sampler, twist_model):
    print(f"\n  {label}...")
    data = {"log_Z": [], "ess": [], "traj_trials": []}
    for trial in range(n_trials):
        np.random.seed(trial * 41)
        traj = np.zeros((K, T + 1, 2))
        traj[:, 0] = x0
        ess_list = []

        log_psi_cached = np.full(K, float(twist_model(x0.reshape(1, 2), 0)))
        log_Z = log_psi_cached[0]

        for t in range(1, T + 1):
            proposals = np.asarray(step_sampler(traj[:, t - 1], n_samples=M)).reshape(K, M, 2)
            log_G = obs_model.log_G(proposals.reshape(-1, 2), t).reshape(K, M)
            if t < T:
                log_twist = np.asarray(twist_model(proposals.reshape(-1, 2), t)).reshape(K, M)
            else:
                log_twist = np.zeros((K, M))
            log_v = log_G + log_twist
            j_sel, log_norm = sample_row_indices(log_v, n_draws=1)
            traj[:, t] = proposals[np.arange(K), j_sel]
            log_weights = log_norm - np.log(M) - log_psi_cached
            log_psi_next = log_twist[np.arange(K), j_sel] if t < T else np.zeros(K)
            log_Z += logsumexp(log_weights) - np.log(K)
            ess_list.append(compute_ess(log_weights))
            indices = systematic_resample(log_weights)
            traj = traj[indices]
            log_psi_cached = log_psi_next[indices]

        data["log_Z"].append(log_Z)
        data["ess"].append(np.mean(ess_list))
        data["traj_trials"].append(traj.copy())
    return data


def run_take1_learned(label, x0, obs_model, step_sampler, o_score):
    print(f"\n  {label}...")
    data = {"log_Z": [], "ess": [], "traj_trials": []}
    for trial in range(n_trials):
        np.random.seed(trial * 41 + 7)
        traj = np.zeros((K, T + 1, 2))
        traj[:, 0] = x0
        ess_list = []

        log_psi_init = float(tweedie_twist_batch_fast(x0.reshape(1, 2), 0, obs_model, o_score, n_rollouts=1)[0])
        log_Z = log_psi_init
        log_psi_cached = np.full(K, log_psi_init)

        for t in range(1, T + 1):
            proposals = np.asarray(step_sampler(traj[:, t - 1], n_samples=M)).reshape(K, M, 2)
            log_G = obs_model.log_G(proposals.reshape(-1, 2), t).reshape(K, M)
            if t < T:
                log_twist = tweedie_twist_batch_fast(proposals.reshape(-1, 2), t, obs_model, o_score, n_rollouts=1).reshape(K, M)
            else:
                log_twist = np.zeros((K, M))
            log_v = log_G + log_twist
            j_sel, log_norm = sample_row_indices(log_v, n_draws=1)
            traj[:, t] = proposals[np.arange(K), j_sel]
            log_weights = log_norm - np.log(M) - log_psi_cached
            log_psi_next = log_twist[np.arange(K), j_sel] if t < T else np.zeros(K)
            log_Z += logsumexp(log_weights) - np.log(K)
            ess_list.append(compute_ess(log_weights))
            indices = systematic_resample(log_weights)
            traj = traj[indices]
            log_psi_cached = log_psi_next[indices]

        data["log_Z"].append(log_Z)
        data["ess"].append(np.mean(ess_list))
        data["traj_trials"].append(traj.copy())
    return data


def run_take2_learned(label, x0, obs_model, composition_sampler, o_score):
    print(f"\n  {label}...")
    data = {"log_Z": [], "ess": [], "traj_trials": []}
    for trial in range(n_trials):
        np.random.seed(trial * 41 + 99)
        traj = np.zeros((K, T + 1, 2))
        traj[:, 0] = x0
        ess_list = []

        log_psi_init = float(tweedie_twist_batch_fast(x0.reshape(1, 2), 0, obs_model, o_score, n_rollouts=1)[0])
        log_Z = log_psi_init
        log_psi_cached = np.full(K, log_psi_init)

        for t in range(1, T + 1):
            x_comp, log_fkc = composition_sampler(traj[:, t - 1], n_samples=M)
            fkc_indices, _ = sample_row_indices(log_fkc, n_draws=M)
            proposals = np.take_along_axis(x_comp, fkc_indices[..., None], axis=1)
            log_G = obs_model.log_G(proposals.reshape(-1, 2), t).reshape(K, M)
            if t < T:
                log_twist = tweedie_twist_batch_fast(proposals.reshape(-1, 2), t, obs_model, o_score, n_rollouts=1).reshape(K, M)
            else:
                log_twist = np.zeros((K, M))
            log_v = log_G + log_twist
            j_sel, log_norm = sample_row_indices(log_v, n_draws=1)
            traj[:, t] = proposals[np.arange(K), j_sel]
            log_weights = log_norm - np.log(M) - log_psi_cached
            log_psi_next = log_twist[np.arange(K), j_sel] if t < T else np.zeros(K)
            log_Z += logsumexp(log_weights) - np.log(K)
            ess_list.append(compute_ess(log_weights))
            indices = systematic_resample(log_weights)
            traj = traj[indices]
            log_psi_cached = log_psi_next[indices]

        data["log_Z"].append(log_Z)
        data["ess"].append(np.mean(ess_list))
        data["traj_trials"].append(traj.copy())
    return data


def build_backend_summary(results, ground_truth_log_Z, mse_kwargs, gt_traj):
    summary = {}
    for name, data in results.items():
        mse_summary = summarize_mse(data["traj_trials"], gt_traj, **mse_kwargs)
        summary[name] = {
            "log_Z": mean_std(data["log_Z"]),
            "ess": mean_std(data["ess"]),
            "mse": mse_summary,
        }
    return {
        "methods": results,
        "summary": summary,
        "proposal_ground_truth_log_Z": float(ground_truth_log_Z),
    }


def print_backend_summary(backend_name, ground_truth_log_Z, results):
    print(f"\n  Summary [{backend_name}]")
    print(f"  {'Method':<25} {'log Z (mean +/- std)':<28} {'ESS':<12}")
    print(f"  {'Proposal GT':<25} {ground_truth_log_Z:<28.4f} {'--':<12}")
    for name, data in results.items():
        logZ_str = f"{np.mean(data['log_Z']):.4f} +/- {np.std(data['log_Z']):.4f}"
        ess_str = f"{np.mean(data['ess']):.2f}"
        print(f"  {name:<25} {logZ_str:<28} {ess_str:<12}")


def run_task(task_name, obs_model, mse_kwargs, x0, twist_suite, learned_step_sampler, composition_sampler, o_score):
    print(f"\n{'=' * 70}")
    print(f"  {task_name}")
    print(f"{'=' * 70}")

    print("\n  Ground-truth references...")
    true_log_Z = estimate_log_Z_from_chain(x0, T, obs_model, exact_transition_sample, TRUE_GT_SAMPLES)
    learned_log_Z = estimate_log_Z_from_chain(x0, T, obs_model, learned_step_sampler, LEARNED_GT_SAMPLES)
    print(f"  true-dynamics log Z      = {true_log_Z:.4f}")
    print(f"  learned-proposal log Z  = {learned_log_Z:.4f}")

    learned_results = {}
    learned_results["Bootstrap"] = run_bootstrap("Bootstrap (learned proposal)", x0, obs_model, learned_step_sampler)
    learned_results["Take1 (Tweedie)"] = run_take1_learned(
        "Take1 (Tweedie, learned proposal)",
        x0,
        obs_model,
        learned_step_sampler,
        o_score,
    )
    for method_name, info in twist_suite.items():
        learned_results[method_name] = run_take3(
            f"{method_name} (learned proposal)",
            x0,
            obs_model,
            learned_step_sampler,
            info["twist"],
        )
    learned_results["Take2 (two-stage)"] = run_take2_learned(
        "Take2 (two-stage, learned proposal)",
        x0,
        obs_model,
        composition_sampler,
        o_score,
    )
    print_backend_summary("learned_scores", learned_log_Z, learned_results)

    exact_results = {}
    exact_results["Bootstrap"] = run_bootstrap("Bootstrap (exact drift)", x0, obs_model, exact_transition_sample)
    exact_results["Take1 (Tweedie)"] = run_take1_learned(
        "Take1 (Tweedie, exact drift)",
        x0,
        obs_model,
        exact_transition_sample,
        o_score,
    )
    for method_name, info in twist_suite.items():
        exact_results[method_name] = run_take3(
            f"{method_name} (exact drift)",
            x0,
            obs_model,
            exact_transition_sample,
            info["twist"],
        )
    print_backend_summary("exact_drift", true_log_Z, exact_results)

    return {
        "task_name": task_name,
        "ground_truth": {
            "true_dynamics_log_Z": float(true_log_Z),
            "learned_proposal_log_Z": float(learned_log_Z),
        },
        "backends": {
            "learned_scores": build_backend_summary(learned_results, learned_log_Z, mse_kwargs, obs_model.gt),
            "exact_drift": build_backend_summary(exact_results, true_log_Z, mse_kwargs, obs_model.gt),
        },
    }


def main(output_path=OUTPUT_PATH):
    print("=" * 70)
    print("UPDATED EXPERIMENT 3: Conditional Generation")
    print("=" * 70)
    print(f"dt={dt_dynamics}, T={T}, K={K}, M={M}")
    print(f"Twist config: n_ref={TWIST_CONFIG['n_ref']}, hidden={TWIST_CONFIG['hidden_dim']}, "
          f"layers={TWIST_CONFIG['n_layers']}, epochs={TWIST_CONFIG['n_epochs']}\n")

    print("--- Phase 1: Collecting training data ---\n")
    print("Collecting equilibrium samples...")
    eq_samples = []
    for start in [WELL_R, WELL_L]:
        x = start.copy() + 0.02 * np.random.randn(2)
        for _ in range(3000):
            x = langevin_step(x, 0.005)
        chain = np.zeros((2000, 2))
        for i in range(2000):
            for _ in range(15):
                x = langevin_step(x, 0.005)
            chain[i] = x
        eq_samples.append(chain)
    eq_samples = np.concatenate(eq_samples, axis=0)
    np.random.shuffle(eq_samples)
    print(f"  Total: {len(eq_samples)} samples")

    print("Collecting 5000 transition pairs...")
    n_walkers = 200
    walkers = np.zeros((n_walkers, 2))
    for i in range(n_walkers):
        walkers[i] = [WELL_R, WELL_L][i % 2].copy() + 0.05 * np.random.randn(2)
    for _ in range(5000):
        walkers = langevin_step(walkers, 0.005)

    n_pairs = 5000
    pairs_per_walker = (n_pairs + n_walkers - 1) // n_walkers
    xt_list, xtp1_list = [], []
    for _ in range(pairs_per_walker):
        for _ in range(20):
            walkers = langevin_step(walkers, 0.005)
        xt_list.append(walkers.copy())
        xtp1_list.append(langevin_step(walkers, dt_dynamics))
    xt_data = np.concatenate(xt_list, axis=0)[:n_pairs]
    xtp1_data = np.concatenate(xtp1_list, axis=0)[:n_pairs]
    print("  Done.")

    print("\nGenerating ground truth trajectory from right well...")
    gt_traj = sample_trajectory(WELL_R, T)
    print(f"  x_0 = ({gt_traj[0,0]:.3f}, {gt_traj[0,1]:.3f})")
    print(f"  x_T = ({gt_traj[-1,0]:.3f}, {gt_traj[-1,1]:.3f})")

    print("\n--- Phase 2: Training NN scores ---\n")
    print("Training marginal score j_s (NN, DSM)...")
    t0 = time.time()
    j_model, _ = train_marginal_score(eq_samples, dim=2, n_epochs=3000, hidden_dim=128, n_layers=4, device=DEVICE)
    j_score = NNMarginalScore(j_model, dim=2, device=DEVICE)
    print(f"  Time: {time.time() - t0:.1f}s\n")

    print("Training conditional score o_s (NN, DSM)...")
    t0 = time.time()
    o_model, _ = train_conditional_score(xt_data, xtp1_data, dim=2, n_epochs=3000, hidden_dim=128, n_layers=4, device=DEVICE)
    o_score = NNConditionalScore(o_model, dim=2, device=DEVICE)
    print(f"  Time: {time.time() - t0:.1f}s\n")

    learned_step_sampler = make_reverse_sde_sampler(o_score)
    composition_sampler = make_composition_sampler(j_score, o_score)

    print("--- Phase 3: Collecting reference trajectories for twist learning ---\n")
    ref_trajs = sample_chain_trajectories(gt_traj[0], T, exact_transition_sample, TWIST_CONFIG["n_ref"])

    obs_up = UpsamplingObs(gt_traj, obs_interval=2, lam=8.0)
    future_targets_up = build_future_reward_targets(ref_trajs, T, obs_up.log_G)
    print(f"  Upsampling target mean at t=0: {future_targets_up[:, 0].mean():.6e}")
    up_twists = train_twist_suite(ref_trajs, future_targets_up, obs_up)

    obs_inp = InpaintingObs(gt_traj, lam=15.0)
    future_targets_inp = build_future_reward_targets(ref_trajs, T, obs_inp.log_G)
    print(f"  Inpainting target mean at t=0: {future_targets_inp[:, 0].mean():.6e}")
    inp_twists = train_twist_suite(ref_trajs, future_targets_inp, obs_inp)

    print("\n--- Phase 4: Running conditional-generation experiments ---")
    missing_times = [t for t in range(T + 1) if t not in obs_up.obs_times]
    upsampling = run_task(
        "UPSAMPLING",
        obs_up,
        dict(missing_times=missing_times),
        gt_traj[0],
        up_twists,
        learned_step_sampler,
        composition_sampler,
        o_score,
    )
    inpainting = run_task(
        "INPAINTING",
        obs_inp,
        dict(coord_idx=1),
        gt_traj[0],
        inp_twists,
        learned_step_sampler,
        composition_sampler,
        o_score,
    )

    output = {
        "experiment": "run_exp3",
        "output_path": output_path,
        "parameters": {
            "dim": dim,
            "T": T,
            "K": K,
            "M": M,
            "S_MAX": S_MAX,
            "N_DIFF_STEPS": N_DIFF_STEPS,
            "n_trials": n_trials,
            "dt_dynamics": dt_dynamics,
            "true_gt_samples": TRUE_GT_SAMPLES,
            "learned_gt_samples": LEARNED_GT_SAMPLES,
            "take1_exact_rollouts": TAKE1_EXACT_ROLLOUTS,
        },
        "twist_training": {
            "config": TWIST_CONFIG,
            "upsampling": {
                name: {"objective": info["objective"], "final_loss": info["final_loss"]}
                for name, info in up_twists.items()
            },
            "inpainting": {
                name: {"objective": info["objective"], "final_loss": info["final_loss"]}
                for name, info in inp_twists.items()
            },
        },
        "ground_truth_trajectory": gt_traj,
        "tasks": {
            "upsampling": upsampling,
            "inpainting": inpainting,
        },
    }
    save_json(output, output_path)
    print(f"\nSaved JSON results to {output_path}")
    print("\n" + "=" * 70)
    print("UPDATED EXPERIMENT 3 COMPLETE")
    print("=" * 70)
    return output


if __name__ == "__main__":
    main()
