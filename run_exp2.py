"""
Updated Experiment 2: 2D Muller-Brown barrier crossing.

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
from updated_code.muller_brown_core import (
    langevin_step,
    sample_trajectory,
    sample_adjacent_pairs,
    BarrierCrossingReward,
    MINIMUM_A,
    MINIMUM_B,
    MINIMUM_C,
    SADDLE_BC,
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

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "results", "run_exp2.json")

dim = 2
dt = 0.01
T = 20
K = 50
M = 12
S_MAX = 2.0
N_DIFF_STEPS = 25
n_trials = 8
n_trials_t2 = 5
MARGINAL_SCORE_CONFIG = {
    "n_eq_per_chain": 3200,
    "hidden_dim": 256,
    "n_layers": 4,
    "n_epochs": 6000,
}
CONDITIONAL_SCORE_CONFIG = {
    "n_transition_pairs": 5000,
    "hidden_dim": 192,
    "n_layers": 3,
    "n_epochs": 3000,
}
TWIST_CONFIG = {
    "n_ref": 20000,
    "hidden_dim": 256,
    "n_layers": 4,
    "n_epochs": 5000,
    "loss_space": "linear",
}
GT_CONFIG = {
    "n_gt": 20000,
}

x_0 = MINIMUM_A.copy()
reward_model = BarrierCrossingReward(
    T=T,
    target=MINIMUM_B,
    saddle=SADDLE_BC,
    lam_endpoint=5.0,
    lam_saddle=3.0,
    T_mid=T // 2,
)

trajectory_reward_fn = make_future_trajectory_reward(reward_model.log_G, T)

print("=" * 70)
print("UPDATED EXPERIMENT 2: 2D Muller-Brown")
print("=" * 70)
print(f"dt={dt}, T={T}, K={K}, M={M}")
print(f"Reverse SDE: S={S_MAX}, n_steps={N_DIFF_STEPS}\n")
print(
    "Marginal score config: "
    f"n_eq_per_chain={MARGINAL_SCORE_CONFIG['n_eq_per_chain']}, "
    f"hidden={MARGINAL_SCORE_CONFIG['hidden_dim']}, "
    f"layers={MARGINAL_SCORE_CONFIG['n_layers']}, "
    f"epochs={MARGINAL_SCORE_CONFIG['n_epochs']}"
)
print(
    "Conditional score config: "
    f"pairs={CONDITIONAL_SCORE_CONFIG['n_transition_pairs']}, "
    f"hidden={CONDITIONAL_SCORE_CONFIG['hidden_dim']}, "
    f"layers={CONDITIONAL_SCORE_CONFIG['n_layers']}, "
    f"epochs={CONDITIONAL_SCORE_CONFIG['n_epochs']}"
)
print(f"Twist config: n_ref={TWIST_CONFIG['n_ref']}, hidden={TWIST_CONFIG['hidden_dim']}, "
      f"layers={TWIST_CONFIG['n_layers']}, epochs={TWIST_CONFIG['n_epochs']}, "
      f"loss={TWIST_CONFIG['loss_space']}")
print(f"GT samples: {GT_CONFIG['n_gt']}\n")


print("--- Phase 1: Collecting training data ---\n")
print(f"Collecting equilibrium samples (3 chains x {MARGINAL_SCORE_CONFIG['n_eq_per_chain']})...")
n_eq_per_chain = MARGINAL_SCORE_CONFIG["n_eq_per_chain"]
eq_samples = []
for start in [MINIMUM_A, MINIMUM_B, MINIMUM_C]:
    x = start.copy() + 0.02 * np.random.randn(2)
    for _ in range(2000):
        x = langevin_step(x, 0.003)
    chain = np.zeros((n_eq_per_chain, 2))
    for i in range(n_eq_per_chain):
        for _ in range(15):
            x = langevin_step(x, 0.003)
        chain[i] = x
    eq_samples.append(chain)
eq_samples = np.concatenate(eq_samples, axis=0)
np.random.shuffle(eq_samples)
print(f"  Total: {len(eq_samples)} samples")

print(f"Collecting {CONDITIONAL_SCORE_CONFIG['n_transition_pairs']} transition pairs...")
xt_data, xtp1_data = sample_adjacent_pairs(CONDITIONAL_SCORE_CONFIG["n_transition_pairs"], dt_traj=dt)
print("  Done.")


print("--- Phase 2: Training NN scores ---\n")
print("Training marginal score j_s (NN, DSM)...")
t0 = time.time()
j_model, _ = train_marginal_score(
    eq_samples,
    dim=2,
    n_epochs=MARGINAL_SCORE_CONFIG["n_epochs"],
    hidden_dim=MARGINAL_SCORE_CONFIG["hidden_dim"],
    n_layers=MARGINAL_SCORE_CONFIG["n_layers"],
    device=DEVICE,
)
j_score = NNMarginalScore(j_model, dim=2, device=DEVICE)
print(f"  Time: {time.time() - t0:.1f}s\n")

print("Training conditional score o_s (NN, DSM)...")
t0 = time.time()
o_model, _ = train_conditional_score(
    xt_data,
    xtp1_data,
    dim=2,
    n_epochs=CONDITIONAL_SCORE_CONFIG["n_epochs"],
    hidden_dim=CONDITIONAL_SCORE_CONFIG["hidden_dim"],
    n_layers=CONDITIONAL_SCORE_CONFIG["n_layers"],
    device=DEVICE,
)
o_score = NNConditionalScore(o_model, dim=2, device=DEVICE)
print(f"  Time: {time.time() - t0:.1f}s\n")


def conditional_score_eval_2d(x_batch, s, x_cond_batch):
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
            device=DEVICE,
        )
        st = torch.full((len(x_arr),), float(s), dtype=torch.float32, device=DEVICE)
        out = o_model(inp, st).cpu().numpy()
    return np.clip(out, -o_score.clip, o_score.clip)


def reverse_sde_sample_2d(x_cond, n_samples=1, S=S_MAX, n_steps=N_DIFF_STEPS):
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
        score = conditional_score_eval_2d(x, s, x_cond_rep)
        x = x + (0.5 * x + score) * ds + np.random.randn(len(x_cond_rep), 2) * np.sqrt(ds)
    if n_cond == 1:
        return x.reshape(n_samples, 2)
    return x.reshape(n_cond, n_samples, 2)


def exact_transition_sample_2d(x_cond, n_samples=1):
    x_cond = np.asarray(x_cond, dtype=np.float64)
    if x_cond.ndim == 1:
        x_batch = np.tile(x_cond, (n_samples, 1))
        return np.asarray(langevin_step(x_batch, dt), dtype=np.float64).reshape(n_samples, 2)
    else:
        x_batch = np.repeat(x_cond.reshape(-1, 2), n_samples, axis=0)
        return np.asarray(langevin_step(x_batch, dt), dtype=np.float64).reshape(len(x_cond), n_samples, 2)


def composition_reverse_sde_2d(x_cond, n_samples=1, S=S_MAX, n_steps=N_DIFF_STEPS):
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
    clip = 15.0

    for step in range(n_steps):
        s = max(S - step * ds, 1e-6)
        x = np.clip(x, -5, 5)

        j_s = np.clip(j_score.score(x, s), -clip, clip)
        o_s = np.clip(conditional_score_eval_2d(x, s, x_cond_rep), -clip, clip)
        j_0 = np.clip(j_score.boltzmann_score(x), -clip, clip)

        j_s = np.nan_to_num(j_s, nan=0.0)
        o_s = np.nan_to_num(o_s, nan=0.0)
        j_0 = np.nan_to_num(j_0, nan=0.0)

        drift = np.clip(o_s - j_s + 0.5 * j_0 + 0.5 * x, -clip, clip)
        fkc_inc = (
            0.5 * dim
            + np.sum(j_s * (j_s - o_s), axis=-1)
            + 0.5 * np.sum(j_0 * (x + o_s - j_s), axis=-1)
        )
        fkc_inc = np.clip(fkc_inc, -50 / S, 50 / S)
        log_w += fkc_inc * ds
        x = x + drift * ds + np.random.randn(len(x_cond_rep), 2) * np.sqrt(ds)

    log_w = np.nan_to_num(log_w, nan=0.0, posinf=50.0, neginf=-50.0)
    if n_cond == 1:
        return x.reshape(n_samples, 2), log_w.reshape(n_samples)
    return x.reshape(n_cond, n_samples, 2), log_w.reshape(n_cond, n_samples)


ALPHA_TWEEDIE = np.exp(-S_MAX / 2.0)
SIGMA2_TWEEDIE = 1.0 - np.exp(-S_MAX)


def tweedie_twist_batch_fast_2d(x_proposals, t, n_rollouts=1, clip=10.0):
    x_proposals = np.asarray(x_proposals, dtype=np.float64).reshape(-1, 2)
    if t >= T:
        return np.zeros(len(x_proposals), dtype=np.float64)

    log_rewards = np.zeros((n_rollouts, len(x_proposals)), dtype=np.float64)
    for r in range(n_rollouts):
        x_hat = x_proposals.copy()
        total = np.zeros(len(x_proposals), dtype=np.float64)
        for t_future in range(t + 1, T + 1):
            z = np.random.randn(len(x_hat), 2)
            score = conditional_score_eval_2d(z, S_MAX, x_hat)
            x_hat = np.clip((z + SIGMA2_TWEEDIE * score) / ALPHA_TWEEDIE, -clip, clip)
            total += reward_model.log_G(x_hat, t_future)
        log_rewards[r] = total
    return logsumexp(log_rewards, axis=0) - np.log(n_rollouts)


def sample_reference_trajectories_2d(n_trajectories):
    trajs = np.zeros((n_trajectories, T + 1, 2), dtype=np.float64)
    trajs[:, 0] = x_0
    x = np.tile(x_0, (n_trajectories, 1))
    for t_idx in range(T):
        x = exact_transition_sample_2d(x, n_samples=1).reshape(n_trajectories, 2)
        trajs[:, t_idx + 1] = x
    return trajs


def estimate_log_Z_exact(n_samples, batch_size=2000):
    log_rewards = np.zeros(n_samples, dtype=np.float64)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = end - start
        trajs = np.zeros((batch, T + 1, 2), dtype=np.float64)
        trajs[:, 0] = x_0
        x = np.tile(x_0, (batch, 1))
        for t_idx in range(T):
            x = exact_transition_sample_2d(x, n_samples=1).reshape(batch, 2)
            trajs[:, t_idx + 1] = x
        step_logs = [reward_model.log_G(trajs[:, t_idx], t_idx) for t_idx in range(T + 1)]
        log_rewards[start:end] = np.sum(np.stack(step_logs, axis=1), axis=1)
    return float(logsumexp(log_rewards) - np.log(n_samples))


def estimate_log_Z_from_chain(step_sampler, n_samples, batch_size=2000):
    log_rewards = np.zeros(n_samples, dtype=np.float64)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = end - start
        trajs = np.zeros((batch, T + 1, 2), dtype=np.float64)
        trajs[:, 0] = x_0
        x = np.tile(x_0, (batch, 1))
        for t_idx in range(T):
            x = np.asarray(step_sampler(x, n_samples=1)).reshape(batch, 2)
            trajs[:, t_idx + 1] = x
        step_logs = [reward_model.log_G(trajs[:, t_idx], t_idx) for t_idx in range(T + 1)]
        log_rewards[start:end] = np.sum(np.stack(step_logs, axis=1), axis=1)
    return float(logsumexp(log_rewards) - np.log(n_samples))


print(f"Collecting {TWIST_CONFIG['n_ref']} reference trajectories...")
n_ref = TWIST_CONFIG["n_ref"]
ref_trajs = sample_reference_trajectories_2d(n_ref)
future_targets = build_future_reward_targets(ref_trajs, T, reward_model.log_G)
print(f"  Mean future target at t=0: {future_targets[:, 0].mean():.6f}")
print(f"  log Z (MC estimate): {np.log(future_targets[:, 0].mean() + 1e-300):.4f}\n")


print("--- Phase 3: Ground truth ---\n")
print(f"True-dynamics log Z via {GT_CONFIG['n_gt']} MC trajectories...")
n_gt = GT_CONFIG["n_gt"]
log_Z_gt = estimate_log_Z_exact(n_gt)
print(f"  true-dynamics log Z     = {log_Z_gt:.4f}")
learned_log_Z = estimate_log_Z_from_chain(reverse_sde_sample_2d, n_gt)
print(f"  learned-proposal log Z = {learned_log_Z:.4f}\n")


print("--- Phase 4: Training twists ---\n")
print("Training Take 3 twist (MC regression, positive psi)...")
t0 = time.time()
twist_mc_model, _ = train_positive_twist_mc(
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
twist_mc = PositiveNNTwist(twist_mc_model, dim=2, device=DEVICE)
print(f"  Time: {time.time() - t0:.1f}s\n")

print("Training Take 3 twist (TD, positive psi)...")
t0 = time.time()
twist_td_model, _ = train_positive_twist_td(
    ref_trajs,
    dim=2,
    T=T,
    log_G_fn=reward_model.log_G,
    terminal_targets=np.ones(n_ref),
    n_epochs=TWIST_CONFIG["n_epochs"],
    hidden_dim=TWIST_CONFIG["hidden_dim"],
    n_layers=TWIST_CONFIG["n_layers"],
    device=DEVICE,
    loss_space=TWIST_CONFIG["loss_space"],
)
twist_td = PositiveNNTwist(twist_td_model, dim=2, device=DEVICE)
print(f"  Time: {time.time() - t0:.1f}s\n")

print("Training Take 3 twist (KL / Lemma 3)...")
t0 = time.time()
twist_kl_model, _ = train_positive_twist_kl(
    ref_trajs,
    future_targets,
    dim=2,
    T=T,
    n_epochs=TWIST_CONFIG["n_epochs"],
    hidden_dim=TWIST_CONFIG["hidden_dim"],
    n_layers=TWIST_CONFIG["n_layers"],
    device=DEVICE,
)
twist_kl = PositiveNNTwist(twist_kl_model, dim=2, device=DEVICE)
print(f"  Time: {time.time() - t0:.1f}s\n")


def run_twisted_method(label, twist_kind="tweedie", twist_model=None, use_composition=False, use_fkc=True,
                       n_trials_local=n_trials, proposal_sampler=None):
    print(f"\nRunning {label}...")
    data = {"log_Z": [], "ess": [], "time": []}
    if proposal_sampler is None:
        proposal_sampler = reverse_sde_sample_2d
    for trial in range(n_trials_local):
        np.random.seed(trial * 41)
        t0 = time.time()
        particles = np.tile(x_0, (K, 1))
        ess_list = []

        if twist_kind == "tweedie":
            log_psi_init = float(tweedie_twist_batch_fast_2d(x_0.reshape(1, 2), 0, n_rollouts=1)[0])
        else:
            log_psi_init = float(twist_model(x_0.reshape(1, 2), 0))

        log_Z = log_psi_init
        log_psi_cached = np.full(K, log_psi_init)

        for t in range(1, T + 1):
            if use_composition:
                x_prop, log_fkc = composition_reverse_sde_2d(particles, n_samples=M)
                if not use_fkc:
                    proposals = x_prop
                else:
                    # True two-stage Take 2: Stage 1 resamples to an empirical p_ref proxy,
                    # then Stage 2 runs the same twist IS as Take 1 on the resampled cloud.
                    indices_stage1, _ = sample_row_indices(log_fkc, n_draws=M)
                    proposals = np.take_along_axis(x_prop, indices_stage1[..., None], axis=1)
            else:
                proposals = np.asarray(proposal_sampler(particles, n_samples=M)).reshape(K, M, 2)

            log_G = reward_model.log_G(proposals.reshape(-1, 2), t).reshape(K, M)
            if t < T:
                if twist_kind == "tweedie":
                    log_twist = tweedie_twist_batch_fast_2d(proposals.reshape(-1, 2), t, n_rollouts=1).reshape(K, M)
                else:
                    log_twist = np.asarray(twist_model(proposals.reshape(-1, 2), t)).reshape(K, M)
            else:
                log_twist = np.zeros((K, M))
            log_v = log_G + log_twist
            j_sel, log_norm = sample_row_indices(log_v, n_draws=1)
            new_particles = proposals[np.arange(K), j_sel]
            log_weights = log_norm - np.log(M) - log_psi_cached
            log_psi_next = log_twist[np.arange(K), j_sel] if t < T else np.zeros(K)
            log_Z += logsumexp(log_weights) - np.log(K)
            ess_list.append(compute_ess(log_weights))
            indices = systematic_resample(log_weights)
            particles = new_particles[indices]
            log_psi_cached = log_psi_next[indices]

        data["time"].append(time.time() - t0)
        data["log_Z"].append(log_Z)
        data["ess"].append(np.mean(ess_list))

    print(f"  log Z = {np.mean(data['log_Z']):.4f} ± {np.std(data['log_Z']):.4f}, ESS = {np.mean(data['ess']):.1f}")
    return data


def run_bootstrap(label, step_sampler):
    print(f"\nRunning {label}...")
    data = {"log_Z": [], "ess": [], "time": []}
    for trial in range(n_trials):
        np.random.seed(trial * 41)
        t0 = time.time()
        particles = np.tile(x_0, (K, 1))
        log_Z = 0.0
        ess_list = []
        for t in range(1, T + 1):
            particles = np.asarray(step_sampler(particles, n_samples=1)).reshape(K, 2)
            log_w = reward_model.log_G(particles, t)
            log_Z += logsumexp(log_w) - np.log(K)
            ess_list.append(compute_ess(log_w))
            indices = systematic_resample(log_w)
            particles = particles[indices]
        data["time"].append(time.time() - t0)
        data["log_Z"].append(log_Z)
        data["ess"].append(np.mean(ess_list))
    print(f"  log Z = {np.mean(data['log_Z']):.4f} ± {np.std(data['log_Z']):.4f}, ESS = {np.mean(data['ess']):.1f}")
    return data


print("--- Phase 5: Running SMC methods ---\n")
results = {}

results["Bootstrap"] = run_bootstrap("Bootstrap", reverse_sde_sample_2d)

results["Take1 (Tweedie)"] = run_twisted_method("Take1 (Tweedie)", twist_kind="tweedie")
results["Take3 (MC)"] = run_twisted_method("Take3 (MC)", twist_kind="learned", twist_model=twist_mc)
results["Take3 (TD)"] = run_twisted_method("Take3 (TD)", twist_kind="learned", twist_model=twist_td)
results["Take3 (KL)"] = run_twisted_method("Take3 (KL)", twist_kind="learned", twist_model=twist_kl)
results["Take2 (two-stage, FKC)"] = run_twisted_method(
    "Take2 (two-stage, FKC)",
    twist_kind="tweedie",
    use_composition=True,
    use_fkc=True,
    n_trials_local=n_trials_t2,
)
results["Take2 (no FKC)"] = run_twisted_method(
    "Take2 (no FKC)",
    twist_kind="tweedie",
    use_composition=True,
    use_fkc=False,
    n_trials_local=n_trials_t2,
)

exact_results = {}
exact_results["Bootstrap"] = run_bootstrap("Bootstrap (exact drift)", exact_transition_sample_2d)
exact_results["Take1 (Tweedie)"] = run_twisted_method(
    "Take1 (Tweedie, exact drift)",
    twist_kind="tweedie",
    proposal_sampler=exact_transition_sample_2d,
)
exact_results["Take3 (MC)"] = run_twisted_method(
    "Take3 (MC, exact drift)",
    twist_kind="learned",
    twist_model=twist_mc,
    proposal_sampler=exact_transition_sample_2d,
)
exact_results["Take3 (TD)"] = run_twisted_method(
    "Take3 (TD, exact drift)",
    twist_kind="learned",
    twist_model=twist_td,
    proposal_sampler=exact_transition_sample_2d,
)
exact_results["Take3 (KL)"] = run_twisted_method(
    "Take3 (KL, exact drift)",
    twist_kind="learned",
    twist_model=twist_kl,
    proposal_sampler=exact_transition_sample_2d,
)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"True-dynamics log Z:     {log_Z_gt:.4f}")
print(f"Learned-proposal log Z: {learned_log_Z:.4f}")
for key in ["Bootstrap", "Take1 (Tweedie)", "Take3 (MC)", "Take3 (TD)", "Take3 (KL)", "Take2 (two-stage, FKC)", "Take2 (no FKC)"]:
    vals = results[key]
    print(f"{key:<24} log Z = {np.mean(vals['log_Z']):.4f} ± {np.std(vals['log_Z']):.4f}, "
          f"ESS = {np.mean(vals['ess']):.1f}")

output = {
    "experiment": "run_exp2",
    "output_path": OUTPUT_PATH,
    "parameters": {
        "dim": dim,
        "dt": dt,
        "T": T,
        "K": K,
        "M": M,
        "S_MAX": S_MAX,
        "N_DIFF_STEPS": N_DIFF_STEPS,
        "n_trials": n_trials,
        "n_trials_t2": n_trials_t2,
        "x_0": x_0,
        "marginal_score_config": MARGINAL_SCORE_CONFIG,
        "conditional_score_config": CONDITIONAL_SCORE_CONFIG,
        "twist_config": TWIST_CONFIG,
        "gt_config": GT_CONFIG,
    },
    "ground_truth": {
        "true_dynamics_log_Z": log_Z_gt,
        "learned_proposal_log_Z": learned_log_Z,
        "log_Z": log_Z_gt,
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
