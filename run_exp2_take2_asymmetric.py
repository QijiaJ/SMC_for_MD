"""
Dedicated Exp 2 benchmark for the Take 2 regime where the marginal score j is
learned more accurately than the conditional score o.

This keeps the Müller-Brown reward from Exp 2, but:
  - trains j on much more equilibrium data / larger network / longer schedule
  - trains o on fewer transition pairs / smaller network / shorter schedule
  - optionally shrinks the effective o-score more than j at evaluation time

It also records direct score-accuracy diagnostics:
  - j_0 against the exact Boltzmann score -∇U
  - o_0 against the exact Euler-step transition score

and estimates both the weak learned-proposal log Z and the stage-1 FKC-resampled
proposal log Z to show whether the FKC step is actually helping.
"""

import argparse
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
from updated_code.muller_brown_core import (  # noqa: E402
    BarrierCrossingReward,
    MINIMUM_A,
    MINIMUM_B,
    MINIMUM_C,
    SADDLE_BC,
    grad_U,
    langevin_step,
    sample_adjacent_pairs,
)
from updated_code.learned_scores import (  # noqa: E402
    train_marginal_score,
    train_conditional_score,
    NNMarginalScore,
    NNConditionalScore,
)

from updated_code.common import (  # noqa: E402
    compute_ess,
    mean_std,
    sample_row_indices,
    save_json,
    systematic_resample,
)


DEFAULT_OUTPUT = os.path.join(
    ROOT,
    "updated_code",
    "results",
    "run_exp2_take2_asymmetric.json",
)


class ScaledMarginalScore:
    def __init__(self, base_score, scale):
        self.base_score = base_score
        self.scale = float(scale)

    def score(self, x, s):
        return self.scale * np.asarray(self.base_score.score(x, s), dtype=np.float64)

    def boltzmann_score(self, x):
        return self.scale * np.asarray(self.base_score.boltzmann_score(x), dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(description="Asymmetric-score Take 2 benchmark for Exp 2.")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")

    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--S-max", type=float, default=2.0)
    parser.add_argument("--n-diff-steps", type=int, default=25)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-trials-t2", type=int, default=6)
    parser.add_argument("--n-gt", type=int, default=20000)
    parser.add_argument("--n-gt-proposal", type=int, default=12000)
    parser.add_argument("--proposal-gt-inner-m", type=int, default=32)
    parser.add_argument("--tweedie-rollouts", type=int, default=4)

    parser.add_argument("--n-eq-per-chain", type=int, default=6000)
    parser.add_argument("--marginal-hidden-dim", type=int, default=256)
    parser.add_argument("--marginal-layers", type=int, default=4)
    parser.add_argument("--marginal-epochs", type=int, default=8000)
    parser.add_argument("--marginal-score-scale", type=float, default=1.0)

    parser.add_argument("--n-transition-pairs", type=int, default=2000)
    parser.add_argument("--conditional-hidden-dim", type=int, default=128)
    parser.add_argument("--conditional-layers", type=int, default=2)
    parser.add_argument("--conditional-epochs", type=int, default=1500)
    parser.add_argument("--conditional-score-scale", type=float, default=0.65)

    parser.add_argument("--score-eval-samples", type=int, default=2000)
    parser.add_argument("--pair-diagnostic-samples", type=int, default=4000)
    parser.add_argument("--lam-endpoint", type=float, default=5.0)
    parser.add_argument("--lam-saddle", type=float, default=3.0)
    return parser.parse_args()


def maybe_apply_quick_overrides(args):
    if not args.quick:
        return args

    args.K = min(args.K, 32)
    args.M = min(args.M, 8)
    args.n_trials = min(args.n_trials, 3)
    args.n_trials_t2 = min(args.n_trials_t2, 2)
    args.n_gt = min(args.n_gt, 4000)
    args.n_gt_proposal = min(args.n_gt_proposal, 2500)
    args.proposal_gt_inner_m = min(args.proposal_gt_inner_m, 16)
    args.n_eq_per_chain = min(args.n_eq_per_chain, 1500)
    args.marginal_hidden_dim = min(args.marginal_hidden_dim, 128)
    args.marginal_layers = min(args.marginal_layers, 3)
    args.marginal_epochs = min(args.marginal_epochs, 1500)
    args.n_transition_pairs = min(args.n_transition_pairs, 800)
    args.conditional_hidden_dim = min(args.conditional_hidden_dim, 96)
    args.conditional_layers = min(args.conditional_layers, 2)
    args.conditional_epochs = min(args.conditional_epochs, 600)
    args.score_eval_samples = min(args.score_eval_samples, 600)
    args.pair_diagnostic_samples = min(args.pair_diagnostic_samples, 1000)
    args.tweedie_rollouts = min(args.tweedie_rollouts, 2)
    return args


def collect_equilibrium_samples(n_eq_per_chain):
    eq_samples = []
    for start in [MINIMUM_A, MINIMUM_B, MINIMUM_C]:
        x = start.copy() + 0.02 * np.random.randn(2)
        for _ in range(2000):
            x = langevin_step(x, 0.003)
        chain = np.zeros((n_eq_per_chain, 2), dtype=np.float64)
        for idx in range(n_eq_per_chain):
            for _ in range(15):
                x = langevin_step(x, 0.003)
            chain[idx] = x
        eq_samples.append(chain)
    eq_samples = np.concatenate(eq_samples, axis=0)
    np.random.shuffle(eq_samples)
    return eq_samples


def conditional_score_eval(o_score_wrapper, x_batch, s, x_cond_batch):
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
            device=o_score_wrapper.device,
        )
        st = torch.full((len(x_arr),), float(s), dtype=torch.float32, device=o_score_wrapper.device)
        out = o_score_wrapper.model(inp, st).cpu().numpy()
    out = np.clip(out, -o_score_wrapper.clip, o_score_wrapper.clip)
    out = np.nan_to_num(out, nan=0.0)
    return out * float(o_score_wrapper.scale)


def reverse_sde_sample_2d(o_score_wrapper, x_cond, n_samples=1, S=2.0, n_steps=25):
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
        score = conditional_score_eval(o_score_wrapper, x, s, x_cond_rep)
        score = np.clip(score, -12.0, 12.0)
        drift = np.clip(0.5 * x + score, -12.0, 12.0)
        x = x + drift * ds + np.random.randn(len(x_cond_rep), 2) * np.sqrt(ds)
    if n_cond == 1:
        return x.reshape(n_samples, 2)
    return x.reshape(n_cond, n_samples, 2)


def exact_transition_sample_2d(x_cond, n_samples, dt):
    x_cond = np.asarray(x_cond, dtype=np.float64)
    if x_cond.ndim == 1:
        x_batch = np.tile(x_cond, (n_samples, 1))
        return np.asarray(langevin_step(x_batch, dt), dtype=np.float64).reshape(n_samples, 2)
    x_batch = np.repeat(x_cond.reshape(-1, 2), n_samples, axis=0)
    return np.asarray(langevin_step(x_batch, dt), dtype=np.float64).reshape(len(x_cond), n_samples, 2)


def composition_reverse_sde_2d(j_score_wrapper, o_score_wrapper, x_cond, n_samples=1, S=2.0, n_steps=25):
    x_cond_arr = np.asarray(x_cond, dtype=np.float64)
    if x_cond_arr.ndim == 1:
        x_cond_arr = x_cond_arr.reshape(1, 2)
    else:
        x_cond_arr = x_cond_arr.reshape(-1, 2)
    n_cond = len(x_cond_arr)
    x_cond_rep = np.repeat(x_cond_arr, n_samples, axis=0)
    ds = S / n_steps
    x = np.random.randn(len(x_cond_rep), 2)
    log_w = np.zeros(len(x_cond_rep), dtype=np.float64)
    clip = 15.0

    for step in range(n_steps):
        s = max(S - step * ds, 1e-6)
        x = np.clip(x, -5, 5)
        j_s = np.clip(j_score_wrapper.score(x, s), -clip, clip)
        o_s = np.clip(conditional_score_eval(o_score_wrapper, x, s, x_cond_rep), -clip, clip)
        j_0 = np.clip(j_score_wrapper.boltzmann_score(x), -clip, clip)

        j_s = np.nan_to_num(j_s, nan=0.0)
        o_s = np.nan_to_num(o_s, nan=0.0)
        j_0 = np.nan_to_num(j_0, nan=0.0)

        drift = np.clip(o_s - j_s + 0.5 * j_0 + 0.5 * x, -clip, clip)
        fkc_inc = (
            1.0
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


def make_tweedie_twist(o_score_wrapper, reward_model, T, S_MAX):
    alpha = np.exp(-S_MAX / 2.0)
    sigma2 = 1.0 - np.exp(-S_MAX)

    def tweedie_twist_batch(x_proposals, t, n_rollouts=1, clip=10.0):
        x_proposals_arr = np.asarray(x_proposals, dtype=np.float64).reshape(-1, 2)
        if t >= T:
            return np.zeros(len(x_proposals_arr), dtype=np.float64)

        log_rewards = np.zeros((n_rollouts, len(x_proposals_arr)), dtype=np.float64)
        for rollout_idx in range(n_rollouts):
            x_hat = x_proposals_arr.copy()
            total = np.zeros(len(x_proposals_arr), dtype=np.float64)
            for t_future in range(t + 1, T + 1):
                z = np.random.randn(len(x_hat), 2)
                score = conditional_score_eval(o_score_wrapper, z, S_MAX, x_hat)
                x_hat = np.clip((z + sigma2 * score) / alpha, -clip, clip)
                total += reward_model.log_G(x_hat, t_future)
            log_rewards[rollout_idx] = total
        return logsumexp(log_rewards, axis=0) - np.log(n_rollouts)

    return tweedie_twist_batch


def estimate_log_Z_exact(x_0, T, dt, reward_model, n_samples, batch_size=2000):
    log_rewards = np.zeros(n_samples, dtype=np.float64)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = end - start
        trajs = np.zeros((batch, T + 1, 2), dtype=np.float64)
        trajs[:, 0] = x_0
        x = np.tile(x_0, (batch, 1))
        for t_idx in range(T):
            x = exact_transition_sample_2d(x, n_samples=1, dt=dt).reshape(batch, 2)
            trajs[:, t_idx + 1] = x
        step_logs = [reward_model.log_G(trajs[:, t_idx], t_idx) for t_idx in range(T + 1)]
        log_rewards[start:end] = np.sum(np.stack(step_logs, axis=1), axis=1)
    return float(logsumexp(log_rewards) - np.log(n_samples))


def estimate_log_Z_from_step(step_sampler, x_0, T, reward_model, n_samples, batch_size=2000):
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


def evaluate_score_accuracy(j_score_wrapper, o_score_wrapper, dt, n_eval):
    eq_eval = []
    per_chain = max(n_eval // 3, 1)
    for start in [MINIMUM_A, MINIMUM_B, MINIMUM_C]:
        x = start.copy() + 0.02 * np.random.randn(2)
        for _ in range(2000):
            x = langevin_step(x, 0.003)
        chain = np.zeros((per_chain, 2), dtype=np.float64)
        for idx in range(per_chain):
            for _ in range(15):
                x = langevin_step(x, 0.003)
            chain[idx] = x
        eq_eval.append(chain)
    eq_eval = np.concatenate(eq_eval, axis=0)[:n_eval]
    true_j = -grad_U(eq_eval)
    pred_j = np.asarray(j_score_wrapper.boltzmann_score(eq_eval), dtype=np.float64).reshape(-1, 2)
    marginal_mse = float(np.mean(np.sum((pred_j - true_j) ** 2, axis=-1)))
    marginal_true_norm2 = float(np.mean(np.sum(true_j ** 2, axis=-1)))

    xt_eval, xtp1_eval = sample_adjacent_pairs(n_eval, dt_traj=dt)
    pred_o = np.asarray(conditional_score_eval(o_score_wrapper, xtp1_eval, 1e-4, xt_eval), dtype=np.float64).reshape(-1, 2)
    mean = xt_eval - dt * grad_U(xt_eval)
    true_o = -(xtp1_eval - mean) / (2.0 * dt)
    conditional_mse = float(np.mean(np.sum((pred_o - true_o) ** 2, axis=-1)))
    conditional_true_norm2 = float(np.mean(np.sum(true_o ** 2, axis=-1)))

    return {
        "marginal_boltzmann_mse": marginal_mse,
        "conditional_transition_mse": conditional_mse,
        "marginal_boltzmann_relative_mse": float(marginal_mse / max(marginal_true_norm2, 1e-12)),
        "conditional_transition_relative_mse": float(conditional_mse / max(conditional_true_norm2, 1e-12)),
        "conditional_over_marginal_relative_mse_ratio": float(
            (conditional_mse / max(conditional_true_norm2, 1e-12))
            / max(marginal_mse / max(marginal_true_norm2, 1e-12), 1e-12)
        ),
    }


def evaluate_take2_rho_proxy(j_score_wrapper, o_score_wrapper, dt, n_eval):
    """
    Practical proxy for the simplified Take 2 criterion in variance.tex.

    The appendix condition is
      rho > 0.5 * sqrt(E_m / E_c)
    with rho = Cov(eps_c, eps_m) / sqrt(E_c E_m).

    We estimate a paired, single-step proxy on shared transition samples:
      eps_c(x_t, x_{t+1}) = \hat{o}_0(x_{t+1}|x_t) - o_0(x_{t+1}|x_t)
      eps_m(x_{t+1})      = \hat{j}_0(x_{t+1}) - j_0(x_{t+1})

    This is not a literal plug-in estimate of the continuous-time integrated
    criterion, but it is a direct sanity check for whether the learned marginal
    and conditional score errors are aligned enough for cancellation to be
    plausible in the Take 2 composition drift.
    """
    xt_eval, xtp1_eval = sample_adjacent_pairs(n_eval, dt_traj=dt)

    pred_j_pair = np.asarray(j_score_wrapper.boltzmann_score(xtp1_eval), dtype=np.float64).reshape(-1, 2)
    true_j_pair = -grad_U(xtp1_eval)
    eps_m = pred_j_pair - true_j_pair

    pred_o = np.asarray(
        conditional_score_eval(o_score_wrapper, xtp1_eval, 1e-4, xt_eval),
        dtype=np.float64,
    ).reshape(-1, 2)
    mean = xt_eval - dt * grad_U(xt_eval)
    true_o = -(xtp1_eval - mean) / (2.0 * dt)
    eps_c = pred_o - true_o

    em_sq = np.sum(eps_m ** 2, axis=-1)
    ec_sq = np.sum(eps_c ** 2, axis=-1)
    inner_prod = np.sum(eps_c * eps_m, axis=-1)
    e_cm_sq = np.sum((eps_c - eps_m) ** 2, axis=-1)

    E_m = float(np.mean(em_sq))
    E_c = float(np.mean(ec_sq))
    cov_int = float(np.mean(inner_prod))
    rho_proxy = float(cov_int / max(np.sqrt(max(E_c * E_m, 1e-18)), 1e-12))
    threshold_proxy = float(0.5 * np.sqrt(max(E_m / max(E_c, 1e-18), 0.0)))
    criterion_margin = float(rho_proxy - threshold_proxy)
    simplified_cov_margin = float(2.0 * cov_int - E_m)

    component_cov = []
    component_rho = []
    for dim_idx in range(eps_c.shape[1]):
        cov_dim = float(np.mean(eps_c[:, dim_idx] * eps_m[:, dim_idx]))
        ec_dim = float(np.mean(eps_c[:, dim_idx] ** 2))
        em_dim = float(np.mean(eps_m[:, dim_idx] ** 2))
        rho_dim = float(cov_dim / max(np.sqrt(max(ec_dim * em_dim, 1e-18)), 1e-12))
        component_cov.append(cov_dim)
        component_rho.append(rho_dim)

    return {
        "sample_count": int(len(xt_eval)),
        "E_c_proxy": E_c,
        "E_m_proxy": E_m,
        "E_c_minus_m_proxy": float(np.mean(e_cm_sq)),
        "cov_int_proxy": cov_int,
        "rho_proxy": rho_proxy,
        "threshold_proxy": threshold_proxy,
        "criterion_margin": criterion_margin,
        "simplified_cov_margin": simplified_cov_margin,
        "criterion_satisfied": bool(rho_proxy > threshold_proxy),
        "component_cov_proxy": component_cov,
        "component_rho_proxy": component_rho,
        "notes": (
            "Single-step paired proxy for the simplified rho criterion. "
            "Uses shared transition samples at s≈0 rather than the full "
            "continuous-time integrated score-error process."
        ),
    }


def summarize_pair_dependence(xt, xtp1):
    xt = np.asarray(xt, dtype=np.float64).reshape(-1, 2)
    xtp1 = np.asarray(xtp1, dtype=np.float64).reshape(-1, 2)
    correlations = []
    for dim_idx in range(xt.shape[1]):
        corr = np.corrcoef(xt[:, dim_idx], xtp1[:, dim_idx])[0, 1]
        if not np.isfinite(corr):
            corr = 0.0
        correlations.append(float(corr))

    X = np.concatenate([xt, np.ones((len(xt), 1), dtype=np.float64)], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X, xtp1, rcond=None)
    pred = X @ coef
    resid_ss = np.sum((xtp1 - pred) ** 2, axis=0)
    total_ss = np.sum((xtp1 - np.mean(xtp1, axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - resid_ss / np.clip(total_ss, 1e-12, None)

    delta = xtp1 - xt
    return {
        "coord_correlations": correlations,
        "mean_abs_coord_correlation": float(np.mean(np.abs(correlations))),
        "linear_r2_per_coord": [float(x) for x in r2],
        "mean_linear_r2": float(np.mean(r2)),
        "mean_squared_displacement": float(np.mean(np.sum(delta ** 2, axis=-1))),
        "mean_step_norm": float(np.mean(np.linalg.norm(delta, axis=-1))),
    }


def run_bootstrap(label, x_0, T, K, step_sampler, reward_model, n_trials):
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
        data["time"].append(float(time.time() - t0))
        data["log_Z"].append(float(log_Z))
        data["ess"].append(float(np.mean(ess_list)))

    print(
        f"  log Z = {np.mean(data['log_Z']):.4f} ± {np.std(data['log_Z']):.4f}, "
        f"ESS = {np.mean(data['ess']):.1f}"
    )
    return data


def run_tweedie_method(
    label,
    x_0,
    T,
    K,
    M,
    reward_model,
    tweedie_twist_batch,
    proposal_sampler,
    n_trials,
    composition_mode=None,
):
    print(f"\nRunning {label}...")
    data = {"log_Z": [], "ess": [], "time": []}
    if composition_mode is not None:
        data["fkc_stage_ess"] = []

    for trial in range(n_trials):
        np.random.seed(trial * 41)
        t0 = time.time()
        particles = np.tile(x_0, (K, 1))
        ess_list = []
        fkc_ess = []

        log_psi_init = float(tweedie_twist_batch(x_0.reshape(1, 2), 0)[0])
        log_Z = log_psi_init
        log_psi_cached = np.full(K, log_psi_init, dtype=np.float64)

        for t in range(1, T + 1):
            if composition_mode is None:
                proposals = np.asarray(proposal_sampler(particles, n_samples=M)).reshape(K, M, 2)
            else:
                x_prop, log_fkc = proposal_sampler(particles, n_samples=M)
                fkc_ess.append(float(np.mean([compute_ess(row) for row in log_fkc])))
                if composition_mode == "single_stage":
                    proposals = x_prop
                elif composition_mode == "two_stage":
                    stage1_idx, _ = sample_row_indices(log_fkc, n_draws=M)
                    proposals = np.take_along_axis(x_prop, stage1_idx[..., None], axis=1)
                elif composition_mode == "no_fkc":
                    proposals = x_prop
                else:
                    raise ValueError(f"Unknown composition_mode={composition_mode!r}")

            log_G = reward_model.log_G(proposals.reshape(-1, 2), t).reshape(K, M)
            if t < T:
                log_twist = tweedie_twist_batch(proposals.reshape(-1, 2), t).reshape(K, M)
            else:
                log_twist = np.zeros((K, M), dtype=np.float64)

            if composition_mode == "single_stage":
                log_v = log_fkc + log_G + log_twist
            else:
                log_v = log_G + log_twist

            j_sel, log_norm = sample_row_indices(log_v, n_draws=1)
            new_particles = proposals[np.arange(K), j_sel]
            log_weights = log_norm - np.log(M) - log_psi_cached
            log_psi_next = log_twist[np.arange(K), j_sel] if t < T else np.zeros(K, dtype=np.float64)
            log_Z += logsumexp(log_weights) - np.log(K)
            ess_list.append(float(compute_ess(log_weights)))
            indices = systematic_resample(log_weights)
            particles = new_particles[indices]
            log_psi_cached = log_psi_next[indices]

        data["time"].append(float(time.time() - t0))
        data["log_Z"].append(float(log_Z))
        data["ess"].append(float(np.mean(ess_list)))
        if composition_mode is not None:
            data["fkc_stage_ess"].append(float(np.mean(fkc_ess)) if fkc_ess else None)

    suffix = ""
    if composition_mode is not None and data["fkc_stage_ess"]:
        suffix = f", Stage-1 FKC ESS = {np.nanmean(data['fkc_stage_ess']):.1f}"
    print(
        f"  log Z = {np.mean(data['log_Z']):.4f} ± {np.std(data['log_Z']):.4f}, "
        f"ESS = {np.mean(data['ess']):.1f}{suffix}"
    )
    return data


def main():
    args = maybe_apply_quick_overrides(parse_args())
    np.random.seed(args.seed)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dim = 2
    x_0 = MINIMUM_A.copy()
    reward_model = BarrierCrossingReward(
        T=args.T,
        target=MINIMUM_B,
        saddle=SADDLE_BC,
        lam_endpoint=args.lam_endpoint,
        lam_saddle=args.lam_saddle,
        T_mid=args.T // 2,
    )

    print("=" * 72)
    print("DEDICATED EXPERIMENT 2: Take 2 with stronger marginal than conditional")
    print("=" * 72)
    print(f"dt={args.dt}, T={args.T}, K={args.K}, M={args.M}")
    print(
        f"Marginal budget: per_chain={args.n_eq_per_chain}, hidden={args.marginal_hidden_dim}, "
        f"layers={args.marginal_layers}, epochs={args.marginal_epochs}, scale={args.marginal_score_scale}"
    )
    print(
        f"Conditional budget: pairs={args.n_transition_pairs}, hidden={args.conditional_hidden_dim}, "
        f"layers={args.conditional_layers}, epochs={args.conditional_epochs}, scale={args.conditional_score_scale}"
    )

    print("\n--- Phase 1: Collecting training data ---\n")
    print(f"Collecting equilibrium samples (3 chains x {args.n_eq_per_chain})...")
    eq_samples = collect_equilibrium_samples(args.n_eq_per_chain)
    print(f"  Total equilibrium samples: {len(eq_samples)}")

    print(f"Collecting {args.n_transition_pairs} transition pairs...")
    xt_data, xtp1_data = sample_adjacent_pairs(args.n_transition_pairs, dt_traj=args.dt)
    print("  Done.")

    print("\n--- Phase 2: Training scores ---\n")
    t0 = time.time()
    j_model, _ = train_marginal_score(
        eq_samples,
        dim=dim,
        n_epochs=args.marginal_epochs,
        hidden_dim=args.marginal_hidden_dim,
        n_layers=args.marginal_layers,
        device=args.device,
    )
    j_score_raw = NNMarginalScore(j_model, dim=dim, device=args.device)
    j_score = ScaledMarginalScore(j_score_raw, scale=args.marginal_score_scale)
    marginal_train_sec = time.time() - t0
    print(f"  Marginal score training time: {marginal_train_sec:.1f}s")

    t0 = time.time()
    o_model, _ = train_conditional_score(
        xt_data,
        xtp1_data,
        dim=dim,
        n_epochs=args.conditional_epochs,
        hidden_dim=args.conditional_hidden_dim,
        n_layers=args.conditional_layers,
        device=args.device,
    )
    o_score = NNConditionalScore(o_model, dim=dim, device=args.device)
    o_score.scale = float(args.conditional_score_scale)
    conditional_train_sec = time.time() - t0
    print(f"  Conditional score training time: {conditional_train_sec:.1f}s")

    print("\n--- Phase 3: Score diagnostics ---\n")
    score_accuracy = evaluate_score_accuracy(
        j_score_wrapper=j_score,
        o_score_wrapper=o_score,
        dt=args.dt,
        n_eval=args.score_eval_samples,
    )
    take2_rho_proxy = evaluate_take2_rho_proxy(
        j_score_wrapper=j_score,
        o_score_wrapper=o_score,
        dt=args.dt,
        n_eval=args.score_eval_samples,
    )
    print(
        f"  marginal rel-MSE = {score_accuracy['marginal_boltzmann_relative_mse']:.4f}, "
        f"conditional rel-MSE = {score_accuracy['conditional_transition_relative_mse']:.4f}, "
        f"ratio = {score_accuracy['conditional_over_marginal_relative_mse_ratio']:.2f}"
    )
    print(
        f"  rho proxy = {take2_rho_proxy['rho_proxy']:.4f}, "
        f"threshold = {take2_rho_proxy['threshold_proxy']:.4f}, "
        f"margin = {take2_rho_proxy['criterion_margin']:.4f}"
    )
    xt_diag, xtp1_diag = sample_adjacent_pairs(args.pair_diagnostic_samples, dt_traj=args.dt)
    pair_dependence = summarize_pair_dependence(xt_diag, xtp1_diag)
    print(
        f"  pair dependence: mean |corr| = {pair_dependence['mean_abs_coord_correlation']:.3f}, "
        f"mean linear R^2 = {pair_dependence['mean_linear_r2']:.3f}, "
        f"MSD = {pair_dependence['mean_squared_displacement']:.3f}"
    )

    tweedie_twist_batch = make_tweedie_twist(
        o_score_wrapper=o_score,
        reward_model=reward_model,
        T=args.T,
        S_MAX=args.S_max,
    )

    learned_step = lambda x, n_samples=1: reverse_sde_sample_2d(  # noqa: E731
        o_score_wrapper=o_score,
        x_cond=x,
        n_samples=n_samples,
        S=args.S_max,
        n_steps=args.n_diff_steps,
    )
    exact_step = lambda x, n_samples=1: exact_transition_sample_2d(x, n_samples=n_samples, dt=args.dt)  # noqa: E731
    composition_step = lambda x, n_samples=1: composition_reverse_sde_2d(  # noqa: E731
        j_score_wrapper=j_score,
        o_score_wrapper=o_score,
        x_cond=x,
        n_samples=n_samples,
        S=args.S_max,
        n_steps=args.n_diff_steps,
    )
    fkc_resampled_step = lambda x, n_samples=1: sample_fkc_resampled_step(  # noqa: E731
        composition_step=composition_step,
        x_cond=x,
        n_samples=n_samples,
        inner_M=args.proposal_gt_inner_m,
    )

    print("\n--- Phase 4: Ground-truth references ---\n")
    log_Z_exact = estimate_log_Z_exact(
        x_0=x_0,
        T=args.T,
        dt=args.dt,
        reward_model=reward_model,
        n_samples=args.n_gt,
    )
    log_Z_learned = estimate_log_Z_from_step(
        step_sampler=learned_step,
        x_0=x_0,
        T=args.T,
        reward_model=reward_model,
        n_samples=args.n_gt_proposal,
    )
    log_Z_fkc_proxy = estimate_log_Z_from_step(
        step_sampler=fkc_resampled_step,
        x_0=x_0,
        T=args.T,
        reward_model=reward_model,
        n_samples=args.n_gt_proposal,
    )
    print(f"  true-dynamics log Z      = {log_Z_exact:.4f}")
    print(f"  weak learned log Z       = {log_Z_learned:.4f}")
    print(f"  stage-1 FKC proxy log Z  = {log_Z_fkc_proxy:.4f}")

    print("\n--- Phase 5: Running methods ---\n")
    methods = {}
    methods["Bootstrap"] = run_bootstrap(
        "Bootstrap",
        x_0=x_0,
        T=args.T,
        K=args.K,
        step_sampler=learned_step,
        reward_model=reward_model,
        n_trials=args.n_trials,
    )
    methods["Bootstrap (FKC proxy proposal)"] = run_bootstrap(
        "Bootstrap (FKC proxy proposal)",
        x_0=x_0,
        T=args.T,
        K=args.K,
        step_sampler=fkc_resampled_step,
        reward_model=reward_model,
        n_trials=args.n_trials,
    )
    methods["Take1 (Tweedie)"] = run_tweedie_method(
        "Take1 (Tweedie)",
        x_0=x_0,
        T=args.T,
        K=args.K,
        M=args.M,
        reward_model=reward_model,
        tweedie_twist_batch=lambda x, t: tweedie_twist_batch(x, t, n_rollouts=args.tweedie_rollouts),
        proposal_sampler=learned_step,
        n_trials=args.n_trials,
    )
    methods["Take2 (single-stage, FKC)"] = run_tweedie_method(
        "Take2 (single-stage, FKC)",
        x_0=x_0,
        T=args.T,
        K=args.K,
        M=args.M,
        reward_model=reward_model,
        tweedie_twist_batch=lambda x, t: tweedie_twist_batch(x, t, n_rollouts=args.tweedie_rollouts),
        proposal_sampler=composition_step,
        n_trials=args.n_trials_t2,
        composition_mode="single_stage",
    )
    methods["Take2 (two-stage, FKC)"] = run_tweedie_method(
        "Take2 (two-stage, FKC)",
        x_0=x_0,
        T=args.T,
        K=args.K,
        M=args.M,
        reward_model=reward_model,
        tweedie_twist_batch=lambda x, t: tweedie_twist_batch(x, t, n_rollouts=args.tweedie_rollouts),
        proposal_sampler=composition_step,
        n_trials=args.n_trials_t2,
        composition_mode="two_stage",
    )
    methods["Take2 (no FKC)"] = run_tweedie_method(
        "Take2 (no FKC)",
        x_0=x_0,
        T=args.T,
        K=args.K,
        M=args.M,
        reward_model=reward_model,
        tweedie_twist_batch=lambda x, t: tweedie_twist_batch(x, t, n_rollouts=args.tweedie_rollouts),
        proposal_sampler=composition_step,
        n_trials=args.n_trials_t2,
        composition_mode="no_fkc",
    )

    exact_drift_methods = {
        "Bootstrap (exact drift)": run_bootstrap(
            "Bootstrap (exact drift)",
            x_0=x_0,
            T=args.T,
            K=args.K,
            step_sampler=exact_step,
            reward_model=reward_model,
            n_trials=args.n_trials,
        ),
        "Take1 (Tweedie, exact drift)": run_tweedie_method(
            "Take1 (Tweedie, exact drift)",
            x_0=x_0,
            T=args.T,
            K=args.K,
            M=args.M,
            reward_model=reward_model,
            tweedie_twist_batch=lambda x, t: tweedie_twist_batch(x, t, n_rollouts=args.tweedie_rollouts),
            proposal_sampler=exact_step,
            n_trials=args.n_trials,
        ),
    }

    summary = {
        key: {
            "log_Z": mean_std(vals["log_Z"]),
            "ess": mean_std(vals["ess"]),
            "time": mean_std(vals["time"]),
            **(
                {"fkc_stage_ess": mean_std([x for x in vals["fkc_stage_ess"] if x is not None])}
                if "fkc_stage_ess" in vals and any(x is not None for x in vals["fkc_stage_ess"])
                else {}
            ),
        }
        for key, vals in methods.items()
    }
    exact_summary = {
        key: {
            "log_Z": mean_std(vals["log_Z"]),
            "ess": mean_std(vals["ess"]),
            "time": mean_std(vals["time"]),
        }
        for key, vals in exact_drift_methods.items()
    }

    for key, vals in summary.items():
        vals["abs_logZ_error_true"] = float(abs(vals["log_Z"]["mean"] - log_Z_exact))
        vals["abs_logZ_error_learned"] = float(abs(vals["log_Z"]["mean"] - log_Z_learned))
        vals["abs_logZ_error_fkc_proxy"] = float(abs(vals["log_Z"]["mean"] - log_Z_fkc_proxy))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"True-dynamics log Z:     {log_Z_exact:.4f}")
    print(f"Weak learned log Z:      {log_Z_learned:.4f}")
    print(f"Stage-1 FKC proxy log Z: {log_Z_fkc_proxy:.4f}")
    for key, vals in summary.items():
        print(
            f"{key:<28} log Z = {vals['log_Z']['mean']:.4f} ± {vals['log_Z']['std']:.4f}, "
            f"|err true| = {vals['abs_logZ_error_true']:.4f}"
        )

    output = {
        "experiment": "run_exp2_take2_asymmetric",
        "output_path": args.output_path,
        "device": args.device,
        "quick": bool(args.quick),
        "parameters": {
            "dim": dim,
            "dt": args.dt,
            "T": args.T,
            "K": args.K,
            "M": args.M,
            "S_MAX": args.S_max,
            "n_diff_steps": args.n_diff_steps,
            "n_trials": args.n_trials,
            "n_trials_t2": args.n_trials_t2,
            "n_gt": args.n_gt,
            "n_gt_proposal": args.n_gt_proposal,
            "proposal_gt_inner_m": args.proposal_gt_inner_m,
            "tweedie_rollouts": args.tweedie_rollouts,
            "reward": {
                "lam_endpoint": args.lam_endpoint,
                "lam_saddle": args.lam_saddle,
                "T_mid": args.T // 2,
            },
            "marginal_score_config": {
                "n_eq_per_chain": args.n_eq_per_chain,
                "hidden_dim": args.marginal_hidden_dim,
                "n_layers": args.marginal_layers,
                "n_epochs": args.marginal_epochs,
                "score_scale": args.marginal_score_scale,
            },
            "conditional_score_config": {
                "n_transition_pairs": args.n_transition_pairs,
                "hidden_dim": args.conditional_hidden_dim,
                "n_layers": args.conditional_layers,
                "n_epochs": args.conditional_epochs,
                "score_scale": args.conditional_score_scale,
            },
            "score_eval_samples": args.score_eval_samples,
            "pair_diagnostic_samples": args.pair_diagnostic_samples,
        },
        "training_times_sec": {
            "marginal_score": float(marginal_train_sec),
            "conditional_score": float(conditional_train_sec),
        },
        "score_accuracy": score_accuracy,
        "take2_rho_proxy": take2_rho_proxy,
        "pair_dependence": pair_dependence,
        "ground_truth": {
            "true_dynamics_log_Z": float(log_Z_exact),
            "weak_learned_proposal_log_Z": float(log_Z_learned),
            "stage1_fkc_proxy_log_Z": float(log_Z_fkc_proxy),
        },
        "methods": methods,
        "exact_drift_methods": exact_drift_methods,
        "summary": summary,
        "exact_drift_summary": exact_summary,
    }
    save_json(output, args.output_path)
    print(f"\nSaved JSON results to {args.output_path}")


def sample_fkc_resampled_step(composition_step, x_cond, n_samples, inner_M):
    x_cond_arr = np.asarray(x_cond, dtype=np.float64)
    scalar = x_cond_arr.ndim == 1
    if scalar:
        x_cond_arr = x_cond_arr.reshape(1, 2)
    x_props, log_fkc = composition_step(x_cond_arr, n_samples=inner_M)
    row_idx, _ = sample_row_indices(log_fkc, n_draws=n_samples)
    if n_samples == 1:
        x_next = x_props[np.arange(len(x_cond_arr)), row_idx]
    else:
        x_next = np.take_along_axis(x_props, row_idx[..., None], axis=1)
    return x_next[0] if scalar else x_next


if __name__ == "__main__":
    main()
