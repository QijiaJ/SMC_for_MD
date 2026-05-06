"""
Experiment 8: non-Markovian route-history reward on the 2D coupled double-well.

This benchmark is designed to be genuinely path-dependent in x_t alone.
Trajectories start in the left well and are rewarded for:

1. reaching the right well at the final time, and
2. committing to the upper gate before the lower gate.

The route preference depends on the first gate hit over the whole path, so the
reward is not Markovian in the physical state x_t alone. We therefore augment
the SMC state with a small route-progress variable and learn Take 3 twists on
that augmented state.
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.special import logsumexp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXPERIMENTS_DIR = os.path.join(ROOT, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

from updated_code.learned_scores import (  # noqa: E402
    train_marginal_score,
    train_conditional_score,
    NNMarginalScore,
    NNConditionalScore,
)

from updated_code.common import (  # noqa: E402
    systematic_resample,
    compute_ess,
    build_future_reward_targets,
    sample_row_indices,
    mean_std,
    save_json,
)
from updated_code.fixed_twist import (  # noqa: E402
    train_positive_twist_mc,
    train_positive_twist_td,
    train_positive_twist_kl,
    PositiveNNTwist,
)


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


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


WELL_R = np.array([1.0, -0.25], dtype=np.float64)
WELL_L = np.array([-1.0, 0.25], dtype=np.float64)
DT_DYNAMICS = 0.02


def langevin_step(x, dt=None, grad_clip=10.0):
    if dt is None:
        dt = DT_DYNAMICS
    x = np.asarray(x, dtype=np.float64)
    scalar = x.ndim == 1
    if scalar:
        x = x[None, :]
    g = np.clip(grad_U(x), -grad_clip, grad_clip)
    noise = np.random.randn(*x.shape) * np.sqrt(2.0 * dt)
    x_new = x - dt * g + noise
    return x_new[0] if scalar else x_new


class FirstHitUpperReward:
    """
    Path-dependent reward carried by a route-progress flag.

    The augmented state is
      z_t = [x_t[0], x_t[1], 1{upper-first already decided}, 1{lower-first already decided}].
    """

    def __init__(
        self,
        T,
        target,
        x_gate_halfwidth=0.35,
        y_gate_threshold=0.45,
        beta_upper=0.25,
        beta_lower=0.35,
        lam_endpoint=10.0,
        endpoint_radius=0.5,
    ):
        self.T = int(T)
        self.target = np.asarray(target, dtype=np.float64)
        self.x_gate_halfwidth = float(x_gate_halfwidth)
        self.y_gate_threshold = float(y_gate_threshold)
        self.beta_upper = float(beta_upper)
        self.beta_lower = float(beta_lower)
        self.lam_endpoint = float(lam_endpoint)
        self.endpoint_radius = float(endpoint_radius)
        self.hist_times = list(range(max(1, self.T // 4), min(self.T, (3 * self.T) // 4) + 1))

    def initial_flags(self, n_particles):
        return np.zeros(int(n_particles), dtype=np.int64)

    def advance_flags(self, x, flags):
        x = np.asarray(x, dtype=np.float64).reshape(-1, 2)
        flags = np.asarray(flags, dtype=np.int64).reshape(-1).copy()
        if len(x) != len(flags):
            raise ValueError("x and flags must have matching batch size")

        undecided = flags == 0
        upper = undecided & (np.abs(x[:, 0]) <= self.x_gate_halfwidth) & (x[:, 1] >= self.y_gate_threshold)
        lower = undecided & (np.abs(x[:, 0]) <= self.x_gate_halfwidth) & (x[:, 1] <= -self.y_gate_threshold)
        flags[upper] = 1
        flags[lower] = -1
        return flags

    def augment_states(self, x, flags):
        x = np.asarray(x, dtype=np.float64).reshape(-1, 2)
        flags = np.asarray(flags, dtype=np.int64).reshape(-1)
        up = (flags == 1).astype(np.float64)
        down = (flags == -1).astype(np.float64)
        return np.column_stack([x, up, down])

    def augment_trajectories(self, x_trajs):
        x_trajs = np.asarray(x_trajs, dtype=np.float64)
        n_traj, t_steps, _ = x_trajs.shape
        flags = np.zeros((n_traj, t_steps), dtype=np.int64)
        current = self.initial_flags(n_traj)
        aug = np.zeros((n_traj, t_steps, 4), dtype=np.float64)
        aug[:, 0] = self.augment_states(x_trajs[:, 0], current)
        for t in range(1, t_steps):
            current = self.advance_flags(x_trajs[:, t], current)
            flags[:, t] = current
            aug[:, t] = self.augment_states(x_trajs[:, t], current)
        return aug, flags

    def log_G(self, aug_state, t):
        aug = np.asarray(aug_state, dtype=np.float64).reshape(-1, 4)
        x = aug[:, :2]
        up = aug[:, 2]
        down = aug[:, 3]
        log_g = self.beta_upper * up - self.beta_lower * down
        if int(t) == self.T:
            diff = x - self.target
            log_g = log_g - self.lam_endpoint * np.sum(diff ** 2, axis=-1)
        return log_g

    def total_log_reward(self, aug_trajs):
        aug_trajs = np.asarray(aug_trajs, dtype=np.float64)
        total = np.zeros(aug_trajs.shape[0], dtype=np.float64)
        for t in range(1, aug_trajs.shape[1]):
            total += np.asarray(self.log_G(aug_trajs[:, t], t), dtype=np.float64).reshape(-1)
        return total

    def _extract_x_and_flags(self, trajs):
        trajs = np.asarray(trajs, dtype=np.float64)
        if trajs.shape[-1] == 2:
            aug, flags = self.augment_trajectories(trajs)
            return trajs, flags, aug
        if trajs.shape[-1] == 4:
            x_trajs = trajs[..., :2]
            flags = np.zeros(trajs.shape[:2], dtype=np.int64)
            flags[trajs[..., 2] > 0.5] = 1
            flags[trajs[..., 3] > 0.5] = -1
            return x_trajs, flags, trajs
        raise ValueError("Trajectories must have final dimension 2 or 4")

    def endpoint_distance(self, trajs):
        x_trajs, _, _ = self._extract_x_and_flags(trajs)
        return np.linalg.norm(x_trajs[:, -1] - self.target, axis=-1)

    def endpoint_success(self, trajs):
        return self.endpoint_distance(trajs) <= self.endpoint_radius

    def upper_first(self, trajs):
        _, flags, _ = self._extract_x_and_flags(trajs)
        return flags[:, -1] == 1

    def lower_first(self, trajs):
        _, flags, _ = self._extract_x_and_flags(trajs)
        return flags[:, -1] == -1

    def decision_made(self, trajs):
        _, flags, _ = self._extract_x_and_flags(trajs)
        return flags[:, -1] != 0

    def decision_time(self, trajs):
        _, flags, _ = self._extract_x_and_flags(trajs)
        decided = flags != 0
        first = np.argmax(decided, axis=1)
        first[~np.any(decided, axis=1)] = self.T + 1
        return first.astype(np.float64)

    def joint_success(self, trajs):
        return self.endpoint_success(trajs) & self.upper_first(trajs)


class CenteredLogTwistNet(nn.Module):
    def __init__(self, dim, T, hidden_dim=128, n_layers=3):
        super().__init__()
        self.time_emb = nn.Embedding(T + 1, 32)
        layers = []
        in_dim = dim + 32
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t_idx):
        t_emb = self.time_emb(t_idx)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h).squeeze(-1)


class CenteredLogTwist:
    def __init__(self, model, dim, log_centers, device="cpu"):
        self.model = model
        self.dim = dim
        self.log_centers = np.asarray(log_centers, dtype=np.float64).reshape(-1)
        self.device = device

    def centered_value(self, x, t):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64)).reshape(-1, self.dim)
        n_pts = len(x)
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=self.device)
            tt = torch.full((n_pts,), int(t), dtype=torch.long, device=self.device)
            out = self.model(xt, tt).cpu().numpy()
        return out.squeeze() if n_pts > 1 else float(out.squeeze())

    def __call__(self, x, t):
        centered = np.asarray(self.centered_value(x, t), dtype=np.float64)
        out = centered + self.log_centers[int(t)]
        if np.ndim(out) == 0:
            return float(out)
        return out


def make_log_center_schedule(future_targets, eps=1e-300):
    future_targets = np.asarray(future_targets, dtype=np.float64)
    log_targets = np.log(np.clip(future_targets, eps, None))
    centers = np.median(log_targets, axis=0)
    centers[-1] = 0.0
    return centers


def train_centered_log_twist_mc(
    trajectories,
    future_targets,
    dim,
    T,
    log_centers,
    n_epochs=2000,
    batch_size=256,
    lr=1e-3,
    hidden_dim=128,
    n_layers=3,
    device="cpu",
    verbose=True,
    target_clip=20.0,
):
    traj = np.asarray(trajectories, dtype=np.float64).reshape(-1, T + 1, dim)
    targets = np.asarray(future_targets, dtype=np.float64).reshape(-1, T + 1)
    log_targets = np.log(np.clip(targets, 1e-300, None))
    centered = np.clip(log_targets - np.asarray(log_centers, dtype=np.float64)[None, :], -target_clip, target_clip)
    n_traj = traj.shape[0]

    model = CenteredLogTwistNet(dim, T, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    traj_t = torch.tensor(traj, dtype=torch.float32, device=device)
    target_t = torch.tensor(centered, dtype=torch.float32, device=device)
    losses = []

    for epoch in range(n_epochs):
        idx = torch.randint(0, n_traj, (batch_size,), device=device)
        t_batch = torch.randint(0, T + 1, (batch_size,), device=device)
        x_batch = traj_t[idx, t_batch]
        y_batch = target_t[idx, t_batch]

        pred = model(x_batch, t_batch)
        loss = F.smooth_l1_loss(pred, y_batch, beta=1.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))

        if verbose and (epoch + 1) % 500 == 0:
            print(f"    [MC centered-log twist] epoch {epoch+1}/{n_epochs}: loss={loss.item():.6f}")

    model.eval()
    return model, losses


def train_centered_log_twist_td(
    trajectories,
    dim,
    T,
    log_G_fn,
    log_centers,
    n_epochs=2000,
    batch_size=256,
    lr=1e-3,
    hidden_dim=128,
    n_layers=3,
    device="cpu",
    verbose=True,
    target_clip=20.0,
):
    traj = np.asarray(trajectories, dtype=np.float64).reshape(-1, T + 1, dim)
    n_traj = traj.shape[0]

    model = CenteredLogTwistNet(dim, T, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    traj_t = torch.tensor(traj, dtype=torch.float32, device=device)
    center_t = torch.tensor(np.asarray(log_centers, dtype=np.float32), dtype=torch.float32, device=device)
    losses = []

    for epoch in range(n_epochs):
        idx = torch.randint(0, n_traj, (batch_size,), device=device)

        x_T = traj_t[idx, T]
        t_T = torch.full((batch_size,), T, device=device, dtype=torch.long)
        pred_T = model(x_T, t_T)
        loss_terminal = F.smooth_l1_loss(pred_T, torch.zeros_like(pred_T), beta=1.0)

        t_batch = torch.randint(0, T, (batch_size,), device=device)
        x_t = traj_t[idx, t_batch]
        x_tp1 = traj_t[idx, t_batch + 1]
        pred_t = model(x_t, t_batch)

        x_tp1_np = x_tp1.detach().cpu().numpy()
        t_np = t_batch.detach().cpu().numpy()
        log_g_np = []
        for i in range(batch_size):
            x_single = np.atleast_2d(x_tp1_np[i])
            log_g_np.append(float(np.asarray(log_G_fn(x_single, int(t_np[i]) + 1)).reshape(-1)[0]))
        log_g = torch.tensor(log_g_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            pred_tp1 = model(x_tp1, t_batch + 1)
            target = log_g + (center_t[t_batch + 1] - center_t[t_batch]) + pred_tp1
            target = torch.clamp(target, min=-target_clip, max=target_clip)
        loss_td = F.smooth_l1_loss(pred_t, target, beta=1.0)

        loss = loss_terminal + loss_td
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))

        if verbose and (epoch + 1) % 500 == 0:
            print(f"    [TD centered-log twist] epoch {epoch+1}/{n_epochs}: loss={loss.item():.6f}")

    model.eval()
    return model, losses


def parse_args():
    parser = argparse.ArgumentParser(
        description="Non-Markovian route-history benchmark on the 2D coupled double-well."
    )
    parser.add_argument(
        "--output-path",
        default=os.path.join(ROOT, "updated_code", "results", "run_exp8_nonmarkov_route.json"),
    )
    parser.add_argument(
        "--plot-base",
        default=os.path.join(ROOT, "updated_code", "results", "exp8_nonmarkov_route"),
    )
    parser.add_argument("--quick", action="store_true", help="Run a fast smoke-test configuration.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["exact", "learned"], default="exact")
    parser.add_argument("--T", type=int, default=36)
    parser.add_argument("--dt", type=float, default=DT_DYNAMICS)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--M", type=int, default=12)
    parser.add_argument("--n-trials", type=int, default=6)
    parser.add_argument("--n-trials-t2", type=int, default=4)
    parser.add_argument("--S-max", type=float, default=1.5)
    parser.add_argument("--n-diff-steps", type=int, default=20)
    parser.add_argument("--tweedie-rollouts", type=int, default=4)
    parser.add_argument("--score-epochs", type=int, default=3000)
    parser.add_argument("--twist-epochs", type=int, default=3500)
    parser.add_argument("--n-eq-per-chain", type=int, default=1800)
    parser.add_argument("--n-transition-pairs", type=int, default=6000)
    parser.add_argument(
        "--marginal-score-scale",
        type=float,
        default=1.0,
        help="Multiplicative scale applied to the learned marginal score j.",
    )
    parser.add_argument(
        "--conditional-score-scale",
        type=float,
        default=1.0,
        help="Multiplicative scale applied to the learned conditional score o.",
    )
    parser.add_argument("--n-ref", type=int, default=12000)
    parser.add_argument("--n-gt", type=int, default=16000)
    parser.add_argument("--x-gate-halfwidth", type=float, default=0.35)
    parser.add_argument("--y-gate-threshold", type=float, default=0.45)
    parser.add_argument("--beta-upper", type=float, default=0.04)
    parser.add_argument("--beta-lower", type=float, default=0.04)
    parser.add_argument("--lam-endpoint", type=float, default=10.0)
    parser.add_argument("--endpoint-radius", type=float, default=0.5)
    return parser.parse_args()


def apply_quick_mode(args):
    if not args.quick:
        return args
    args.K = min(args.K, 24)
    args.M = min(args.M, 6)
    args.n_trials = min(args.n_trials, 2)
    args.n_trials_t2 = min(args.n_trials_t2, 1)
    args.score_epochs = min(args.score_epochs, 80)
    args.twist_epochs = min(args.twist_epochs, 120)
    args.tweedie_rollouts = min(args.tweedie_rollouts, 2)
    args.n_eq_per_chain = min(args.n_eq_per_chain, 200)
    args.n_transition_pairs = min(args.n_transition_pairs, 400)
    args.n_ref = min(args.n_ref, 256)
    args.n_gt = min(args.n_gt, 320)
    return args


def sample_x_trajectories(step_sampler, x0, T, n_trajectories, batch_size=512):
    x_trajs = np.zeros((n_trajectories, T + 1, 2), dtype=np.float64)
    x0 = np.asarray(x0, dtype=np.float64)
    for start in range(0, n_trajectories, batch_size):
        end = min(start + batch_size, n_trajectories)
        batch = end - start
        x = np.tile(x0, (batch, 1))
        x_trajs[start:end, 0] = x
        for t in range(T):
            x = np.asarray(step_sampler(x, n_samples=1), dtype=np.float64).reshape(batch, 2)
            x_trajs[start:end, t + 1] = x
    return x_trajs


def normalize_log_weights(log_w):
    log_w = np.asarray(log_w, dtype=np.float64).reshape(-1)
    return np.exp(log_w - logsumexp(log_w))


def visitation_histogram(x_trajs, times, bins, hist_range, traj_weights=None):
    x_trajs = np.asarray(x_trajs, dtype=np.float64)
    times = list(times)
    window = x_trajs[:, times, :].reshape(-1, 2)
    weights = None
    if traj_weights is not None:
        traj_weights = np.asarray(traj_weights, dtype=np.float64).reshape(-1)
        weights = np.repeat(traj_weights / max(len(times), 1), len(times))
    hist, x_edges, y_edges = np.histogram2d(
        window[:, 0],
        window[:, 1],
        bins=bins,
        range=hist_range,
        weights=weights,
    )
    hist = np.asarray(hist, dtype=np.float64)
    hist /= max(np.sum(hist), 1e-300)
    return hist, x_edges, y_edges


def js_divergence(p_hist, q_hist, eps=1e-12):
    p = np.asarray(p_hist, dtype=np.float64).reshape(-1)
    q = np.asarray(q_hist, dtype=np.float64).reshape(-1)
    p = np.clip(p / max(np.sum(p), eps), eps, None)
    q = np.clip(q / max(np.sum(q), eps), eps, None)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    return float(np.sum(values * weights))


def sanitize_log_values(values, floor=-300.0, ceil=80.0):
    values = np.asarray(values, dtype=np.float64)
    return np.nan_to_num(values, nan=floor, posinf=ceil, neginf=floor)


def target_distribution_summary(x_trajs, aug_trajs, log_rewards, reward_model, bins, hist_range):
    weights = normalize_log_weights(log_rewards)
    hist, x_edges, y_edges = visitation_histogram(
        x_trajs,
        reward_model.hist_times,
        bins=bins,
        hist_range=hist_range,
        traj_weights=weights,
    )
    endpoint_success = reward_model.endpoint_success(aug_trajs).astype(np.float64)
    upper_first = reward_model.upper_first(aug_trajs).astype(np.float64)
    lower_first = reward_model.lower_first(aug_trajs).astype(np.float64)
    joint_success = reward_model.joint_success(aug_trajs).astype(np.float64)
    decision_made = reward_model.decision_made(aug_trajs).astype(np.float64)
    endpoint_distance = reward_model.endpoint_distance(aug_trajs)
    decision_time = reward_model.decision_time(aug_trajs)
    return {
        "endpoint_success_rate": weighted_mean(endpoint_success, weights),
        "upper_first_rate": weighted_mean(upper_first, weights),
        "lower_first_rate": weighted_mean(lower_first, weights),
        "joint_success_rate": weighted_mean(joint_success, weights),
        "decision_made_rate": weighted_mean(decision_made, weights),
        "endpoint_distance_mean": weighted_mean(endpoint_distance, weights),
        "decision_time_mean": weighted_mean(decision_time, weights),
        "visitation_histogram": hist,
        "hist_x_edges": x_edges,
        "hist_y_edges": y_edges,
    }


def summarize_method_trajectories(x_trajs, aug_trajs, reward_model, target_summary, bins, hist_range):
    hist, _, _ = visitation_histogram(
        x_trajs,
        reward_model.hist_times,
        bins=bins,
        hist_range=hist_range,
    )
    log_rewards = reward_model.total_log_reward(aug_trajs)
    endpoint_success = reward_model.endpoint_success(aug_trajs).astype(np.float64)
    upper_first = reward_model.upper_first(aug_trajs).astype(np.float64)
    lower_first = reward_model.lower_first(aug_trajs).astype(np.float64)
    joint_success = reward_model.joint_success(aug_trajs).astype(np.float64)
    decision_made = reward_model.decision_made(aug_trajs).astype(np.float64)
    endpoint_distance = reward_model.endpoint_distance(aug_trajs)
    decision_time = reward_model.decision_time(aug_trajs)
    return {
        "mean_log_reward": float(np.mean(log_rewards)),
        "endpoint_success_rate": float(np.mean(endpoint_success)),
        "upper_first_rate": float(np.mean(upper_first)),
        "lower_first_rate": float(np.mean(lower_first)),
        "joint_success_rate": float(np.mean(joint_success)),
        "decision_made_rate": float(np.mean(decision_made)),
        "endpoint_distance_mean": float(np.mean(endpoint_distance)),
        "decision_time_mean": float(np.mean(decision_time)),
        "visitation_jsd": js_divergence(hist, target_summary["visitation_histogram"]),
        "visitation_histogram": hist,
    }


def plot_summary_figure(plot_base, reward_model, target_summary, exemplar_trajs, hist_range):
    os.makedirs(os.path.dirname(plot_base), exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, "updated_code", "results", ".mplconfig"))
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    x_grid = np.linspace(hist_range[0][0], hist_range[0][1], 240)
    y_grid = np.linspace(hist_range[1][0], hist_range[1][1], 240)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = U(np.stack([xx, yy], axis=-1))
    levels = np.linspace(np.percentile(zz, 5), np.percentile(zz, 75), 18)

    panels = [
        ("Reference Target Occupancy", target_summary["visitation_histogram"], None),
        ("Bootstrap", None, exemplar_trajs.get("Bootstrap")),
        ("Terminal-only IS", None, exemplar_trajs.get("Terminal-only IS")),
        ("Take3 (KL)", None, exemplar_trajs.get("Take3 (KL)")),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), constrained_layout=True)
    for ax, (title, hist, trajs) in zip(axes.flat, panels):
        ax.contour(xx, yy, zz, levels=levels, colors="0.65", linewidths=0.6)
        if hist is not None:
            ax.imshow(
                hist.T,
                origin="lower",
                extent=(hist_range[0][0], hist_range[0][1], hist_range[1][0], hist_range[1][1]),
                aspect="auto",
                cmap="magma",
                alpha=0.92,
            )
        elif trajs is not None:
            n_plot = min(len(trajs), 48)
            x_trajs = np.asarray(trajs[:n_plot], dtype=np.float64)
            for traj in x_trajs:
                ax.plot(traj[:, 0], traj[:, 1], color="tab:blue", alpha=0.16, linewidth=1.0)
            window = x_trajs[:, reward_model.hist_times, :].reshape(-1, 2)
            ax.scatter(window[:, 0], window[:, 1], s=4, alpha=0.12, color="tab:orange")

        upper_box = Rectangle(
            (-reward_model.x_gate_halfwidth, reward_model.y_gate_threshold),
            2.0 * reward_model.x_gate_halfwidth,
            hist_range[1][1] - reward_model.y_gate_threshold,
            fill=False,
            linewidth=1.1,
            linestyle="--",
            edgecolor="tab:green",
        )
        lower_box = Rectangle(
            (-reward_model.x_gate_halfwidth, hist_range[1][0]),
            2.0 * reward_model.x_gate_halfwidth,
            reward_model.y_gate_threshold - hist_range[1][0],
            fill=False,
            linewidth=1.1,
            linestyle="--",
            edgecolor="tab:red",
        )
        ax.add_patch(upper_box)
        ax.add_patch(lower_box)
        endpoint_circle = plt.Circle(
            reward_model.target,
            reward_model.endpoint_radius,
            fill=False,
            color="tab:red",
            linewidth=1.2,
        )
        ax.add_patch(endpoint_circle)
        ax.scatter(WELL_L[0], WELL_L[1], color="tab:green", s=55, label="Start")
        ax.scatter(WELL_R[0], WELL_R[1], color="tab:red", s=55, label="Target")
        ax.set_title(title)
        ax.set_xlim(hist_range[0])
        ax.set_ylim(hist_range[1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.savefig(plot_base + ".png", dpi=180, bbox_inches="tight")
    fig.savefig(plot_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = apply_quick_mode(parse_args())
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 72)
    print("EXPERIMENT 8: Non-Markovian Route-History Reward")
    print("=" * 72)
    print(f"Using device: {DEVICE}")
    print(f"backend={args.backend}, T={args.T}, dt={args.dt}, K={args.K}, M={args.M}, quick={args.quick}")

    reward_model = FirstHitUpperReward(
        T=args.T,
        target=WELL_R,
        x_gate_halfwidth=args.x_gate_halfwidth,
        y_gate_threshold=args.y_gate_threshold,
        beta_upper=args.beta_upper,
        beta_lower=args.beta_lower,
        lam_endpoint=args.lam_endpoint,
        endpoint_radius=args.endpoint_radius,
    )
    x0 = WELL_L.copy()
    hist_bins = 44
    hist_range = [[-1.8, 1.8], [-1.6, 1.6]]

    score_config = {
        "n_eq_per_chain": args.n_eq_per_chain,
        "n_transition_pairs": args.n_transition_pairs,
        "hidden_dim": 192,
        "n_layers": 4,
        "n_epochs": args.score_epochs,
    }
    twist_config = {
        "n_ref": args.n_ref,
        "hidden_dim": 256,
        "n_layers": 4,
        "n_epochs": args.twist_epochs,
        "loss_space": "log",
        "mc_lr": 1e-3,
        "td_lr": 7e-4,
        "kl_lr": 1e-3,
        "center_target_clip": 20.0,
    }

    print("\n--- Phase 1: Collecting score-training data ---")
    eq_samples = []
    for start in [WELL_R, WELL_L]:
        x = start.copy() + 0.02 * np.random.randn(2)
        for _ in range(2000 if not args.quick else 50):
            x = langevin_step(x, 0.005)
        chain = np.zeros((score_config["n_eq_per_chain"], 2), dtype=np.float64)
        for i in range(score_config["n_eq_per_chain"]):
            for _ in range(12 if not args.quick else 2):
                x = langevin_step(x, 0.005)
            chain[i] = x
        eq_samples.append(chain)
    eq_samples = np.concatenate(eq_samples, axis=0)
    np.random.shuffle(eq_samples)

    n_walkers = 200 if not args.quick else 40
    walkers = np.zeros((n_walkers, 2), dtype=np.float64)
    for i in range(n_walkers):
        walkers[i] = [WELL_R, WELL_L][i % 2].copy() + 0.05 * np.random.randn(2)
    for _ in range(5000 if not args.quick else 80):
        walkers = langevin_step(walkers, 0.005)

    pairs_per_walker = (score_config["n_transition_pairs"] + n_walkers - 1) // n_walkers
    xt_list = []
    xtp1_list = []
    for _ in range(pairs_per_walker):
        for _ in range(20 if not args.quick else 2):
            walkers = langevin_step(walkers, 0.005)
        xt_list.append(walkers.copy())
        xtp1_list.append(langevin_step(walkers, args.dt))
    xt_data = np.concatenate(xt_list, axis=0)[:score_config["n_transition_pairs"]]
    xtp1_data = np.concatenate(xtp1_list, axis=0)[:score_config["n_transition_pairs"]]
    print(f"  Equilibrium samples: {len(eq_samples)}")
    print(f"  Transition pairs: {len(xt_data)}")

    print("\n--- Phase 2: Training score models ---")
    t0 = time.time()
    j_model, _ = train_marginal_score(
        eq_samples,
        dim=2,
        n_epochs=score_config["n_epochs"],
        hidden_dim=score_config["hidden_dim"],
        n_layers=score_config["n_layers"],
        device=DEVICE,
    )
    j_score = NNMarginalScore(j_model, dim=2, device=DEVICE)
    print(f"  Marginal score trained in {time.time() - t0:.1f}s")

    t0 = time.time()
    o_model, _ = train_conditional_score(
        xt_data,
        xtp1_data,
        dim=2,
        n_epochs=score_config["n_epochs"],
        hidden_dim=score_config["hidden_dim"],
        n_layers=score_config["n_layers"],
        device=DEVICE,
    )
    o_score = NNConditionalScore(o_model, dim=2, device=DEVICE)
    print(f"  Conditional score trained in {time.time() - t0:.1f}s")
    if args.marginal_score_scale != 1.0 or args.conditional_score_scale != 1.0:
        print(
            "  Applying score corruption: "
            f"marginal scale={args.marginal_score_scale:.3f}, "
            f"conditional scale={args.conditional_score_scale:.3f}"
        )

    def marginal_score_eval_2d(x_batch, s):
        raw = np.asarray(j_score.score(x_batch, s), dtype=np.float64)
        return np.clip(args.marginal_score_scale * raw, -j_score.clip, j_score.clip)

    def boltzmann_score_eval_2d(x_batch):
        raw = np.asarray(j_score.boltzmann_score(x_batch), dtype=np.float64)
        return np.clip(args.marginal_score_scale * raw, -j_score.clip, j_score.clip)

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
        return np.clip(args.conditional_score_scale * out, -o_score.clip, o_score.clip)

    def exact_transition_sample(x_cond, n_samples=1):
        x_cond = np.asarray(x_cond, dtype=np.float64)
        if x_cond.ndim == 1:
            x_batch = np.tile(x_cond, (n_samples, 1))
            return np.asarray(langevin_step(x_batch, args.dt), dtype=np.float64).reshape(n_samples, 2)
        x_batch = np.repeat(x_cond.reshape(-1, 2), n_samples, axis=0)
        return np.asarray(langevin_step(x_batch, args.dt), dtype=np.float64).reshape(len(x_cond), n_samples, 2)

    def reverse_sde_sample_2d(x_cond, n_samples=1, S=args.S_max, n_steps=args.n_diff_steps):
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

    def composition_reverse_sde_2d(x_cond, n_samples=1, S=args.S_max, n_steps=args.n_diff_steps):
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
            x = np.clip(x, -5.0, 5.0)
            j_s = np.clip(marginal_score_eval_2d(x, s), -clip, clip)
            o_s = np.clip(conditional_score_eval_2d(x, s, x_cond_rep), -clip, clip)
            j_0 = np.clip(boltzmann_score_eval_2d(x), -clip, clip)
            j_s = np.nan_to_num(j_s, nan=0.0)
            o_s = np.nan_to_num(o_s, nan=0.0)
            j_0 = np.nan_to_num(j_0, nan=0.0)

            drift = np.clip(o_s - j_s + 0.5 * j_0 + 0.5 * x, -clip, clip)
            fkc_inc = (
                0.5 * x.shape[-1]
                + np.sum(j_s * (j_s - o_s), axis=-1)
                + 0.5 * np.sum(j_0 * (x + o_s - j_s), axis=-1)
            )
            fkc_inc = np.clip(fkc_inc, -20.0 / S, 20.0 / S)
            log_w += fkc_inc * ds
            x = x + drift * ds + np.random.randn(len(x_cond_rep), 2) * np.sqrt(ds)

        if n_cond == 1:
            return x.reshape(n_samples, 2), log_w.reshape(n_samples)
        return x.reshape(n_cond, n_samples, 2), log_w.reshape(n_cond, n_samples)

    alpha_tweedie = np.exp(-args.S_max / 2.0)
    sigma2_tweedie = 1.0 - np.exp(-args.S_max)

    def tweedie_twist_batch_augmented(aug_states, t, n_rollouts=None, clip=10.0):
        aug_states = np.asarray(aug_states, dtype=np.float64).reshape(-1, 4)
        if t >= args.T:
            return np.zeros(len(aug_states), dtype=np.float64)
        if n_rollouts is None:
            n_rollouts = args.tweedie_rollouts

        x0_batch = aug_states[:, :2]
        flag0 = np.zeros(len(aug_states), dtype=np.int64)
        flag0[aug_states[:, 2] > 0.5] = 1
        flag0[aug_states[:, 3] > 0.5] = -1

        log_rewards = np.zeros((n_rollouts, len(aug_states)), dtype=np.float64)
        for r in range(n_rollouts):
            x_hat = x0_batch.copy()
            flags_hat = flag0.copy()
            total = np.zeros(len(aug_states), dtype=np.float64)
            for t_future in range(t + 1, args.T + 1):
                z = np.random.randn(len(x_hat), 2)
                score = conditional_score_eval_2d(z, args.S_max, x_hat)
                x_hat = np.clip((z + sigma2_tweedie * score) / alpha_tweedie, -clip, clip)
                flags_hat = reward_model.advance_flags(x_hat, flags_hat)
                aug_hat = reward_model.augment_states(x_hat, flags_hat)
                total += reward_model.log_G(aug_hat, t_future)
            log_rewards[r] = total
        return logsumexp(log_rewards, axis=0) - np.log(n_rollouts)

    def collect_target_summary(label, step_sampler):
        print(f"Sampling {args.n_gt} trajectories for {label} target summary...")
        x_trajs = sample_x_trajectories(step_sampler, x0, args.T, args.n_gt, batch_size=256)
        aug_trajs, _ = reward_model.augment_trajectories(x_trajs)
        log_rewards = reward_model.total_log_reward(aug_trajs)
        log_Z = float(logsumexp(log_rewards) - np.log(len(log_rewards)))
        summary = target_distribution_summary(x_trajs, aug_trajs, log_rewards, reward_model, hist_bins, hist_range)
        return {
            "log_Z": log_Z,
            "summary": summary,
            "x_trajs": x_trajs,
            "aug_trajs": aug_trajs,
            "log_rewards": log_rewards,
        }

    print("\n--- Phase 3: Target summaries and twist targets ---")
    target_exact = collect_target_summary("exact-dynamics", exact_transition_sample)
    target_learned = collect_target_summary("learned-proposal", reverse_sde_sample_2d)
    selected_target = target_exact if args.backend == "exact" else target_learned
    print(f"  Exact-dynamics log Z:    {target_exact['log_Z']:.4f}")
    print(f"  Learned-proposal log Z:  {target_learned['log_Z']:.4f}")
    print(f"  Chosen proposal GT log Z ({args.backend}): {selected_target['log_Z']:.4f}")
    print(
        "  Selected target route split: "
        f"upper={selected_target['summary']['upper_first_rate']:.3f}, "
        f"lower={selected_target['summary']['lower_first_rate']:.3f}, "
        f"undecided={1.0 - selected_target['summary']['decision_made_rate']:.3f}, "
        f"joint={selected_target['summary']['joint_success_rate']:.3f}"
    )

    ref_step_sampler = exact_transition_sample if args.backend == "exact" else reverse_sde_sample_2d
    print(f"Sampling {twist_config['n_ref']} reference trajectories for {args.backend} twist learning...")
    ref_x_trajs = sample_x_trajectories(ref_step_sampler, x0, args.T, twist_config["n_ref"], batch_size=256)
    ref_aug_trajs, _ = reward_model.augment_trajectories(ref_x_trajs)
    future_targets = build_future_reward_targets(ref_aug_trajs, args.T, reward_model.log_G)
    log_centers = make_log_center_schedule(future_targets)
    print(f"  Reference Monte Carlo log Z estimate: {np.log(np.mean(future_targets[:, 0]) + 1e-300):.4f}")

    print("\n--- Phase 4: Training twists on augmented state ---")
    t0 = time.time()
    twist_mc_model, mc_losses = train_centered_log_twist_mc(
        ref_aug_trajs,
        future_targets,
        dim=4,
        T=args.T,
        log_centers=log_centers,
        n_epochs=twist_config["n_epochs"],
        lr=twist_config["mc_lr"],
        hidden_dim=twist_config["hidden_dim"],
        n_layers=twist_config["n_layers"],
        device=DEVICE,
        target_clip=twist_config["center_target_clip"],
    )
    twist_mc = CenteredLogTwist(twist_mc_model, dim=4, log_centers=log_centers, device=DEVICE)
    print(f"  Take 3 MC twist: {time.time() - t0:.1f}s")

    t0 = time.time()
    twist_td_model, td_losses = train_centered_log_twist_td(
        ref_aug_trajs,
        dim=4,
        T=args.T,
        log_G_fn=reward_model.log_G,
        log_centers=log_centers,
        n_epochs=twist_config["n_epochs"],
        lr=twist_config["td_lr"],
        hidden_dim=twist_config["hidden_dim"],
        n_layers=twist_config["n_layers"],
        device=DEVICE,
        target_clip=twist_config["center_target_clip"],
    )
    twist_td = CenteredLogTwist(twist_td_model, dim=4, log_centers=log_centers, device=DEVICE)
    print(f"  Take 3 TD twist: {time.time() - t0:.1f}s")

    t0 = time.time()
    twist_kl_model, kl_losses = train_positive_twist_kl(
        ref_aug_trajs,
        future_targets,
        dim=4,
        T=args.T,
        n_epochs=twist_config["n_epochs"],
        lr=twist_config["kl_lr"],
        hidden_dim=twist_config["hidden_dim"],
        n_layers=twist_config["n_layers"],
        device=DEVICE,
    )
    twist_kl = PositiveNNTwist(twist_kl_model, dim=4, device=DEVICE)
    print(f"  Take 3 KL twist: {time.time() - t0:.1f}s")

    exemplar_trajs = {}

    def record_trial_metrics(data, x_trajs, aug_trajs):
        metrics = summarize_method_trajectories(
            x_trajs,
            aug_trajs,
            reward_model,
            selected_target["summary"],
            hist_bins,
            hist_range,
        )
        data["mean_log_reward"].append(metrics["mean_log_reward"])
        data["endpoint_success_rate"].append(metrics["endpoint_success_rate"])
        data["upper_first_rate"].append(metrics["upper_first_rate"])
        data["lower_first_rate"].append(metrics["lower_first_rate"])
        data["joint_success_rate"].append(metrics["joint_success_rate"])
        data["decision_made_rate"].append(metrics["decision_made_rate"])
        data["endpoint_distance_mean"].append(metrics["endpoint_distance_mean"])
        data["decision_time_mean"].append(metrics["decision_time_mean"])
        data["visitation_jsd"].append(metrics["visitation_jsd"])
        return metrics

    def new_metric_store():
        return {
            "log_Z": [],
            "ess": [],
            "time": [],
            "mean_log_reward": [],
            "endpoint_success_rate": [],
            "upper_first_rate": [],
            "lower_first_rate": [],
            "joint_success_rate": [],
            "decision_made_rate": [],
            "endpoint_distance_mean": [],
            "decision_time_mean": [],
            "visitation_jsd": [],
        }

    def run_bootstrap(label, step_sampler):
        print(f"\nRunning {label}...")
        data = new_metric_store()
        for trial in range(args.n_trials):
            np.random.seed(args.seed + 97 * trial)
            t0 = time.time()
            particles = np.tile(x0, (args.K, 1))
            flags = reward_model.initial_flags(args.K)
            x_trajs = np.zeros((args.K, args.T + 1, 2), dtype=np.float64)
            aug_trajs = np.zeros((args.K, args.T + 1, 4), dtype=np.float64)
            x_trajs[:, 0] = x0
            aug_trajs[:, 0] = reward_model.augment_states(x_trajs[:, 0], flags)
            log_Z = 0.0
            ess_list = []

            for t in range(1, args.T + 1):
                particles = np.asarray(step_sampler(particles, n_samples=1), dtype=np.float64).reshape(args.K, 2)
                flags = reward_model.advance_flags(particles, flags)
                aug = reward_model.augment_states(particles, flags)
                x_trajs[:, t] = particles
                aug_trajs[:, t] = aug
                log_w = reward_model.log_G(aug, t)
                log_Z += logsumexp(log_w) - np.log(args.K)
                ess_list.append(compute_ess(log_w))
                ancestors = systematic_resample(log_w)
                particles = particles[ancestors]
                flags = flags[ancestors]
                x_trajs = x_trajs[ancestors]
                aug_trajs = aug_trajs[ancestors]

            data["time"].append(time.time() - t0)
            data["log_Z"].append(float(log_Z))
            data["ess"].append(float(np.mean(ess_list)))
            record_trial_metrics(data, x_trajs, aug_trajs)
            if trial == 0 and label not in exemplar_trajs:
                exemplar_trajs[label] = x_trajs.copy()

        print(
            f"  log Z = {np.mean(data['log_Z']):.4f} ± {np.std(data['log_Z']):.4f}, "
            f"ESS = {np.mean(data['ess']):.2f}, "
            f"upper/lower = {np.mean(data['upper_first_rate']):.3f}/{np.mean(data['lower_first_rate']):.3f}, "
            f"joint success = {np.mean(data['joint_success_rate']):.3f}"
        )
        return data

    def run_terminal_only_is(label, step_sampler):
        print(f"\nRunning {label}...")
        data = new_metric_store()
        for trial in range(args.n_trials):
            np.random.seed(args.seed + 97 * trial)
            t0 = time.time()
            x_trajs = sample_x_trajectories(step_sampler, x0, args.T, args.K, batch_size=args.K)
            aug_trajs, _ = reward_model.augment_trajectories(x_trajs)
            log_w = reward_model.total_log_reward(aug_trajs)
            log_Z = float(logsumexp(log_w) - np.log(args.K))
            ess = float(compute_ess(log_w))
            ancestors = systematic_resample(log_w)
            sampled_x_trajs = x_trajs[ancestors]
            sampled_aug_trajs = aug_trajs[ancestors]
            data["time"].append(time.time() - t0)
            data["log_Z"].append(log_Z)
            data["ess"].append(ess)
            record_trial_metrics(data, sampled_x_trajs, sampled_aug_trajs)
            if trial == 0 and label not in exemplar_trajs:
                exemplar_trajs[label] = sampled_x_trajs.copy()

        print(
            f"  log Z = {np.mean(data['log_Z']):.4f} ± {np.std(data['log_Z']):.4f}, "
            f"ESS = {np.mean(data['ess']):.2f}, "
            f"upper/lower = {np.mean(data['upper_first_rate']):.3f}/{np.mean(data['lower_first_rate']):.3f}, "
            f"joint success = {np.mean(data['joint_success_rate']):.3f}"
        )
        return data

    def run_take3(label, step_sampler, twist_model):
        print(f"\nRunning {label}...")
        data = new_metric_store()
        for trial in range(args.n_trials):
            np.random.seed(args.seed + 97 * trial)
            t0 = time.time()
            particles = np.tile(x0, (args.K, 1))
            flags = reward_model.initial_flags(args.K)
            x_trajs = np.zeros((args.K, args.T + 1, 2), dtype=np.float64)
            aug_trajs = np.zeros((args.K, args.T + 1, 4), dtype=np.float64)
            x_trajs[:, 0] = x0
            aug_trajs[:, 0] = reward_model.augment_states(x_trajs[:, 0], flags)
            ess_list = []

            log_psi_init = float(sanitize_log_values(twist_model(aug_trajs[0, 0].reshape(1, 4), 0)))
            log_psi_cached = np.full(args.K, log_psi_init, dtype=np.float64)
            log_Z = log_psi_init

            for t in range(1, args.T + 1):
                proposals_x = np.asarray(step_sampler(particles, n_samples=args.M), dtype=np.float64).reshape(args.K, args.M, 2)
                proposal_flags = np.repeat(flags[:, None], args.M, axis=1).reshape(-1)
                proposal_flags = reward_model.advance_flags(proposals_x.reshape(-1, 2), proposal_flags).reshape(args.K, args.M)
                proposals_aug = reward_model.augment_states(
                    proposals_x.reshape(-1, 2),
                    proposal_flags.reshape(-1),
                ).reshape(args.K, args.M, 4)
                log_G = reward_model.log_G(proposals_aug.reshape(-1, 4), t).reshape(args.K, args.M)
                if t < args.T:
                    log_twist = sanitize_log_values(twist_model(proposals_aug.reshape(-1, 4), t)).reshape(args.K, args.M)
                else:
                    log_twist = np.zeros((args.K, args.M), dtype=np.float64)

                log_v = log_G + log_twist
                j_sel, log_norm = sample_row_indices(log_v, n_draws=1)
                chosen_x = proposals_x[np.arange(args.K), j_sel]
                chosen_aug = proposals_aug[np.arange(args.K), j_sel]
                chosen_flags = proposal_flags[np.arange(args.K), j_sel]
                x_trajs[:, t] = chosen_x
                aug_trajs[:, t] = chosen_aug
                log_weights = sanitize_log_values(log_norm - np.log(args.M) - log_psi_cached)
                log_Z += logsumexp(log_weights) - np.log(args.K)
                ess_list.append(compute_ess(log_weights))
                log_psi_next = log_twist[np.arange(args.K), j_sel] if t < args.T else np.zeros(args.K, dtype=np.float64)
                ancestors = systematic_resample(log_weights)
                particles = chosen_x[ancestors]
                flags = chosen_flags[ancestors]
                x_trajs = x_trajs[ancestors]
                aug_trajs = aug_trajs[ancestors]
                log_psi_cached = log_psi_next[ancestors]

            data["time"].append(time.time() - t0)
            data["log_Z"].append(float(log_Z))
            data["ess"].append(float(np.mean(ess_list)))
            record_trial_metrics(data, x_trajs, aug_trajs)
            if trial == 0 and label not in exemplar_trajs:
                exemplar_trajs[label] = x_trajs.copy()

        print(
            f"  log Z = {np.mean(data['log_Z']):.4f} ± {np.std(data['log_Z']):.4f}, "
            f"ESS = {np.mean(data['ess']):.2f}, "
            f"upper/lower = {np.mean(data['upper_first_rate']):.3f}/{np.mean(data['lower_first_rate']):.3f}, "
            f"joint success = {np.mean(data['joint_success_rate']):.3f}"
        )
        return data

    def run_take1(label, step_sampler):
        print(f"\nRunning {label}...")
        data = new_metric_store()
        for trial in range(args.n_trials):
            np.random.seed(args.seed + 97 * trial)
            t0 = time.time()
            particles = np.tile(x0, (args.K, 1))
            flags = reward_model.initial_flags(args.K)
            x_trajs = np.zeros((args.K, args.T + 1, 2), dtype=np.float64)
            aug_trajs = np.zeros((args.K, args.T + 1, 4), dtype=np.float64)
            x_trajs[:, 0] = x0
            aug_trajs[:, 0] = reward_model.augment_states(x_trajs[:, 0], flags)
            ess_list = []

            log_psi_init = float(sanitize_log_values(tweedie_twist_batch_augmented(aug_trajs[0, 0].reshape(1, 4), 0))[0])
            log_psi_cached = np.full(args.K, log_psi_init, dtype=np.float64)
            log_Z = log_psi_init

            for t in range(1, args.T + 1):
                proposals_x = np.asarray(step_sampler(particles, n_samples=args.M), dtype=np.float64).reshape(args.K, args.M, 2)
                proposal_flags = np.repeat(flags[:, None], args.M, axis=1).reshape(-1)
                proposal_flags = reward_model.advance_flags(proposals_x.reshape(-1, 2), proposal_flags).reshape(args.K, args.M)
                proposals_aug = reward_model.augment_states(
                    proposals_x.reshape(-1, 2),
                    proposal_flags.reshape(-1),
                ).reshape(args.K, args.M, 4)
                log_G = reward_model.log_G(proposals_aug.reshape(-1, 4), t).reshape(args.K, args.M)
                if t < args.T:
                    log_twist = sanitize_log_values(tweedie_twist_batch_augmented(proposals_aug.reshape(-1, 4), t)).reshape(
                        args.K, args.M
                    )
                else:
                    log_twist = np.zeros((args.K, args.M), dtype=np.float64)

                log_v = log_G + log_twist
                j_sel, log_norm = sample_row_indices(log_v, n_draws=1)
                chosen_x = proposals_x[np.arange(args.K), j_sel]
                chosen_aug = proposals_aug[np.arange(args.K), j_sel]
                chosen_flags = proposal_flags[np.arange(args.K), j_sel]
                x_trajs[:, t] = chosen_x
                aug_trajs[:, t] = chosen_aug
                log_weights = sanitize_log_values(log_norm - np.log(args.M) - log_psi_cached)
                log_Z += logsumexp(log_weights) - np.log(args.K)
                ess_list.append(compute_ess(log_weights))
                log_psi_next = log_twist[np.arange(args.K), j_sel] if t < args.T else np.zeros(args.K, dtype=np.float64)
                ancestors = systematic_resample(log_weights)
                particles = chosen_x[ancestors]
                flags = chosen_flags[ancestors]
                x_trajs = x_trajs[ancestors]
                aug_trajs = aug_trajs[ancestors]
                log_psi_cached = log_psi_next[ancestors]

            data["time"].append(time.time() - t0)
            data["log_Z"].append(float(log_Z))
            data["ess"].append(float(np.mean(ess_list)))
            record_trial_metrics(data, x_trajs, aug_trajs)
            if trial == 0 and label not in exemplar_trajs:
                exemplar_trajs[label] = x_trajs.copy()

        print(
            f"  log Z = {np.mean(data['log_Z']):.4f} ± {np.std(data['log_Z']):.4f}, "
            f"ESS = {np.mean(data['ess']):.2f}, "
            f"upper/lower = {np.mean(data['upper_first_rate']):.3f}/{np.mean(data['lower_first_rate']):.3f}, "
            f"joint success = {np.mean(data['joint_success_rate']):.3f}"
        )
        return data

    def run_take2(label):
        print(f"\nRunning {label}...")
        data = new_metric_store()
        for trial in range(args.n_trials_t2):
            np.random.seed(args.seed + 211 * trial)
            t0 = time.time()
            particles = np.tile(x0, (args.K, 1))
            flags = reward_model.initial_flags(args.K)
            x_trajs = np.zeros((args.K, args.T + 1, 2), dtype=np.float64)
            aug_trajs = np.zeros((args.K, args.T + 1, 4), dtype=np.float64)
            x_trajs[:, 0] = x0
            aug_trajs[:, 0] = reward_model.augment_states(x_trajs[:, 0], flags)
            ess_list = []

            log_psi_init = float(sanitize_log_values(tweedie_twist_batch_augmented(aug_trajs[0, 0].reshape(1, 4), 0))[0])
            log_psi_cached = np.full(args.K, log_psi_init, dtype=np.float64)
            log_Z = log_psi_init

            for t in range(1, args.T + 1):
                x_comp, log_fkc = composition_reverse_sde_2d(particles, n_samples=args.M)
                fkc_indices, _ = sample_row_indices(log_fkc, n_draws=args.M)
                proposals_x = np.take_along_axis(x_comp, fkc_indices[..., None], axis=1)
                proposal_flags = np.repeat(flags[:, None], args.M, axis=1).reshape(-1)
                proposal_flags = reward_model.advance_flags(proposals_x.reshape(-1, 2), proposal_flags).reshape(args.K, args.M)
                proposals_aug = reward_model.augment_states(
                    proposals_x.reshape(-1, 2),
                    proposal_flags.reshape(-1),
                ).reshape(args.K, args.M, 4)
                log_G = reward_model.log_G(proposals_aug.reshape(-1, 4), t).reshape(args.K, args.M)
                if t < args.T:
                    log_twist = sanitize_log_values(tweedie_twist_batch_augmented(proposals_aug.reshape(-1, 4), t)).reshape(
                        args.K, args.M
                    )
                else:
                    log_twist = np.zeros((args.K, args.M), dtype=np.float64)

                log_v = log_G + log_twist
                j_sel, log_norm = sample_row_indices(log_v, n_draws=1)
                chosen_x = proposals_x[np.arange(args.K), j_sel]
                chosen_aug = proposals_aug[np.arange(args.K), j_sel]
                chosen_flags = proposal_flags[np.arange(args.K), j_sel]
                x_trajs[:, t] = chosen_x
                aug_trajs[:, t] = chosen_aug
                log_weights = sanitize_log_values(log_norm - np.log(args.M) - log_psi_cached)
                log_Z += logsumexp(log_weights) - np.log(args.K)
                ess_list.append(compute_ess(log_weights))
                log_psi_next = log_twist[np.arange(args.K), j_sel] if t < args.T else np.zeros(args.K, dtype=np.float64)
                ancestors = systematic_resample(log_weights)
                particles = chosen_x[ancestors]
                flags = chosen_flags[ancestors]
                x_trajs = x_trajs[ancestors]
                aug_trajs = aug_trajs[ancestors]
                log_psi_cached = log_psi_next[ancestors]

            data["time"].append(time.time() - t0)
            data["log_Z"].append(float(log_Z))
            data["ess"].append(float(np.mean(ess_list)))
            record_trial_metrics(data, x_trajs, aug_trajs)
            if trial == 0 and label not in exemplar_trajs:
                exemplar_trajs[label] = x_trajs.copy()

        print(
            f"  log Z = {np.mean(data['log_Z']):.4f} ± {np.std(data['log_Z']):.4f}, "
            f"ESS = {np.mean(data['ess']):.2f}, "
            f"upper/lower = {np.mean(data['upper_first_rate']):.3f}/{np.mean(data['lower_first_rate']):.3f}, "
            f"joint success = {np.mean(data['joint_success_rate']):.3f}"
        )
        return data

    print("\n--- Phase 5: Running methods ---")
    proposal_sampler = exact_transition_sample if args.backend == "exact" else reverse_sde_sample_2d
    methods = {}
    methods["Bootstrap"] = run_bootstrap("Bootstrap", proposal_sampler)
    methods["Terminal-only IS"] = run_terminal_only_is("Terminal-only IS", proposal_sampler)
    methods["Take1 (Tweedie)"] = run_take1("Take1 (Tweedie)", proposal_sampler)
    methods["Take3 (MC)"] = run_take3("Take3 (MC)", proposal_sampler, twist_mc)
    methods["Take3 (TD)"] = run_take3("Take3 (TD)", proposal_sampler, twist_td)
    methods["Take3 (KL)"] = run_take3("Take3 (KL)", proposal_sampler, twist_kl)
    if args.backend == "learned":
        methods["Take2 (two-stage)"] = run_take2("Take2 (two-stage)")

    summary = {
        name: {metric: mean_std(values) for metric, values in data.items()}
        for name, data in methods.items()
    }

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Proposal backend: {args.backend}")
    print(f"Proposal GT log Z: {selected_target['log_Z']:.4f}")
    for name, stats in summary.items():
        print(
            f"{name:<24} log Z = {stats['log_Z']['mean']:.4f} ± {stats['log_Z']['std']:.4f}, "
            f"ESS = {stats['ess']['mean']:.2f}, "
            f"upper/lower = {stats['upper_first_rate']['mean']:.3f}/{stats['lower_first_rate']['mean']:.3f}, "
            f"JSD = {stats['visitation_jsd']['mean']:.3f}"
        )

    plot_summary_figure(args.plot_base, reward_model, selected_target["summary"], exemplar_trajs, hist_range)

    output = {
        "experiment": "run_exp8_nonmarkov_route",
        "description": (
            "2D coupled double-well benchmark with a first-hit upper-vs-lower route reward. "
            "The reward is non-Markovian in x_t alone and is implemented through an augmented route-progress state."
        ),
        "output_path": args.output_path,
        "plot_base": args.plot_base,
        "device": DEVICE,
        "parameters": {
            "backend": args.backend,
            "T": args.T,
            "dt": args.dt,
            "K": args.K,
            "M": args.M,
            "S_MAX": args.S_max,
            "n_diff_steps": args.n_diff_steps,
            "tweedie_rollouts": args.tweedie_rollouts,
            "n_trials": args.n_trials,
            "n_trials_t2": args.n_trials_t2,
            "quick": args.quick,
            "x0": x0,
            "score_config": score_config,
            "score_scaling": {
                "marginal_score_scale": args.marginal_score_scale,
                "conditional_score_scale": args.conditional_score_scale,
            },
            "twist_config": twist_config,
            "reward": {
                "type": "first_hit_upper_route",
                "target": reward_model.target,
                "x_gate_halfwidth": reward_model.x_gate_halfwidth,
                "y_gate_threshold": reward_model.y_gate_threshold,
                "beta_upper": reward_model.beta_upper,
                "beta_lower": reward_model.beta_lower,
                "lam_endpoint": reward_model.lam_endpoint,
                "endpoint_radius": reward_model.endpoint_radius,
                "hist_times": reward_model.hist_times,
            },
            "histogram": {
                "bins": hist_bins,
                "range": hist_range,
            },
        },
        "ground_truth": {
            "exact_dynamics": {
                "log_Z": target_exact["log_Z"],
                "summary": target_exact["summary"],
            },
            "learned_proposal": {
                "log_Z": target_learned["log_Z"],
                "summary": target_learned["summary"],
            },
            "selected_backend": args.backend,
            "selected_log_Z": selected_target["log_Z"],
        },
        "twist_training": {
            "reference_backend": args.backend,
            "reference_log_Z_estimate": float(np.log(np.mean(future_targets[:, 0]) + 1e-300)),
            "log_center_schedule": log_centers,
            "Take3 (MC)": {
                "objective": "MC_centered_log",
                "final_loss": float(mc_losses[-1]) if mc_losses else None,
            },
            "Take3 (TD)": {
                "objective": "TD_centered_log",
                "final_loss": float(td_losses[-1]) if td_losses else None,
            },
            "Take3 (KL)": {"objective": "KL", "final_loss": float(kl_losses[-1]) if kl_losses else None},
        },
        "methods": methods,
        "summary": {
            "proposal_ground_truth_log_Z": selected_target["log_Z"],
            "methods": summary,
        },
    }
    save_json(output, args.output_path)
    print(f"\nSaved JSON results to {args.output_path}")
    print(f"Saved summary figure to {args.plot_base}.png/.pdf")


if __name__ == "__main__":
    main()
