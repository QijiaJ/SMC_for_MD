import json
import os
import numpy as np
from scipy.special import logsumexp


def systematic_resample(log_weights, K=None):
    """Systematic resampling from log-weights."""
    if K is None:
        K = len(log_weights)
    log_w_norm = np.asarray(log_weights, dtype=np.float64) - logsumexp(log_weights)
    w = np.exp(log_w_norm)
    positions = (np.arange(K) + np.random.uniform()) / K
    cumsum = np.cumsum(w)
    indices = np.searchsorted(cumsum, positions)
    return np.clip(indices, 0, len(w) - 1)


def resample_multinomial(log_weights, n_samples):
    """Multinomial resampling from log-weights."""
    log_w_norm = np.asarray(log_weights, dtype=np.float64) - logsumexp(log_weights)
    w = np.exp(log_w_norm)
    if np.any(~np.isfinite(w)) or np.sum(w) <= 0.0:
        return np.random.choice(len(log_weights), size=n_samples)
    w = w / np.sum(w)
    return np.random.choice(len(log_weights), size=n_samples, p=w)


def sample_row_indices(log_weights, n_draws=1):
    """
    Row-wise categorical sampling from a 2D array of log-weights.

    Args:
        log_weights: array of shape (N, M)
        n_draws: number of draws per row

    Returns:
        indices: shape (N,) if n_draws == 1 else (N, n_draws)
        log_norm: logsumexp per row, shape (N,)
    """
    log_weights = np.asarray(log_weights, dtype=np.float64)
    log_norm = logsumexp(log_weights, axis=1, keepdims=True)
    weights = np.exp(log_weights - log_norm)
    invalid = (~np.isfinite(weights)).any(axis=1) | (np.sum(weights, axis=1) < 1e-300)
    if np.any(invalid):
        weights[invalid] = 1.0 / weights.shape[1]
    cum = np.cumsum(weights, axis=1)
    u = np.random.rand(weights.shape[0], n_draws)
    indices = np.sum(u[..., None] > cum[:, None, :], axis=-1)
    indices = np.minimum(indices, weights.shape[1] - 1)
    if n_draws == 1:
        indices = indices.reshape(-1)
    return indices, log_norm.reshape(-1)


def compute_ess(log_weights):
    """ESS from log-weights."""
    log_w_norm = np.asarray(log_weights, dtype=np.float64) - logsumexp(log_weights)
    w = np.exp(log_w_norm)
    return 1.0 / np.sum(w ** 2)


def build_future_reward_targets(trajectories, T, log_G_fn):
    """
    Build positive future-value targets:
      psi_t(x_t) = E[prod_{s=t+1}^T G_s(x_s) | x_t]
    using Monte Carlo trajectories from the reference process.

    For a realized trajectory, the regression target at time t is
      prod_{s=t+1}^T G_s(x_s),
    and psi_T = 1.
    """
    trajectories = np.asarray(trajectories)
    n_traj = trajectories.shape[0]
    targets = np.ones((n_traj, T + 1), dtype=np.float64)
    running_log = np.zeros(n_traj, dtype=np.float64)

    for t in range(T - 1, -1, -1):
        x_next = trajectories[:, t + 1]
        inc = np.asarray(log_G_fn(x_next, t + 1), dtype=np.float64).reshape(n_traj)
        running_log += inc
        targets[:, t] = np.exp(np.clip(running_log, -700.0, 80.0))

    return targets


def make_future_trajectory_reward(log_G_fn, T):
    """
    Convert an incremental log-potential G_t into a future-only reward callback
    compatible with tweedie_rollout_twist(..., trajectory_reward=True).
    """
    def trajectory_reward(trajectory, t_start):
        total = 0.0
        for offset in range(1, len(trajectory)):
            t_actual = t_start + offset
            if t_actual <= T:
                x_t = np.atleast_2d(np.asarray(trajectory[offset], dtype=np.float64))
                total += float(np.asarray(log_G_fn(x_t, t_actual)).reshape(-1)[0])
        return total

    return trajectory_reward


def mean_std(values):
    """Return mean/std summary for a sequence of numeric values."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(data, output_path):
    """Save JSON, creating parent directories if needed."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(data), f, indent=2, sort_keys=True)
    return output_path
