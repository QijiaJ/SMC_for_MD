import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict


class PositiveTwistNet(nn.Module):
    """
    Positive twist network.

    The paper's objectives regress psi_t itself in linear space. We therefore
    keep the network output positive and only take logs at SMC evaluation time.
    """

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
        # Keep psi nonnegative; logging/clipping happens in the loss/wrapper.
        # Avoid a large additive floor here because Exp 3 targets can be far
        # below 1e-8.
        raw = self.net(h).squeeze(-1)
        raw = torch.clamp(torch.nan_to_num(raw, nan=0.0, posinf=40.0, neginf=-40.0), -40.0, 40.0)
        return F.softplus(raw) + 1e-12


def _prepare_trajectories(trajectories, dim, T):
    trajectories = np.asarray(trajectories)
    if dim == 1:
        return trajectories.reshape(-1, T + 1, 1)
    return trajectories.reshape(-1, T + 1, dim)


def _clone_model_state(model):
    return OrderedDict((key, value.detach().cpu().clone()) for key, value in model.state_dict().items())


def _restore_model_state(model, state):
    model.load_state_dict(state)


def _model_parameters_are_finite(model):
    return all(torch.isfinite(param).all().item() for param in model.parameters())


def _reset_optimizer_state(optimizer):
    optimizer.state.clear()


def _halve_optimizer_lr(optimizer, min_lr=1e-5):
    new_lr = None
    for group in optimizer.param_groups:
        group["lr"] = max(float(group["lr"]) * 0.5, float(min_lr))
        new_lr = group["lr"]
    return new_lr


def _twist_regression_loss(psi_pred, target, loss_space="linear", eps=1e-45):
    dtype_eps = torch.finfo(psi_pred.dtype).tiny
    safe_eps = max(float(eps), float(dtype_eps))
    target = torch.clamp(torch.nan_to_num(target, nan=safe_eps, posinf=1e6, neginf=safe_eps), min=safe_eps, max=1e6)
    psi_pred = torch.clamp(torch.nan_to_num(psi_pred, nan=safe_eps, posinf=1e6, neginf=safe_eps), min=safe_eps, max=1e6)
    if loss_space == "linear":
        return torch.mean((psi_pred - target) ** 2)
    if loss_space == "log":
        pred_log = torch.log(torch.clamp(psi_pred, min=safe_eps))
        target_log = torch.log(target)
        return torch.mean((pred_log - target_log) ** 2)
    raise ValueError(f"Unsupported loss_space={loss_space!r}. Use 'linear' or 'log'.")


def train_positive_twist_mc(trajectories, targets, dim, T,
                            n_epochs=2000, batch_size=256, lr=1e-3,
                            hidden_dim=128, n_layers=3,
                            device="cpu", verbose=True,
                            loss_space="linear"):
    """
    MC regression on future-value targets in linear space.

    `targets[i, t]` should equal the realized future reward from time t.
    """
    traj = _prepare_trajectories(trajectories, dim, T)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, T + 1)
    n_traj = traj.shape[0]

    model = PositiveTwistNet(dim, T, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    last_good_state = _clone_model_state(model)

    traj_t = torch.tensor(traj, dtype=torch.float32, device=device)
    target_t = torch.tensor(targets, dtype=torch.float32, device=device)
    losses = []

    for epoch in range(n_epochs):
        idx = torch.randint(0, n_traj, (batch_size,), device=device)
        t_batch = torch.randint(0, T + 1, (batch_size,), device=device)
        x_batch = traj_t[idx, t_batch]
        y_batch = target_t[idx, t_batch]

        psi_pred = model(x_batch, t_batch)
        loss = _twist_regression_loss(psi_pred, y_batch, loss_space=loss_space)
        if not torch.isfinite(loss):
            restored_lr = _halve_optimizer_lr(optimizer)
            _restore_model_state(model, last_good_state)
            _reset_optimizer_state(optimizer)
            if verbose:
                print(f"    [MC twist] epoch {epoch+1}/{n_epochs}: non-finite loss, restoring last good state and lowering lr to {restored_lr:.2e}")
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            restored_lr = _halve_optimizer_lr(optimizer)
            _restore_model_state(model, last_good_state)
            _reset_optimizer_state(optimizer)
            optimizer.zero_grad(set_to_none=True)
            if verbose:
                print(f"    [MC twist] epoch {epoch+1}/{n_epochs}: non-finite gradient, restoring last good state and lowering lr to {restored_lr:.2e}")
            continue
        optimizer.step()
        if not _model_parameters_are_finite(model):
            restored_lr = _halve_optimizer_lr(optimizer)
            _restore_model_state(model, last_good_state)
            _reset_optimizer_state(optimizer)
            if verbose:
                print(f"    [MC twist] epoch {epoch+1}/{n_epochs}: non-finite parameters, restoring last good state and lowering lr to {restored_lr:.2e}")
            continue
        last_good_state = _clone_model_state(model)
        losses.append(float(loss.item()))

        if verbose and (epoch + 1) % 500 == 0:
            print(f"    [MC twist] epoch {epoch+1}/{n_epochs}: loss={loss.item():.6f}")

    model.eval()
    return model, losses


def train_positive_twist_td(trajectories, dim, T, log_G_fn,
                            terminal_targets=None,
                            n_epochs=2000, batch_size=256, lr=1e-3,
                            hidden_dim=128, n_layers=3,
                            device="cpu", verbose=True,
                            loss_space="linear"):
    """
    TD training for future-only twists:
      psi_t(x_t) = E[G_{t+1}(x_{t+1}) * psi_{t+1}(x_{t+1}) | x_t],
      psi_T = terminal_target.
    """
    traj = _prepare_trajectories(trajectories, dim, T)
    n_traj = traj.shape[0]
    if terminal_targets is None:
        terminal_targets = np.ones(n_traj, dtype=np.float64)
    terminal_targets = np.asarray(terminal_targets, dtype=np.float64).reshape(-1)

    model = PositiveTwistNet(dim, T, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    last_good_state = _clone_model_state(model)

    traj_t = torch.tensor(traj, dtype=torch.float32, device=device)
    terminal_t = torch.tensor(terminal_targets, dtype=torch.float32, device=device)
    losses = []

    for epoch in range(n_epochs):
        idx = torch.randint(0, n_traj, (batch_size,), device=device)

        x_T = traj_t[idx, T]
        t_T = torch.full((batch_size,), T, device=device, dtype=torch.long)
        psi_T = model(x_T, t_T)
        loss_terminal = _twist_regression_loss(psi_T, terminal_t[idx], loss_space=loss_space)

        t_batch = torch.randint(0, T, (batch_size,), device=device)
        x_t = traj_t[idx, t_batch]
        x_tp1 = traj_t[idx, t_batch + 1]

        g_np = []
        x_tp1_np = x_tp1.detach().cpu().numpy()
        t_np = t_batch.detach().cpu().numpy()
        for i in range(batch_size):
            x_single = np.atleast_2d(x_tp1_np[i])
            g_np.append(float(np.exp(np.clip(log_G_fn(x_single, int(t_np[i]) + 1)[0], -700.0, 80.0))))
        g_tp1 = torch.tensor(g_np, dtype=torch.float32, device=device)

        psi_t = model(x_t, t_batch)
        with torch.no_grad():
            psi_tp1 = model(x_tp1, t_batch + 1)
            target = g_tp1 * psi_tp1
        loss_td = _twist_regression_loss(psi_t, target, loss_space=loss_space)

        loss = loss_terminal + loss_td
        if not torch.isfinite(loss):
            restored_lr = _halve_optimizer_lr(optimizer)
            _restore_model_state(model, last_good_state)
            _reset_optimizer_state(optimizer)
            if verbose:
                print(f"    [TD twist] epoch {epoch+1}/{n_epochs}: non-finite loss, restoring last good state and lowering lr to {restored_lr:.2e}")
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            restored_lr = _halve_optimizer_lr(optimizer)
            _restore_model_state(model, last_good_state)
            _reset_optimizer_state(optimizer)
            optimizer.zero_grad(set_to_none=True)
            if verbose:
                print(f"    [TD twist] epoch {epoch+1}/{n_epochs}: non-finite gradient, restoring last good state and lowering lr to {restored_lr:.2e}")
            continue
        optimizer.step()
        if not _model_parameters_are_finite(model):
            restored_lr = _halve_optimizer_lr(optimizer)
            _restore_model_state(model, last_good_state)
            _reset_optimizer_state(optimizer)
            if verbose:
                print(f"    [TD twist] epoch {epoch+1}/{n_epochs}: non-finite parameters, restoring last good state and lowering lr to {restored_lr:.2e}")
            continue
        last_good_state = _clone_model_state(model)
        losses.append(float(loss.item()))

        if verbose and (epoch + 1) % 500 == 0:
            print(f"    [TD twist] epoch {epoch+1}/{n_epochs}: loss={loss.item():.6f}")

    model.eval()
    return model, losses


def train_positive_twist_kl(trajectories, targets, dim, T,
                            n_epochs=2000, batch_size=256, lr=1e-3,
                            hidden_dim=128, n_layers=3,
                            device="cpu", verbose=True,
                            eps=1e-12):
    """
    Self-normalized KL twist learning from Lemma 3.

    For each sampled pair (x_t, r_future), optimize

      log E_p[psi_t(x_t)] - E_sigma[log psi_t(x_t)]

    where sigma(x_t) ∝ p_ref(x_t) * psi_t^*(x_t). With samples from p_ref only,
    the sigma expectation is estimated by importance weighting with the realized
    future reward target.
    """
    traj = _prepare_trajectories(trajectories, dim, T)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, T + 1)
    n_traj = traj.shape[0]

    model = PositiveTwistNet(dim, T, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    last_good_state = _clone_model_state(model)

    traj_t = torch.tensor(traj, dtype=torch.float32, device=device)
    target_t = torch.tensor(targets, dtype=torch.float32, device=device)
    losses = []

    for epoch in range(n_epochs):
        idx = torch.randint(0, n_traj, (batch_size,), device=device)
        t_batch = torch.randint(0, T + 1, (batch_size,), device=device)
        x_batch = traj_t[idx, t_batch]
        y_batch = torch.clamp(
            torch.nan_to_num(target_t[idx, t_batch], nan=eps, posinf=1e6, neginf=eps),
            min=eps,
            max=1e6,
        )
        psi_pred = torch.clamp(
            torch.nan_to_num(model(x_batch, t_batch), nan=eps, posinf=1e6, neginf=eps),
            min=eps,
            max=1e6,
        )
        log_y = torch.log(y_batch)
        normalized_targets = torch.softmax(log_y, dim=0)
        log_psi = torch.log(psi_pred)

        # Lemma 3 objective:
        #   log E_p[psi_t] - E_sigma[log psi_t]
        log_mean_psi = torch.logsumexp(log_psi, dim=0) - np.log(float(log_psi.numel()))
        loss = log_mean_psi - torch.sum(normalized_targets * log_psi)
        if not torch.isfinite(loss):
            restored_lr = _halve_optimizer_lr(optimizer)
            _restore_model_state(model, last_good_state)
            _reset_optimizer_state(optimizer)
            if verbose:
                print(f"    [KL twist] epoch {epoch+1}/{n_epochs}: non-finite loss, restoring last good state and lowering lr to {restored_lr:.2e}")
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            restored_lr = _halve_optimizer_lr(optimizer)
            _restore_model_state(model, last_good_state)
            _reset_optimizer_state(optimizer)
            optimizer.zero_grad(set_to_none=True)
            if verbose:
                print(f"    [KL twist] epoch {epoch+1}/{n_epochs}: non-finite gradient, restoring last good state and lowering lr to {restored_lr:.2e}")
            continue
        optimizer.step()
        if not _model_parameters_are_finite(model):
            restored_lr = _halve_optimizer_lr(optimizer)
            _restore_model_state(model, last_good_state)
            _reset_optimizer_state(optimizer)
            if verbose:
                print(f"    [KL twist] epoch {epoch+1}/{n_epochs}: non-finite parameters, restoring last good state and lowering lr to {restored_lr:.2e}")
            continue
        last_good_state = _clone_model_state(model)
        losses.append(float(loss.item()))

        if verbose and (epoch + 1) % 500 == 0:
            print(f"    [KL twist] epoch {epoch+1}/{n_epochs}: loss={loss.item():.6f}")

    model.eval()
    return model, losses


class PositiveNNTwist:
    """Wrapper that exposes log psi while storing a positive-valued model."""

    def __init__(self, model, dim, device="cpu"):
        self.model = model
        self.dim = dim
        self.device = device

    def value(self, x, t):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64)).reshape(-1, self.dim)
        n_pts = len(x)
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=self.device)
            tt = torch.full((n_pts,), int(t), dtype=torch.long, device=self.device)
            out = self.model(xt, tt).cpu().numpy()
        return out.squeeze() if n_pts > 1 else float(out.squeeze())

    def __call__(self, x, t):
        vals = np.asarray(self.value(x, t), dtype=np.float64)
        logged = np.log(np.clip(vals, 1e-300, None))
        if np.ndim(logged) == 0:
            return float(logged)
        return logged
