import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ScoreNet(nn.Module):
    """Time-conditioned MLP score model with forward signature model(x, s)."""

    def __init__(self, input_dim, output_dim, hidden_dim=128, n_layers=3):
        super().__init__()
        hidden_dim = int(hidden_dim)
        n_layers = max(int(n_layers), 1)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        layers = []
        in_dim = int(input_dim) + hidden_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x, s):
        if s.ndim == 0:
            s = s.reshape(1)
        s = s.reshape(-1, 1).to(dtype=x.dtype, device=x.device)
        temb = self.time_embed(s)
        if temb.shape[0] == 1 and x.shape[0] > 1:
            temb = temb.expand(x.shape[0], -1)
        return self.net(torch.cat([x, temb], dim=-1))


def _clone_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _restore_state(model, state):
    model.load_state_dict(state)


def _model_finite(model):
    return all(torch.isfinite(param).all().item() for param in model.parameters())


def _ou_noised_batch(clean, s):
    alpha = torch.exp(-0.5 * s).unsqueeze(-1)
    sigma = torch.sqrt(torch.clamp(1.0 - torch.exp(-s), min=1e-10)).unsqueeze(-1)
    eps = torch.randn_like(clean)
    xs = alpha * clean + sigma * eps
    target = -eps / sigma
    return xs, torch.clamp(torch.nan_to_num(target, nan=0.0, posinf=25.0, neginf=-25.0), -25.0, 25.0)


def _train_dsm_model(clean_data, context_data, dim, input_dim, n_epochs, batch_size,
                     lr, hidden_dim, n_layers, device, verbose, label):
    clean_np = np.nan_to_num(np.asarray(clean_data, dtype=np.float64).reshape(-1, dim), nan=0.0, posinf=5.0, neginf=-5.0)
    context_np = None
    if context_data is not None:
        context_np = np.nan_to_num(np.asarray(context_data, dtype=np.float64).reshape(len(clean_np), -1), nan=0.0, posinf=5.0, neginf=-5.0)

    clean = torch.tensor(clean_np, dtype=torch.float32, device=device)
    context = None if context_np is None else torch.tensor(context_np, dtype=torch.float32, device=device)
    model = ScoreNet(input_dim=input_dim, output_dim=dim, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    last_good = _clone_state(model)
    losses = []
    n = clean.shape[0]

    for epoch in range(int(n_epochs)):
        idx = torch.randint(0, n, (int(batch_size),), device=device)
        s = torch.rand(int(batch_size), device=device) * 2.0 + 1e-4
        xs, target = _ou_noised_batch(clean[idx], s)
        if context is None:
            inp = xs
        else:
            inp = torch.cat([xs, context[idx]], dim=-1)
        inp = torch.clamp(torch.nan_to_num(inp, nan=0.0, posinf=6.0, neginf=-6.0), -6.0, 6.0)
        pred = torch.clamp(torch.nan_to_num(model(inp, s), nan=0.0, posinf=25.0, neginf=-25.0), -25.0, 25.0)
        loss = torch.mean((pred - target) ** 2)

        if not torch.isfinite(loss):
            _restore_state(model, last_good)
            optimizer.state.clear()
            if verbose:
                print(f"    [{label}] epoch {epoch + 1}/{n_epochs}: non-finite loss; restored last good state")
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            _restore_state(model, last_good)
            optimizer.state.clear()
            optimizer.zero_grad(set_to_none=True)
            if verbose:
                print(f"    [{label}] epoch {epoch + 1}/{n_epochs}: non-finite gradient; restored last good state")
            continue
        optimizer.step()
        if not _model_finite(model):
            _restore_state(model, last_good)
            optimizer.state.clear()
            if verbose:
                print(f"    [{label}] epoch {epoch + 1}/{n_epochs}: non-finite parameters; restored last good state")
            continue
        last_good = _clone_state(model)
        losses.append(float(loss.item()))

        if verbose and (epoch + 1) % 500 == 0:
            print(f"    [{label}] epoch {epoch + 1}/{n_epochs}: loss={loss.item():.6f}")

    model.eval()
    return model, losses


def train_marginal_score(samples, dim, n_epochs=3000, batch_size=256, lr=1e-3,
                         hidden_dim=128, n_layers=3, device="cpu", verbose=True):
    return _train_dsm_model(
        clean_data=samples,
        context_data=None,
        dim=dim,
        input_dim=dim,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        device=device,
        verbose=verbose,
        label="Marginal score",
    )


def train_conditional_score(x_t, x_tp1, dim, n_epochs=3000, batch_size=256, lr=1e-3,
                            hidden_dim=128, n_layers=3, device="cpu", verbose=True):
    """Train o_s(x_{t+1}^s, x_t): next-state score conditioned on current state.

    The public API takes transition pairs in their natural order,
    (current state, next state). The DSM clean variable must be x_{t+1},
    while x_t is provided as conditioning context.
    """
    x_t = np.asarray(x_t, dtype=np.float64).reshape(-1, dim)
    x_tp1 = np.asarray(x_tp1, dtype=np.float64).reshape(-1, dim)
    if len(x_t) != len(x_tp1):
        raise ValueError("x_t and x_tp1 must have the same number of rows.")
    return _train_dsm_model(
        clean_data=x_tp1,
        context_data=x_t,
        dim=dim,
        input_dim=2 * dim,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        device=device,
        verbose=verbose,
        label="Conditional score",
    )


class NNMarginalScore:
    def __init__(self, model, dim, device="cpu", clip=25.0):
        self.model = model
        self.dim = int(dim)
        self.device = device
        self.clip = float(clip)

    def score(self, x, s):
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1, self.dim)
        with torch.no_grad():
            xt = torch.tensor(x_arr, dtype=torch.float32, device=self.device)
            st = torch.full((len(x_arr),), float(s), dtype=torch.float32, device=self.device)
            out = self.model(xt, st).cpu().numpy()
        out = np.clip(np.nan_to_num(out, nan=0.0, posinf=self.clip, neginf=-self.clip), -self.clip, self.clip)
        return out if np.asarray(x).ndim > 1 else out.reshape(self.dim)

    def boltzmann_score(self, x):
        return self.score(x, 1e-4)

    def __call__(self, x, s):
        return self.score(x, s)


class NNConditionalScore:
    def __init__(self, model, dim, device="cpu", clip=25.0):
        self.model = model
        self.dim = int(dim)
        self.device = device
        self.clip = float(clip)

    def score(self, x, s, x_cond):
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1, self.dim)
        cond = np.asarray(x_cond, dtype=np.float64)
        if cond.ndim == 1:
            cond_arr = cond.reshape(1, self.dim)
        else:
            cond_arr = cond.reshape(-1, self.dim)
        if len(cond_arr) == 1 and len(x_arr) > 1:
            cond_arr = np.tile(cond_arr, (len(x_arr), 1))
        elif len(cond_arr) != len(x_arr):
            raise ValueError("x and x_cond must have matching batch sizes.")
        with torch.no_grad():
            inp = torch.tensor(np.concatenate([x_arr, cond_arr], axis=-1), dtype=torch.float32, device=self.device)
            st = torch.full((len(x_arr),), float(s), dtype=torch.float32, device=self.device)
            out = self.model(inp, st).cpu().numpy()
        out = np.clip(np.nan_to_num(out, nan=0.0, posinf=self.clip, neginf=-self.clip), -self.clip, self.clip)
        return out if np.asarray(x).ndim > 1 else out.reshape(self.dim)

    def __call__(self, x, s, x_cond):
        return self.score(x, s, x_cond)
