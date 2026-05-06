import numpy as np


def U(x):
    x = np.asarray(x, dtype=np.float64)
    return (x ** 2 - 1.0) ** 2


def grad_U(x):
    x = np.asarray(x, dtype=np.float64)
    return 4.0 * x * (x ** 2 - 1.0)


def langevin_step(x, dt=0.05, n_samples=None, grad_clip=10.0):
    x_input = np.asarray(x, dtype=np.float64)
    scalar_input = x_input.ndim == 0
    x_arr = x_input.reshape(-1)

    if n_samples is not None:
        n_samples = int(n_samples)
        if x_arr.size == 1:
            x_arr = np.full(n_samples, float(x_arr[0]), dtype=np.float64)
        elif x_arr.size != n_samples:
            raise ValueError("n_samples must match the number of provided 1D states.")

    grad = np.clip(grad_U(x_arr), -grad_clip, grad_clip)
    out = x_arr - float(dt) * grad + np.sqrt(2.0 * float(dt)) * np.random.randn(*x_arr.shape)
    if scalar_input and n_samples is None:
        return float(out[0])
    return out


def sample_equilibrium(n_samples, dt=0.01, n_burnin=10000, thin=10):
    n_samples = int(n_samples)
    x = float(np.random.choice([-1.0, 1.0]) + 0.1 * np.random.randn())
    for _ in range(int(n_burnin)):
        x = langevin_step(x, dt)

    samples = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        for _ in range(int(thin)):
            x = langevin_step(x, dt)
        samples[i] = x
    return samples


def sample_trajectory(x_0, T, dt=0.05):
    traj = np.zeros(int(T) + 1, dtype=np.float64)
    traj[0] = float(x_0)
    for t in range(int(T)):
        traj[t + 1] = langevin_step(traj[t], dt)
    return traj


def log_reward(x, x_target, lam):
    x = np.asarray(x, dtype=np.float64)
    return -float(lam) * (x - float(x_target)) ** 2
