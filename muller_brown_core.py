import numpy as np


MB_A = np.array([-200.0, -100.0, -170.0, 15.0], dtype=np.float64)
MB_a = np.array([-1.0, -1.0, -6.5, 0.7], dtype=np.float64)
MB_b = np.array([0.0, 0.0, 11.0, 0.6], dtype=np.float64)
MB_c = np.array([-10.0, -10.0, -6.5, 0.7], dtype=np.float64)
MB_X = np.array([1.0, 0.0, -0.5, -1.0], dtype=np.float64)
MB_Y = np.array([0.0, 0.5, 1.5, 1.0], dtype=np.float64)
MB_SCALE = 0.05

MINIMUM_A = np.array([-0.558, 1.442], dtype=np.float64)
MINIMUM_B = np.array([0.623, 0.028], dtype=np.float64)
MINIMUM_C = np.array([-0.050, 0.467], dtype=np.float64)
SADDLE_BC = np.array([0.212, 0.293], dtype=np.float64)


def U(x):
    x = np.asarray(x, dtype=np.float64)
    scalar = x.ndim == 1
    leading_shape = x.shape[:-1]
    points = x.reshape(-1, 2)
    dx = points[:, :1] - MB_X.reshape(1, -1)
    dy = points[:, 1:] - MB_Y.reshape(1, -1)
    expo = MB_a * dx ** 2 + MB_b * dx * dy + MB_c * dy ** 2
    values = MB_SCALE * np.sum(MB_A * np.exp(np.clip(expo, -80.0, 80.0)), axis=1)
    return float(values[0]) if scalar else values.reshape(leading_shape)


def grad_U(x):
    x = np.asarray(x, dtype=np.float64)
    scalar = x.ndim == 1
    leading_shape = x.shape[:-1]
    points = x.reshape(-1, 2)
    dx = points[:, :1] - MB_X.reshape(1, -1)
    dy = points[:, 1:] - MB_Y.reshape(1, -1)
    expo = MB_a * dx ** 2 + MB_b * dx * dy + MB_c * dy ** 2
    exp_terms = np.exp(np.clip(expo, -80.0, 80.0))
    gx = np.sum(MB_A * exp_terms * (2.0 * MB_a * dx + MB_b * dy), axis=1)
    gy = np.sum(MB_A * exp_terms * (MB_b * dx + 2.0 * MB_c * dy), axis=1)
    grad = MB_SCALE * np.stack([gx, gy], axis=-1)
    return grad[0] if scalar else grad.reshape(leading_shape + (2,))


def langevin_step(x, dt, grad_clip=20.0):
    x = np.asarray(x, dtype=np.float64)
    scalar = x.ndim == 1
    if scalar:
        x = x[None, :]
    grad = np.clip(grad_U(x), -grad_clip, grad_clip)
    out = x - float(dt) * grad + np.sqrt(2.0 * float(dt)) * np.random.randn(*x.shape)
    return out[0] if scalar else out


def sample_trajectory(x_0, T, dt=0.01):
    traj = np.zeros((int(T) + 1, 2), dtype=np.float64)
    traj[0] = np.asarray(x_0, dtype=np.float64)
    for t in range(int(T)):
        traj[t + 1] = langevin_step(traj[t], dt)
    return traj


def sample_adjacent_pairs(n_pairs, dt_traj=0.01):
    n_pairs = int(n_pairs)
    n_walkers = min(512, max(64, n_pairs // 20))
    starts = np.stack([MINIMUM_A, MINIMUM_B, MINIMUM_C], axis=0)
    walkers = starts[np.random.randint(0, len(starts), size=n_walkers)] + 0.05 * np.random.randn(n_walkers, 2)
    for _ in range(2500):
        walkers = langevin_step(walkers, 0.003)

    xt = np.zeros((n_pairs, 2), dtype=np.float64)
    xtp1 = np.zeros((n_pairs, 2), dtype=np.float64)
    filled = 0
    while filled < n_pairs:
        batch = min(n_walkers, n_pairs - filled)
        current = walkers[:batch].copy()
        nxt = langevin_step(current, dt_traj)
        xt[filled:filled + batch] = current
        xtp1[filled:filled + batch] = nxt
        walkers[:batch] = nxt
        for _ in range(8):
            walkers = langevin_step(walkers, 0.003)
        filled += batch
    return xt, xtp1


class BarrierCrossingReward:
    def __init__(self, T, target, saddle, lam_endpoint=5.0, lam_saddle=3.0, T_mid=None):
        self.T = int(T)
        self.target = np.asarray(target, dtype=np.float64)
        self.saddle = np.asarray(saddle, dtype=np.float64)
        self.lam_endpoint = float(lam_endpoint)
        self.lam_saddle = float(lam_saddle)
        self.T_mid = self.T // 2 if T_mid is None else int(T_mid)

    def log_G(self, x, t):
        x = np.asarray(x, dtype=np.float64).reshape(-1, 2)
        out = np.zeros(len(x), dtype=np.float64)
        if int(t) == self.T_mid:
            out += -self.lam_saddle * np.sum((x - self.saddle) ** 2, axis=1)
        if int(t) == self.T:
            out += -self.lam_endpoint * np.sum((x - self.target) ** 2, axis=1)
        return out


class RouteOccupancyReward:
    def __init__(
        self,
        T,
        target,
        route_center,
        lam_endpoint=6.0,
        lam_route=4.5,
        route_sigma=0.18,
        route_start=None,
        route_end=None,
        endpoint_radius=0.28,
        route_threshold=0.5,
    ):
        self.T = int(T)
        self.target = np.asarray(target, dtype=np.float64)
        self.route_center = np.asarray(route_center, dtype=np.float64)
        self.lam_endpoint = float(lam_endpoint)
        self.lam_route = float(lam_route)
        self.route_sigma = float(route_sigma)
        self.route_start = self.T // 4 if route_start is None else int(route_start)
        self.route_end = (3 * self.T) // 4 if route_end is None else int(route_end)
        self.route_times = tuple(range(self.route_start, self.route_end + 1))
        self.endpoint_radius = float(endpoint_radius)
        self.route_threshold = float(route_threshold)

    def _route_kernel(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1, 2)
        sqdist = np.sum((x - self.route_center) ** 2, axis=1)
        return np.exp(-0.5 * sqdist / max(self.route_sigma ** 2, 1e-12))

    def log_G(self, x, t):
        x = np.asarray(x, dtype=np.float64).reshape(-1, 2)
        out = np.zeros(len(x), dtype=np.float64)
        if int(t) in self.route_times:
            out += self.lam_route * self._route_kernel(x) / max(len(self.route_times), 1)
        if int(t) == self.T:
            out += -self.lam_endpoint * np.sum((x - self.target) ** 2, axis=1)
        return out

    def route_score(self, trajs):
        trajs = np.asarray(trajs, dtype=np.float64)
        window = trajs[:, self.route_times, :].reshape(-1, 2)
        score = self._route_kernel(window).reshape(trajs.shape[0], len(self.route_times))
        return np.mean(score, axis=1)

    def endpoint_distance(self, trajs):
        trajs = np.asarray(trajs, dtype=np.float64)
        return np.linalg.norm(trajs[:, -1, :] - self.target, axis=1)

    def endpoint_success(self, trajs):
        return self.endpoint_distance(trajs) <= self.endpoint_radius

    def joint_success(self, trajs):
        return self.endpoint_success(trajs) & (self.route_score(trajs) >= self.route_threshold)
