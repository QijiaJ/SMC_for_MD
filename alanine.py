import os
import numpy as np
from scipy.spatial.distance import jensenshannon


DEFAULT_C7EQ = np.deg2rad(np.array([-80.0, 75.0], dtype=np.float64))
DEFAULT_C7AX = np.deg2rad(np.array([70.0, -60.0], dtype=np.float64))
DEFAULT_ALPHA_R = np.deg2rad(np.array([-60.0, -40.0], dtype=np.float64))


def _load_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def wrap_angles(angles):
    angles = np.asarray(angles, dtype=np.float64)
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def angular_difference(a, b):
    return wrap_angles(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))


def squared_angular_distance(angles, center):
    diff = angular_difference(angles, center)
    return np.sum(diff ** 2, axis=-1)


def basin_mask(angles, center, radius):
    return np.sqrt(squared_angular_distance(angles, center)) <= float(radius)


def angles_to_features(angles):
    angles = np.asarray(angles, dtype=np.float64)
    phi = angles[..., 0]
    psi = angles[..., 1]
    return np.stack(
        [np.cos(phi), np.sin(phi), np.cos(psi), np.sin(psi)],
        axis=-1,
    )


def normalize_torsion_features(features, eps=1e-8):
    features = np.asarray(features, dtype=np.float64).copy()
    for offset in (0, 2):
        pair = features[..., offset:offset + 2]
        norm = np.linalg.norm(pair, axis=-1, keepdims=True)
        features[..., offset:offset + 2] = pair / np.clip(norm, eps, None)
    return features


def features_to_angles(features):
    features = normalize_torsion_features(features)
    phi = np.arctan2(features[..., 1], features[..., 0])
    psi = np.arctan2(features[..., 3], features[..., 2])
    return wrap_angles(np.stack([phi, psi], axis=-1))


def circular_mean(angles):
    angles = wrap_angles(np.asarray(angles, dtype=np.float64))
    return np.arctan2(np.mean(np.sin(angles), axis=0), np.mean(np.cos(angles), axis=0))


def _dihedral_from_points(p0, p1, p2, p3, eps=1e-12):
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.clip(np.linalg.norm(b1, axis=-1, keepdims=True), eps, None)

    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)
    return np.arctan2(y, x)


def compute_phi_psi_from_coordinates(positions, phi_indices, psi_indices):
    positions = np.asarray(positions, dtype=np.float64)
    phi_idx = np.asarray(phi_indices, dtype=np.int64).reshape(4)
    psi_idx = np.asarray(psi_indices, dtype=np.int64).reshape(4)
    phi = _dihedral_from_points(
        positions[..., phi_idx[0], :],
        positions[..., phi_idx[1], :],
        positions[..., phi_idx[2], :],
        positions[..., phi_idx[3], :],
    )
    psi = _dihedral_from_points(
        positions[..., psi_idx[0], :],
        positions[..., psi_idx[1], :],
        positions[..., psi_idx[2], :],
        positions[..., psi_idx[3], :],
    )
    return wrap_angles(np.stack([phi, psi], axis=-1))


def _ensure_traj_shape(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[-1] == 2:
        return arr[None, ...]
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError("Expected trajectory array with shape (n_traj, n_frames, 2).")
    return arr


def _maybe_convert_degrees(angles):
    angles = np.asarray(angles, dtype=np.float64)
    if np.nanmax(np.abs(angles)) > 3.5:
        return np.deg2rad(angles), True
    return angles, False


def _load_generic_npz_series(npz_data):
    keys = sorted(npz_data.files)
    if not keys:
        return None, None
    arrays = [np.asarray(npz_data[k], dtype=np.float64) for k in keys]
    shapes = {arr.shape for arr in arrays}
    if len(shapes) != 1:
        return None, None

    first = arrays[0]
    if first.ndim == 2 and first.shape[-1] in (2, 4):
        stacked = np.stack(arrays, axis=0)
        source = "generic_npz_angles" if first.shape[-1] == 2 else "generic_npz_features"
        return stacked, source
    if first.ndim == 3 and first.shape[-1] in (2, 4):
        stacked = np.concatenate(arrays, axis=0)
        source = "generic_npz_angles_batched" if first.shape[-1] == 2 else "generic_npz_features_batched"
        return stacked, source
    return None, None


def load_alanine_trajectories(data_path, phi_indices=None, psi_indices=None):
    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".npz":
        data = np.load(data_path, allow_pickle=True)
        keys = set(data.files)
        if "angles" in keys:
            angles = _ensure_traj_shape(data["angles"])
            source = "angles"
        elif {"phi", "psi"}.issubset(keys):
            phi = np.asarray(data["phi"], dtype=np.float64)
            psi = np.asarray(data["psi"], dtype=np.float64)
            if phi.ndim == 1:
                phi = phi[None, :]
                psi = psi[None, :]
            angles = np.stack([phi, psi], axis=-1)
            source = "phi_psi"
        elif "features" in keys:
            features = np.asarray(data["features"], dtype=np.float64)
            if features.ndim == 2 and features.shape[-1] == 4:
                features = features[None, ...]
            if features.ndim != 3 or features.shape[-1] != 4:
                raise ValueError("Expected feature array with shape (n_traj, n_frames, 4).")
            angles = features_to_angles(features)
            source = "features"
        elif "positions" in keys:
            positions = np.asarray(data["positions"], dtype=np.float64)
            if positions.ndim == 3:
                positions = positions[None, ...]
            if positions.ndim != 4:
                raise ValueError("Expected positions with shape (n_traj, n_frames, n_atoms, 3).")
            if phi_indices is None:
                phi_indices = data["phi_indices"] if "phi_indices" in keys else None
            if psi_indices is None:
                psi_indices = data["psi_indices"] if "psi_indices" in keys else None
            if phi_indices is None or psi_indices is None:
                raise ValueError("Need phi_indices and psi_indices to compute torsions from positions.")
            angles = compute_phi_psi_from_coordinates(positions, phi_indices, psi_indices)
            source = "positions"
        else:
            generic, source = _load_generic_npz_series(data)
            if generic is None:
                raise ValueError(
                    "Could not find supported alanine trajectory arrays. "
                    "Expected one of: angles, phi/psi, features, positions, "
                    "or a generic mdshare-style npz with arr_* trajectories."
                )
            if generic.shape[-1] == 2:
                angles = _ensure_traj_shape(generic)
            else:
                if generic.ndim == 2:
                    generic = generic[None, ...]
                angles = features_to_angles(generic)
    elif ext == ".npy":
        arr = np.load(data_path, allow_pickle=True)
        if arr.ndim >= 2 and arr.shape[-1] == 2:
            angles = _ensure_traj_shape(arr)
            source = "angles_npy"
        elif arr.ndim >= 2 and arr.shape[-1] == 4:
            features = arr if arr.ndim == 3 else arr[None, ...]
            angles = features_to_angles(features)
            source = "features_npy"
        else:
            raise ValueError("Unsupported .npy shape. Expected (..., 2) angles or (..., 4) features.")
    else:
        raise ValueError("Unsupported file extension. Use .npz or .npy for precomputed trajectories.")

    angles, converted_from_degrees = _maybe_convert_degrees(angles)
    angles = wrap_angles(angles)
    metadata = {
        "source": source,
        "converted_from_degrees": bool(converted_from_degrees),
        "n_trajectories": int(angles.shape[0]),
        "n_frames": int(angles.shape[1]),
    }
    return angles, metadata


def subsample_rows(arr, max_rows, rng):
    arr = np.asarray(arr)
    if max_rows is None or len(arr) <= max_rows:
        return arr
    idx = rng.choice(len(arr), size=max_rows, replace=False)
    return arr[idx]


def make_windows(angles, horizon, stride=1):
    angles = _ensure_traj_shape(angles)
    windows = []
    indices = []
    for traj_idx, traj in enumerate(angles):
        n_frames = len(traj)
        if n_frames < horizon + 1:
            continue
        for start in range(0, n_frames - horizon, stride):
            windows.append(traj[start:start + horizon + 1])
            indices.append((traj_idx, start))
    if not windows:
        return np.zeros((0, horizon + 1, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.int64)
    return np.stack(windows, axis=0), np.asarray(indices, dtype=np.int64)


def split_train_test_trajectories(angles, train_fraction=0.8, seed=0):
    angles = _ensure_traj_shape(angles)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(angles))
    n_train = max(1, int(round(train_fraction * len(angles))))
    if len(angles) > 1:
        n_train = min(n_train, len(angles) - 1)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    return angles[train_idx], angles[test_idx], train_idx, test_idx


def split_train_test_temporal(angles, train_fraction=0.8, gap=0):
    angles = _ensure_traj_shape(angles)
    n_traj, n_frames, _ = angles.shape
    cut = int(round(train_fraction * n_frames))
    cut = min(max(cut, 1), n_frames - 1)
    gap = max(int(gap), 0)

    train_end = max(1, cut - gap)
    test_start = min(n_frames - 1, cut + gap)
    if train_end >= test_start:
        train_end = max(1, cut)
        test_start = min(n_frames - 1, cut)
    if train_end >= test_start:
        raise ValueError("Temporal split gap is too large for the available trajectory length.")

    train_angles = angles[:, :train_end, :]
    test_angles = angles[:, test_start:, :]
    split_info = {
        "mode": "temporal_with_gap",
        "n_trajectories": int(n_traj),
        "n_frames_total": int(n_frames),
        "train_frame_range": [0, int(train_end)],
        "test_frame_range": [int(test_start), int(n_frames)],
        "gap_frames": int(test_start - train_end),
    }
    return train_angles, test_angles, split_info


def fit_tica(feature_trajectories, lag, n_components=2, ridge=1e-6):
    feature_trajectories = np.asarray(feature_trajectories, dtype=np.float64)
    if feature_trajectories.ndim != 3:
        raise ValueError("Expected feature trajectories with shape (n_traj, n_frames, dim).")

    all_frames = feature_trajectories.reshape(-1, feature_trajectories.shape[-1])
    mean = np.mean(all_frames, axis=0)

    xs = []
    ys = []
    for traj in feature_trajectories:
        if len(traj) <= lag:
            continue
        xs.append(traj[:-lag] - mean)
        ys.append(traj[lag:] - mean)
    if not xs:
        raise ValueError("Not enough frames to fit TICA at the requested lag.")

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    c0 = 0.5 * ((x.T @ x) + (y.T @ y)) / len(x)
    c_tau = 0.5 * ((x.T @ y) + (y.T @ x)) / len(x)
    c0 = c0 + ridge * np.eye(c0.shape[0])

    mat = np.linalg.solve(c0, c_tau)
    eigvals, eigvecs = np.linalg.eig(mat)
    order = np.argsort(np.real(eigvals))[::-1]
    eigvals = np.real(eigvals[order])
    eigvecs = np.real(eigvecs[:, order])

    comps = []
    for i in range(min(n_components, eigvecs.shape[1])):
        vec = eigvecs[:, i]
        norm = np.sqrt(np.clip(vec.T @ c0 @ vec, 1e-12, None))
        comps.append(vec / norm)
    components = np.stack(comps, axis=1)
    return {
        "mean": mean,
        "components": components,
        "eigenvalues": eigvals[:components.shape[1]],
        "lag": int(lag),
    }


def tica_transform(features, tica_model):
    features = np.asarray(features, dtype=np.float64)
    centered = features - tica_model["mean"]
    return centered @ tica_model["components"]


def _safe_distribution(hist):
    hist = np.asarray(hist, dtype=np.float64)
    total = np.sum(hist)
    if total <= 0.0:
        return np.full(hist.size, 1.0 / hist.size)
    probs = hist.reshape(-1) / total
    probs = np.clip(probs, 1e-12, None)
    probs = probs / np.sum(probs)
    return probs


def histogram_js_distance_2d(samples_a, samples_b, bins, value_range):
    samples_a = np.asarray(samples_a, dtype=np.float64)
    samples_b = np.asarray(samples_b, dtype=np.float64)
    hist_a, x_edges, y_edges = np.histogram2d(
        samples_a[:, 0], samples_a[:, 1], bins=bins, range=value_range
    )
    hist_b, _, _ = np.histogram2d(
        samples_b[:, 0], samples_b[:, 1], bins=[x_edges, y_edges]
    )
    js_distance = float(jensenshannon(_safe_distribution(hist_a), _safe_distribution(hist_b), base=2.0))
    return {
        "js_distance": js_distance,
        "js_divergence": js_distance ** 2,
        "hist_a": hist_a,
        "hist_b": hist_b,
        "x_edges": x_edges,
        "y_edges": y_edges,
    }


def compute_free_energy_surface(samples, bins, value_range):
    samples = np.asarray(samples, dtype=np.float64)
    hist, x_edges, y_edges = np.histogram2d(
        samples[:, 0],
        samples[:, 1],
        bins=bins,
        range=value_range,
    )
    prob = hist / np.clip(np.sum(hist), 1e-12, None)
    free_energy = -np.log(np.clip(prob, 1e-12, None))
    free_energy = free_energy - np.min(free_energy[np.isfinite(free_energy)])
    return {
        "hist": hist,
        "probability": prob,
        "free_energy": free_energy,
        "x_edges": x_edges,
        "y_edges": y_edges,
    }


def plot_histogram_comparison(ref_stats, sample_stats, output_base, title, axis_labels):
    plt = _load_pyplot()
    if plt is None:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    vmax = max(
        np.max(ref_stats["probability"]),
        np.max(sample_stats["probability"]),
        1e-12,
    )
    for ax, stats, panel_title in [
        (axes[0], ref_stats, "Reference"),
        (axes[1], sample_stats, "Generated"),
    ]:
        im = ax.imshow(
            stats["probability"].T,
            origin="lower",
            aspect="auto",
            extent=[
                stats["x_edges"][0],
                stats["x_edges"][-1],
                stats["y_edges"][0],
                stats["y_edges"][-1],
            ],
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(panel_title)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    plt.savefig(output_base + ".png", dpi=160, bbox_inches="tight")
    plt.savefig(output_base + ".pdf", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_free_energy_comparison(ref_stats, sample_stats, output_base, title, axis_labels):
    plt = _load_pyplot()
    if plt is None:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    vmax = max(
        np.nanmax(ref_stats["free_energy"]),
        np.nanmax(sample_stats["free_energy"]),
    )
    for ax, stats, panel_title in [
        (axes[0], ref_stats, "Reference"),
        (axes[1], sample_stats, "Generated"),
    ]:
        im = ax.imshow(
            stats["free_energy"].T,
            origin="lower",
            aspect="auto",
            extent=[
                stats["x_edges"][0],
                stats["x_edges"][-1],
                stats["y_edges"][0],
                stats["y_edges"][-1],
            ],
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(panel_title)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    plt.savefig(output_base + ".png", dpi=160, bbox_inches="tight")
    plt.savefig(output_base + ".pdf", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True
