import csv
import glob
import os
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import logsumexp

from updated_code.alanine import fit_tica, tica_transform


TORSION_ORDER = ("phi", "psi", "omega", "chi1", "chi2", "chi3", "chi4")
MAX_RESIDUES = 4
MAX_TORSIONS = MAX_RESIDUES * len(TORSION_ORDER)
AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def _safe_str(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _safe_str(value.item())
    return str(value)


def _looks_like_peptide_sequence(sequence):
    seq = _safe_str(sequence).upper()
    return len(seq) == MAX_RESIDUES and all(ch in AA_ALPHABET for ch in seq)


def flatten_torsion_angles(angles):
    angles = np.asarray(angles, dtype=np.float64)
    if angles.ndim == 2:
        return angles
    if angles.ndim == 3:
        return angles.reshape(angles.shape[0], -1)
    raise ValueError("Expected angles with shape (n_frames, n_torsions) or (n_frames, n_residues, n_torsions_per_residue).")


def flatten_torsion_mask(mask):
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=np.float64)
    if mask.ndim == 1:
        return mask
    if mask.ndim == 2:
        return mask.reshape(-1)
    raise ValueError("Expected torsion mask with shape (n_torsions,) or (n_residues, n_torsions_per_residue).")


def torsion_angles_to_features(angles, mask=None):
    angles = flatten_torsion_angles(angles)
    cos = np.cos(angles)
    sin = np.sin(angles)
    features = np.stack([cos, sin], axis=-1).reshape(angles.shape[0], -1)
    if mask is not None:
        mask_flat = flatten_torsion_mask(mask)
        pair_mask = np.repeat(mask_flat, 2)
        features = features * pair_mask[None, :]
    return features


def torsion_features_to_angles(features, n_torsions):
    features = np.asarray(features, dtype=np.float64)
    reshaped = features.reshape(features.shape[:-1] + (n_torsions, 2))
    return np.arctan2(reshaped[..., 1], reshaped[..., 0])


def load_tetrapeptide_npz(path):
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    name = os.path.splitext(os.path.basename(path))[0]

    if "angles" in keys:
        angles = flatten_torsion_angles(data["angles"])
        if angles.shape[-1] > MAX_TORSIONS:
            raise ValueError(f"{path} has {angles.shape[-1]} torsions; expected at most {MAX_TORSIONS}.")
        mask = flatten_torsion_mask(data["mask"]) if "mask" in keys else None
        features = torsion_angles_to_features(angles, mask=mask)
        source = "angles"
    elif "features" in keys:
        features = np.asarray(data["features"], dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"{path} features must have shape (n_frames, dim).")
        n_torsions = features.shape[-1] // 2
        angles = torsion_features_to_angles(features, n_torsions=n_torsions)
        mask = flatten_torsion_mask(data["mask"]) if "mask" in keys else None
        source = "features"
    else:
        raise ValueError(f"{path} must contain either 'angles' or 'features'.")

    sequence = None
    if "sequence" in keys:
        sequence = _safe_str(data["sequence"])
    elif "name" in keys:
        sequence = _safe_str(data["name"])

    canonical_sequence = name if _looks_like_peptide_sequence(name) else (sequence if sequence is not None else name)

    return {
        "name": name,
        "sequence": canonical_sequence,
        "stored_sequence": sequence,
        "angles": angles,
        "features": features,
        "mask": mask,
        "source": source,
        "path": path,
    }


def read_split_names(split_csv):
    names = []
    with open(split_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            token = row[0].strip()
            if not token or token.lower() in {"name", "sequence", "split"}:
                continue
            names.append(token)
    return names


def load_tetrapeptide_directory(data_dir, split_csv=None, max_peptides=None, seed=0):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if split_csv is not None:
        allowed = set(read_split_names(split_csv))
        filtered = []
        for path in files:
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem in allowed:
                filtered.append(path)
        files = filtered
    if max_peptides is not None and len(files) > max_peptides:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(files), size=max_peptides, replace=False)
        files = [files[i] for i in sorted(idx)]
    return [load_tetrapeptide_npz(path) for path in files]


def run_kmeans(data, k, seed=0, n_init=5, max_iter=100):
    data = np.asarray(data, dtype=np.float64)
    rng = np.random.default_rng(seed)
    best = None
    best_inertia = None
    n = len(data)
    if n < k:
        raise ValueError(f"Need at least {k} samples for k-means, got {n}.")

    for _ in range(n_init):
        centers = data[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(max_iter):
            d2 = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
            new_labels = np.argmin(d2, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                members = data[labels == j]
                if len(members) == 0:
                    centers[j] = data[rng.integers(0, n)]
                else:
                    centers[j] = np.mean(members, axis=0)
        inertia = float(np.sum((data - centers[labels]) ** 2))
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best = (centers.copy(), labels.copy())
    return {"centers": best[0], "labels": best[1], "inertia": best_inertia}


def estimate_markov_chain(labels, n_states, lag_frames):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    for idx in range(len(labels) - lag_frames):
        counts[labels[idx], labels[idx + lag_frames]] += 1.0

    T = np.zeros_like(counts)
    for i in range(n_states):
        row_sum = np.sum(counts[i])
        if row_sum <= 0.0:
            T[i, i] = 1.0
        else:
            T[i] = counts[i] / row_sum

    pi = np.bincount(labels, minlength=n_states).astype(np.float64)
    pi = pi / np.clip(np.sum(pi), 1e-12, None)
    return {"counts": counts, "transition_matrix": T, "stationary": pi}


def precompute_matrix_powers(T, max_power):
    powers = [np.eye(T.shape[0], dtype=np.float64)]
    for _ in range(max_power):
        powers.append(powers[-1] @ T)
    return powers


def fit_reference_state_model(features, lag_frames, n_states=10, n_tica_dims=5, seed=0):
    tica_model = fit_tica(features[None, :, :], lag=lag_frames, n_components=max(2, n_tica_dims))
    coords = tica_transform(features, tica_model)[:, :n_tica_dims]
    km = run_kmeans(coords, k=n_states, seed=seed)
    labels = km["labels"]
    msm = estimate_markov_chain(labels, n_states=n_states, lag_frames=lag_frames)
    state_centers = []
    for j in range(n_states):
        members = coords[labels == j]
        if len(members) == 0:
            state_centers.append(km["centers"][j])
        else:
            state_centers.append(np.mean(members, axis=0))
    state_centers = np.stack(state_centers, axis=0)
    return {
        "tica_model": tica_model,
        "tica_coords": coords,
        "labels": labels,
        "transition_matrix": msm["transition_matrix"],
        "stationary": msm["stationary"],
        "counts": msm["counts"],
        "state_centers": state_centers,
        "n_states": n_states,
        "lag_frames": lag_frames,
    }


def choose_start_end_states(T, pi, path_length, mode="hard"):
    powers = precompute_matrix_powers(T, path_length - 1)
    flux = pi[:, None] * T
    candidates = []
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if i == j:
                continue
            if flux[i, j] <= 0.0:
                continue
            if powers[path_length - 1][i, j] <= 0.0:
                continue
            candidates.append((flux[i, j], i, j))
    if not candidates:
        raise ValueError("Could not find a non-trivial start/end state pair with non-zero conditioned path probability.")
    candidates.sort(key=lambda x: x[0])

    mode = str(mode).lower()
    if mode == "hard":
        idx = 0
    elif mode == "medium":
        idx = len(candidates) // 2
    elif mode == "easy":
        idx = len(candidates) - 1
    else:
        raise ValueError(f"Unsupported state-pair mode: {mode}")

    chosen_flux, start_state, end_state = candidates[idx]
    meta = {
        "mode": mode,
        "candidate_rank": int(idx),
        "n_candidates": int(len(candidates)),
        "chosen_flux": float(chosen_flux),
        "min_flux": float(candidates[0][0]),
        "median_flux": float(candidates[len(candidates) // 2][0]),
        "max_flux": float(candidates[-1][0]),
    }
    return start_state, end_state, powers, meta


def sample_conditioned_state_paths(T, start_state, end_state, path_length, n_paths, rng):
    powers = precompute_matrix_powers(T, path_length - 1)
    n_states = T.shape[0]
    paths = np.full((n_paths, path_length), end_state, dtype=np.int64)
    valid = np.ones(n_paths, dtype=bool)

    for p in range(n_paths):
        cur = start_state
        paths[p, 0] = start_state
        for idx in range(path_length - 2):
            remain = (path_length - 1) - idx
            denom = powers[remain][cur, end_state]
            numer = T[cur] * powers[remain - 1][:, end_state]
            if denom <= 0.0 or np.sum(numer) <= 0.0:
                valid[p] = False
                break
            probs = numer / np.sum(numer)
            cur = int(rng.choice(n_states, p=probs))
            paths[p, idx + 1] = cur
        paths[p, -1] = end_state
    return paths, valid, powers


def conditioned_path_log_prob(path, T, powers, start_state, end_state):
    path = np.asarray(path, dtype=np.int64)
    if len(path) == 0:
        return -np.inf
    if int(path[0]) != int(start_state) or int(path[-1]) != int(end_state):
        return -np.inf
    log_prob = 0.0
    for idx in range(len(path) - 1):
        cur = int(path[idx])
        nxt = int(path[idx + 1])
        remain = (len(path) - 1) - idx
        denom = powers[remain][cur, end_state]
        numer = T[cur, nxt] * powers[remain - 1][nxt, end_state]
        if denom <= 0.0 or numer <= 0.0:
            return -np.inf
        log_prob += np.log(numer) - np.log(denom)
    return float(log_prob)


def state_visit_distribution(paths, n_states):
    counts = np.zeros(n_states, dtype=np.float64)
    for path in np.asarray(paths, dtype=np.int64):
        counts += np.bincount(path, minlength=n_states)
    probs = counts / np.clip(np.sum(counts), 1e-12, None)
    probs = np.clip(probs, 1e-12, None)
    return probs / np.sum(probs)


def compute_tps_metrics(reference_paths, sample_paths, T_ref, powers, start_state, end_state):
    p_ref = state_visit_distribution(reference_paths, T_ref.shape[0])
    p_sample = state_visit_distribution(sample_paths, T_ref.shape[0])
    js_distance = float(jensenshannon(p_ref, p_sample, base=2.0))

    log_probs = np.array(
        [conditioned_path_log_prob(path, T_ref, powers, start_state, end_state) for path in sample_paths],
        dtype=np.float64,
    )
    valid = np.isfinite(log_probs)
    probs = np.zeros_like(log_probs)
    probs[valid] = np.exp(log_probs[valid])
    return {
        "js_distance": js_distance,
        "js_divergence": js_distance ** 2,
        "valid_path_rate": float(np.mean(valid)),
        "average_path_probability": float(np.mean(probs)),
        "average_log_path_probability": float(np.mean(log_probs[valid])) if np.any(valid) else None,
    }


def nearest_state_labels(coords, state_centers):
    coords = np.asarray(coords, dtype=np.float64)
    d2 = np.sum((coords[:, None, :] - state_centers[None, :, :]) ** 2, axis=-1)
    return np.argmin(d2, axis=1)


def discretize_feature_paths(paths, tica_model, state_centers, lag_frames, path_length):
    paths = np.asarray(paths, dtype=np.float64)
    frame_idx = np.arange(path_length, dtype=np.int64) * int(lag_frames)
    if frame_idx[-1] >= paths.shape[1]:
        raise ValueError(
            f"Need at least {frame_idx[-1] + 1} frames to discretize with "
            f"path_length={path_length}, lag_frames={lag_frames}; got {paths.shape[1]}."
        )
    flat = paths[:, frame_idx, :].reshape(-1, paths.shape[-1])
    coords = tica_transform(flat, tica_model)[:, :state_centers.shape[1]]
    labels = nearest_state_labels(coords, state_centers)
    return labels.reshape(paths.shape[0], path_length)


def select_start_state_windows(features, labels, start_state, horizon_frames, stride=1, max_windows=None, rng=None):
    windows = []
    starts = []
    for idx in range(0, len(features) - horizon_frames, stride):
        if labels[idx] != start_state:
            continue
        windows.append(features[idx:idx + horizon_frames + 1])
        starts.append(idx)
    if not windows:
        return np.zeros((0, horizon_frames + 1, features.shape[-1]), dtype=np.float64), np.zeros(0, dtype=np.int64)
    windows = np.stack(windows, axis=0)
    starts = np.asarray(starts, dtype=np.int64)
    if max_windows is not None and len(windows) > max_windows:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(len(windows), size=max_windows, replace=False)
        windows = windows[idx]
        starts = starts[idx]
    return windows, starts


def sample_start_features(features, labels, start_state, n_samples, rng):
    pool = np.where(labels == start_state)[0]
    if len(pool) == 0:
        raise ValueError("No frames assigned to the chosen start state.")
    chosen = rng.choice(pool, size=n_samples, replace=True)
    return features[chosen], chosen


def endpoint_log_reward_factory(tica_model, state_centers, end_state, horizon_frames, lam, mode="soft_state"):
    state_centers = np.asarray(state_centers, dtype=np.float64)
    end_state = int(end_state)

    def endpoint_log_G(features, t):
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features[None, :]
        if t != horizon_frames:
            return np.zeros(features.shape[0], dtype=np.float64)
        coords = tica_transform(features, tica_model)[:, :state_centers.shape[1]]
        diff = coords[:, None, :] - state_centers[None, :, :]
        d2 = np.sum(diff ** 2, axis=-1)
        logits = -float(lam) * d2
        if mode == "soft_state":
            return logits[:, end_state] - logsumexp(logits, axis=1)
        if mode == "center_distance":
            return logits[:, end_state]
        raise ValueError(f"Unsupported reward mode: {mode!r}")

    return endpoint_log_G


def average_state_path(reference_labels, lag_frames, path_length):
    frame_idx = np.linspace(0, len(reference_labels) - 1, path_length, dtype=np.int64)
    return reference_labels[frame_idx]


def load_mdgen_like_path_file(path):
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    if "state_paths" in keys:
        return {"state_paths": np.asarray(data["state_paths"], dtype=np.int64), "source": "state_paths"}
    if "features" in keys:
        return {"features": np.asarray(data["features"], dtype=np.float64), "source": "features"}
    if "angles" in keys:
        angles = flatten_torsion_angles(data["angles"])
        return {"features": torsion_angles_to_features(angles), "source": "angles"}
    raise ValueError(f"{path} must contain state_paths, features, or angles.")


def load_manifest_rows(manifest_csv):
    rows = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            rows.append(row)
    return rows


def _load_mdtraj():
    try:
        import mdtraj as md
    except ImportError:
        return None
    return md


def compute_padded_torsions_from_trajectory(topology_path, trajectory_path, stride=1):
    md = _load_mdtraj()
    if md is None:
        raise ImportError(
            "mdtraj is required to preprocess raw tetrapeptide trajectories. "
            "Install it in the current Python environment, e.g. "
            "`conda install -c conda-forge mdtraj` or `pip install mdtraj`, "
            "then rerun prep_tetrapeptide_data.py."
        )

    traj = md.load(trajectory_path, top=topology_path, stride=stride)
    n_frames = traj.n_frames
    n_res = traj.topology.n_residues
    if n_res != MAX_RESIDUES:
        raise ValueError(f"Expected tetrapeptide with {MAX_RESIDUES} residues, found {n_res} in {trajectory_path}.")

    angles = np.zeros((n_frames, MAX_RESIDUES, len(TORSION_ORDER)), dtype=np.float64)
    mask = np.zeros((MAX_RESIDUES, len(TORSION_ORDER)), dtype=np.float64)

    torsion_fns = {
        "phi": md.compute_phi,
        "psi": md.compute_psi,
        "omega": md.compute_omega,
        "chi1": md.compute_chi1,
        "chi2": md.compute_chi2,
        "chi3": md.compute_chi3,
        "chi4": md.compute_chi4,
    }

    for torsion_name, fn in torsion_fns.items():
        atom_idx, values = fn(traj)
        torsion_slot = TORSION_ORDER.index(torsion_name)
        for j in range(atom_idx.shape[0]):
            residue_index = traj.topology.atom(int(atom_idx[j, 1])).residue.index
            angles[:, residue_index, torsion_slot] = values[:, j]
            mask[residue_index, torsion_slot] = 1.0

    sequence = "".join(THREE_TO_ONE.get(res.name.upper(), res.name[:1].upper()) for res in traj.topology.residues)
    residue_names = np.array([res.name for res in traj.topology.residues], dtype=object)
    return {
        "angles": angles,
        "mask": mask,
        "sequence": sequence,
        "residue_names": residue_names,
        "n_frames": n_frames,
    }
