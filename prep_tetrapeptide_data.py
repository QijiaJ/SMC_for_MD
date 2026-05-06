"""
Preprocess raw tetrapeptide trajectories into padded torsion-angle npz files.

Input manifest CSV columns:
  name,topology,trajectory

Outputs one npz per row containing:
  - angles: (n_frames, 4, 7)
  - mask: (4, 7)
  - sequence
  - residue_names
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from updated_code.common import save_json
from updated_code.tetrapeptide_tps import compute_padded_torsions_from_trajectory, load_manifest_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess tetrapeptide trajectories into padded torsion npz files.")
    parser.add_argument("--manifest-csv", required=True, help="CSV with columns: name,topology,trajectory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--metadata-json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_manifest_rows(args.manifest_csv)
    metadata = []

    for row in rows:
        name = row["name"]
        out_path = os.path.join(args.output_dir, f"{name}.npz")
        packed = compute_padded_torsions_from_trajectory(
            topology_path=row["topology"],
            trajectory_path=row["trajectory"],
            stride=args.stride,
        )
        import numpy as np

        np.savez(
            out_path,
            angles=packed["angles"],
            mask=packed["mask"],
            sequence=packed["sequence"],
            residue_names=packed["residue_names"],
            source_topology=row["topology"],
            source_trajectory=row["trajectory"],
        )
        metadata.append(
            {
                "name": name,
                "sequence": packed["sequence"],
                "n_frames": int(packed["n_frames"]),
                "output_path": out_path,
            }
        )
        print(f"Wrote {out_path}")

    metadata_path = args.metadata_json or os.path.join(args.output_dir, "metadata.json")
    save_json({"items": metadata}, metadata_path)
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
