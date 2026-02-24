"""
Upload the AC-Solver transformer dataset to HuggingFace Hub.

Streams all 12 shards lazily (never loads the full ~490 MB at once) and pushes
as Parquet files via Dataset.from_generator() + push_to_hub().

Schema per row:
    presentation  : Sequence(int8, length=256)  — two relators padded to lmax=128
    pres_idx      : int32  — which of the 1190 Miller-Schupp seed presentations
    phase         : int32  — MCMC phase index (0–127)
    chain         : int32  — Markov chain index (0–11)
    gs_solved     : bool   — whether the seed presentation is GS-solved

Usage:
    # Set HF_TOKEN env var or pass --token
    python -m ac_solver.transformer.upload_dataset \\
        --dataset-dir dataset/transformer_ds \\
        --repo-id <HF_USERNAME>/ac-solver-dataset

    # Dry run (validates generator, prints first 3 rows, no upload)
    python -m ac_solver.transformer.upload_dataset \\
        --dataset-dir dataset/transformer_ds \\
        --repo-id <HF_USERNAME>/ac-solver-dataset \\
        --dry-run
"""

import argparse
import json
import os

import numpy as np
from datasets import Dataset, Features, Sequence, Value

from ac_solver.transformer.prepare_dataset import load_gs_solved_indices

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_THIS_DIR)), "dataset", "transformer_ds"
)

FEATURES = Features(
    {
        "presentation": Sequence(feature=Value("int8"), length=256),
        "pres_idx": Value("int32"),
        "phase": Value("int32"),
        "chain": Value("int32"),
        "gs_solved": Value("bool"),
    }
)


def make_generator(shards_dir: str, gs_solved_set: set):
    """
    Yield one dict per presentation, streaming shards one at a time.

    Parameters
    ----------
    shards_dir  : path to the directory containing shard_NNNN_*.npy files
    gs_solved_set : set of int — pres_idx values that are GS-solved
    """
    shard_idx = 0
    while True:
        pres_path = os.path.join(
            shards_dir, f"shard_{shard_idx:04d}_presentations.npy"
        )
        meta_path = os.path.join(
            shards_dir, f"shard_{shard_idx:04d}_metadata.npy"
        )
        if not os.path.exists(pres_path):
            break  # no more shards

        presentations = np.load(pres_path)  # (N, 256) int8
        metadata = np.load(meta_path)       # (N, 3)  int32

        for row in range(len(presentations)):
            pres_idx = int(metadata[row, 0])
            yield {
                "presentation": presentations[row].tolist(),
                "pres_idx": pres_idx,
                "phase": int(metadata[row, 1]),
                "chain": int(metadata[row, 2]),
                "gs_solved": pres_idx in gs_solved_set,
            }

        shard_idx += 1


def main(args):
    dataset_dir = args.dataset_dir
    shards_dir = os.path.join(dataset_dir, "shards")
    config_path = os.path.join(dataset_dir, "config.json")

    # Validate inputs
    if not os.path.isdir(shards_dir):
        raise FileNotFoundError(
            f"Shards directory not found: {shards_dir}\n"
            "Run data_generator.py first."
        )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found: {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    n_shards = cfg["n_shards"]
    total_rows = cfg["total_generated"]
    lmax = cfg["lmax"]

    print(f"Dataset dir   : {dataset_dir}")
    print(f"Shards        : {n_shards}")
    print(f"Total rows    : {total_rows:,}")
    print(f"lmax          : {lmax}")
    print(f"Target repo   : {args.repo_id}")

    # Load GS-solved label set
    print("\nLoading GS-solved indices...")
    gs_solved_set = load_gs_solved_indices()
    print(f"  GS-solved seeds: {len(gs_solved_set)} / {cfg['n_presentations']}")

    # --- DRY RUN ---
    if args.dry_run:
        print("\n[DRY RUN] Sampling first 3 rows from generator...")
        gen = make_generator(shards_dir, gs_solved_set)
        for i, row in enumerate(gen):
            print(f"  Row {i}: pres_idx={row['pres_idx']}  phase={row['phase']}  "
                  f"chain={row['chain']}  gs_solved={row['gs_solved']}  "
                  f"presentation[:6]={row['presentation'][:6]}")
            if i >= 2:
                break
        print("[DRY RUN] Generator OK. Re-run without --dry-run to upload.")
        return

    # --- BUILD DATASET ---
    print("\nBuilding HuggingFace Dataset from generator (streaming shards)...")
    dataset = Dataset.from_generator(
        make_generator,
        gen_kwargs={"shards_dir": shards_dir, "gs_solved_set": gs_solved_set},
        features=FEATURES,
    )
    print(f"Dataset built: {dataset}")

    # --- UPLOAD ---
    token = args.token or os.environ.get("HF_TOKEN")

    print(f"\nPushing to Hub: {args.repo_id} ...")
    dataset.push_to_hub(
        repo_id=args.repo_id,
        split="train",
        private=args.private,
        token=token,
        max_shard_size="500MB",
        commit_message="Upload AC-Solver transformer dataset (1.8M presentations)",
    )
    print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{args.repo_id}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload AC-Solver transformer dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=_DEFAULT_DATASET_DIR,
        help="Directory containing config.json and shards/ subdirectory "
             f"(default: {_DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo id, e.g. <HF_USERNAME>/ac-solver-dataset",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (default: reads HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create a private dataset repository (default: public)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate generator and print sample rows without uploading",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
