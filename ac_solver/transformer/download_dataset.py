"""
Download the AC-Solver dataset from HuggingFace and reconstruct numpy shards.

When you clone the GitHub repo the `dataset/transformer_ds/shards/` directory is
absent (the raw numpy files are too large for git).  This script pulls the
dataset from HuggingFace Hub and writes it back to the exact shard format that
`train_lm.py` expects:

    dataset/transformer_ds/shards/
        shard_0000_presentations.npy   (153600, 256) int8
        shard_0000_metadata.npy        (153600, 3)   int32
        shard_0001_presentations.npy
        shard_0001_metadata.npy
        ...
        shard_0011_presentations.npy
        shard_0011_metadata.npy

Memory usage: only one shard buffer (~39 MB) is kept in RAM at a time.

Usage:
    # Default: reconstruct shards from mhieuuu/ac-solver-dataset
    python -m ac_solver.transformer.download_dataset

    # Custom repo or output directory
    python -m ac_solver.transformer.download_dataset \\
        --repo-id mhieuuu/ac-solver-dataset \\
        --output-dir dataset/transformer_ds

Requirements:
    pip install datasets huggingface_hub
"""

import argparse
import json
import os

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_THIS_DIR)), "dataset", "transformer_ds"
)
_DEFAULT_REPO_ID = "mhieuuu/ac-solver-dataset"

# Must match the original generation config
_SHARD_SIZE = 153_600   # rows per shard
_N_SHARDS = 12
_LMAX = 128
_TOTAL_ROWS = _SHARD_SIZE * _N_SHARDS   # 1,827,840


def main(args):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The `datasets` package is required.\n"
            "Install it with:  pip install datasets huggingface_hub"
        )

    shards_dir = os.path.join(args.output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    print(f"Repo          : {args.repo_id}")
    print(f"Output shards : {shards_dir}")
    print(f"Expected rows : {_TOTAL_ROWS:,}  ({_N_SHARDS} shards × {_SHARD_SIZE:,} rows)")

    # Check for already-complete shards so we can resume
    existing = sorted(
        f for f in os.listdir(shards_dir) if f.endswith("_presentations.npy")
    )
    completed_shards = len(existing)
    rows_to_skip = completed_shards * _SHARD_SIZE

    if completed_shards == _N_SHARDS:
        print(f"\nAll {_N_SHARDS} shards already present. Nothing to do.")
        print("Delete shards/ and re-run to force a fresh download.")
        return

    if completed_shards > 0:
        print(f"\nResuming: {completed_shards} shards already done, "
              f"skipping first {rows_to_skip:,} rows.")

    # Stream the dataset to keep peak RAM at one shard buffer (~39 MB)
    print("\nStreaming dataset from HuggingFace Hub...")
    ds = load_dataset(args.repo_id, split="train", streaming=True)

    shard_idx = completed_shards
    pres_buf = []   # list of lists, each length 256
    meta_buf = []   # list of [pres_idx, phase, chain]
    rows_seen = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(ds, total=_TOTAL_ROWS, initial=rows_to_skip,
                        desc="Downloading", unit="rows")
    except ImportError:
        iterator = ds

    for row in iterator:
        rows_seen += 1

        # Skip rows already saved in previous run
        if rows_seen <= rows_to_skip:
            continue

        pres_buf.append(row["presentation"])
        meta_buf.append([row["pres_idx"], row["phase"], row["chain"]])

        if len(pres_buf) == _SHARD_SIZE:
            _save_shard(shards_dir, shard_idx, pres_buf, meta_buf)
            shard_idx += 1
            pres_buf = []
            meta_buf = []

    # Save any remaining rows as a partial last shard (shouldn't happen with
    # a complete dataset, but handles edge cases gracefully)
    if pres_buf:
        print(f"  Warning: {len(pres_buf)} leftover rows — saving as partial shard.")
        _save_shard(shards_dir, shard_idx, pres_buf, meta_buf)

    # Write config.json so data_generator.py / train_lm.py can read metadata
    _write_config(args.output_dir)

    print(f"\nDone. {shard_idx} shards written to {shards_dir}")
    print("You can now run:")
    print("  python -m ac_solver.transformer.train_lm")


def _save_shard(shards_dir, shard_idx, pres_buf, meta_buf):
    pres_arr = np.array(pres_buf, dtype=np.int8)    # (N, 256)
    meta_arr = np.array(meta_buf, dtype=np.int32)   # (N, 3)

    pres_path = os.path.join(shards_dir, f"shard_{shard_idx:04d}_presentations.npy")
    meta_path = os.path.join(shards_dir, f"shard_{shard_idx:04d}_metadata.npy")

    np.save(pres_path, pres_arr)
    np.save(meta_path, meta_arr)
    print(f"  Saved shard {shard_idx:04d}  ({len(pres_buf):,} rows)  → {pres_path}")


def _write_config(output_dir):
    config = {
        "n_presentations": 1190,
        "n_phases": 128,
        "n_chains": 12,
        "n_moves": 1000,
        "lmax": _LMAX,
        "seed": 42,
        "shard_size": _SHARD_SIZE,
        "n_shards": _N_SHARDS,
        "total_generated": _TOTAL_ROWS,
    }
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Written config → {config_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download AC-Solver dataset from HuggingFace and reconstruct shards"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=_DEFAULT_REPO_ID,
        help=f"HuggingFace dataset repo ID (default: {_DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Root output directory — shards/ will be created inside it "
             f"(default: {_DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
