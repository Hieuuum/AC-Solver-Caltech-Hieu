"""
Dataset generation for the transformer language model (Algorithm 6 from Appendix D).

Generates ~1.8 million balanced presentations by applying random AC' moves to the
1190 Miller-Schupp seed presentations through 128 phases of gradually increasing
maximum relator length.

Usage:
    python -m ac_solver.transformer.data_generator [options]

Example:
    python -m ac_solver.transformer.data_generator --n-workers 8 --seed 42
"""

import argparse
import json
import math
import os
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from ac_solver.agents.utils import load_initial_states_from_text_file
from ac_solver.envs.ac_moves import ACMove


def resize_presentation_np(presentation, old_max, new_max):
    """Resize a numpy int8 presentation array to a new max_relator_length.

    This is a numpy-native alternative to change_max_relator_length_of_presentation
    (which only works with list inputs due to an assertion in convert_relators_to_presentation).

    Parameters:
        presentation: numpy array of dtype int8, length 2 * old_max
        old_max: current max_relator_length
        new_max: desired max_relator_length (must be >= actual word lengths)

    Returns:
        numpy array of dtype int8, length 2 * new_max
    """
    r0_len = np.count_nonzero(presentation[:old_max])
    r1_len = np.count_nonzero(presentation[old_max : 2 * old_max])

    assert new_max >= r0_len, (
        f"new_max ({new_max}) must be >= r0 length ({r0_len})"
    )
    assert new_max >= r1_len, (
        f"new_max ({new_max}) must be >= r1 length ({r1_len})"
    )

    new_pres = np.zeros(2 * new_max, dtype=np.int8)
    new_pres[:r0_len] = presentation[:r0_len]
    new_pres[new_max : new_max + r1_len] = presentation[old_max : old_max + r1_len]
    return new_pres


def get_word_lengths(presentation, max_relator_length):
    """Extract word lengths [len_r0, len_r1] from a presentation array.

    Parameters:
        presentation: numpy array of dtype int8
        max_relator_length: the max relator length used in the array layout

    Returns:
        list of two ints
    """
    r0_len = int(np.count_nonzero(presentation[:max_relator_length]))
    r1_len = int(np.count_nonzero(presentation[max_relator_length : 2 * max_relator_length]))
    return [r0_len, r1_len]


def apply_random_ac_moves(presentation, max_relator_length, lengths, n_moves, rng):
    """Apply n_moves random AC' moves to a presentation.

    Parameters:
        presentation: numpy array of dtype int8
        max_relator_length: int, max length each relator can take
        lengths: list of two ints, current word lengths
        n_moves: number of random moves to apply
        rng: numpy.random.Generator instance for reproducibility

    Returns:
        (presentation, lengths) after applying all moves
    """
    for _ in range(n_moves):
        move_id = int(rng.integers(0, 12))
        presentation, lengths = ACMove(
            move_id, presentation, max_relator_length, lengths
        )
    return presentation, lengths


def generate_dataset_for_presentation(args_tuple):
    """Generate all presentations for one initial P0 (Algorithm 6 inner loop).

    This function is designed to be called via multiprocessing.Pool.map().

    Parameters:
        args_tuple: (p0_index, p0_list, config_dict) where
            p0_index: int, index of the initial presentation (0-1189)
            p0_list: list of ints, the initial presentation in PPO encoding
            config_dict: dict with keys n_phases, n_chains, n_moves, lmax, seed

    Returns:
        (presentations, metadata) where
            presentations: numpy array of shape (n_phases * n_chains, 2 * lmax), dtype int8
            metadata: numpy array of shape (n_phases * n_chains, 3), dtype int32
                columns: [origin_index, phase_index, chain_index]
    """
    p0_index, p0_list, config = args_tuple

    n_phases = config["n_phases"]
    n_chains = config["n_chains"]
    n_moves = config["n_moves"]
    lmax = config["lmax"]
    seed = config["seed"]

    # Deterministic RNG seeded by (global_seed, presentation_index)
    rng = np.random.default_rng([seed, p0_index])

    # Convert initial presentation to numpy array and determine its layout
    p0 = np.array(p0_list, dtype=np.int8)
    old_max = len(p0) // 2

    # Compute l = longest relator length in P0
    lengths_p0 = get_word_lengths(p0, old_max)
    l = max(lengths_p0)

    # Resize P0 to lmax layout for storage
    p0_lmax = resize_presentation_np(p0, old_max, lmax)

    # Incremental increase per phase
    linc = (lmax - l) / n_phases

    # Output buffers
    total_output = n_phases * n_chains
    presentations = np.zeros((total_output, 2 * lmax), dtype=np.int8)
    metadata = np.zeros((total_output, 3), dtype=np.int32)

    # Chain states: each chain maintains its current presentation (in lmax layout)
    chain_states = [p0_lmax.copy() for _ in range(n_chains)]

    output_idx = 0
    for phase_i in range(n_phases):
        for chain_j in range(n_chains):
            # Sample max relator length for this phase
            li_low = l + phase_i * linc
            li_high = l + (phase_i + 1) * linc
            li = int(math.floor(rng.uniform(li_low, li_high)))
            li = max(li, max(get_word_lengths(chain_states[chain_j], lmax)))
            li = min(li, lmax)

            # Resize chain state from lmax to li for AC move application
            working_pres = resize_presentation_np(chain_states[chain_j], lmax, li)
            working_lengths = get_word_lengths(working_pres, li)

            # Apply N random AC' moves
            working_pres, working_lengths = apply_random_ac_moves(
                working_pres, li, working_lengths, n_moves, rng
            )

            # Resize back to lmax for storage
            result_lmax = resize_presentation_np(working_pres, li, lmax)
            chain_states[chain_j] = result_lmax

            # Store result
            presentations[output_idx] = result_lmax
            metadata[output_idx] = [p0_index, phase_i, chain_j]
            output_idx += 1

    return presentations, metadata


def generate_full_dataset(args):
    """Main entry point: generate the full ~1.8M presentation dataset.

    Parameters:
        args: argparse.Namespace with fields:
            n_phases, n_chains, n_moves, lmax, output_dir, n_workers, seed
    """
    print("Loading initial presentations...")
    initial_states = load_initial_states_from_text_file("all")
    n_presentations = len(initial_states)
    total_per_p0 = args.n_phases * args.n_chains
    total_dataset = n_presentations * total_per_p0

    print(f"Configuration:")
    print(f"  Initial presentations: {n_presentations}")
    print(f"  Phases: {args.n_phases}")
    print(f"  Chains per presentation: {args.n_chains}")
    print(f"  Moves per phase: {args.n_moves}")
    print(f"  Max relator length: {args.lmax}")
    print(f"  Total output presentations: {total_dataset:,}")
    print(f"  Workers: {args.n_workers}")
    print(f"  Seed: {args.seed}")

    # Prepare arguments for each worker
    config = {
        "n_phases": args.n_phases,
        "n_chains": args.n_chains,
        "n_moves": args.n_moves,
        "lmax": args.lmax,
        "seed": args.seed,
    }
    worker_args = [
        (idx, initial_states[idx], config) for idx in range(n_presentations)
    ]

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run generation
    start_time = time.time()

    all_presentations = np.zeros((total_dataset, 2 * args.lmax), dtype=np.int8)
    all_metadata = np.zeros((total_dataset, 3), dtype=np.int32)

    if args.n_workers <= 1:
        # Single-process mode (useful for debugging)
        results = []
        for wa in tqdm(worker_args, desc="Generating dataset"):
            results.append(generate_dataset_for_presentation(wa))
    else:
        # Multi-process mode
        with Pool(processes=args.n_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(generate_dataset_for_presentation, worker_args),
                    total=n_presentations,
                    desc="Generating dataset",
                )
            )

    # Assemble results
    print("Assembling results...")
    offset = 0
    for presentations, metadata in results:
        n = len(presentations)
        all_presentations[offset : offset + n] = presentations
        all_metadata[offset : offset + n] = metadata
        offset += n

    elapsed = time.time() - start_time
    print(f"Generation complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save outputs
    pres_path = os.path.join(args.output_dir, "presentations.npy")
    meta_path = os.path.join(args.output_dir, "metadata.npy")
    config_path = os.path.join(args.output_dir, "config.json")

    print(f"Saving presentations to {pres_path}...")
    np.save(pres_path, all_presentations)

    print(f"Saving metadata to {meta_path}...")
    np.save(meta_path, all_metadata)

    config_out = {
        "n_presentations": n_presentations,
        "n_phases": args.n_phases,
        "n_chains": args.n_chains,
        "n_moves": args.n_moves,
        "lmax": args.lmax,
        "seed": args.seed,
        "total_generated": total_dataset,
        "elapsed_seconds": elapsed,
    }
    with open(config_path, "w") as f:
        json.dump(config_out, f, indent=2)
    print(f"Saved config to {config_path}")

    # Quick validation
    print("\nValidation:")
    print(f"  Shape: {all_presentations.shape}")
    print(f"  Dtype: {all_presentations.dtype}")
    file_size_mb = os.path.getsize(pres_path) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate transformer training dataset (Algorithm 6)"
    )
    parser.add_argument(
        "--n-phases",
        type=int,
        default=128,
        help="Number of phases per chain (default: 128)",
    )
    parser.add_argument(
        "--n-chains",
        type=int,
        default=12,
        help="Number of parallel chains per initial presentation (default: 12)",
    )
    parser.add_argument(
        "--n-moves",
        type=int,
        default=1000,
        help="Number of random AC' moves per phase (default: 1000)",
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=128,
        help="Upper bound on relator length (default: 128)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/transformer_dataset",
        help="Output directory (default: data/transformer_dataset)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of worker processes (default: cpu_count)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    if args.n_workers is None:
        args.n_workers = os.cpu_count() or 1
    return args


if __name__ == "__main__":
    args = parse_args()
    generate_full_dataset(args)
