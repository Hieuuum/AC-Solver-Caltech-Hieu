"""
Dataset generation for the transformer language model (Algorithm 6 from Appendix D).

Generates ~1.8 million balanced presentations by applying random AC' moves to the
1190 Miller-Schupp seed presentations through 128 phases of gradually increasing
maximum relator length.

Supports incremental shard-based saving with resume, so long-running generation
can survive interruptions without losing progress.

Usage:
    python -m ac_solver.transformer.data_generator [options]

Examples:
    # Full generation with incremental saving
    python -m ac_solver.transformer.data_generator --n-workers 8 --seed 42

    # Resume after interruption
    python -m ac_solver.transformer.data_generator --resume

    # Small test run
    python -m ac_solver.transformer.data_generator --n-phases 2 --n-chains 2 --n-moves 10 --shard-size 50
"""

import argparse
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from ac_solver.agents.utils import load_initial_states_from_text_file


# ---------------------------------------------------------------------------
# Pre-computed AC' move parameter table (avoids per-call modular arithmetic).
# Each entry: (move_type, i, j, sign)
#   move_type 0 = concatenation (AC'1), 1 = conjugation (AC'2)
#   i = relator index to modify (0 or 1)
#   j = other relator (concat) or generator index 1=x, 2=y (conjugate)
#   sign = +1 or -1
# ---------------------------------------------------------------------------
_MOVE_PARAMS = (
    (0, 1, 0, 1),   # 0:  r1 -> r1*r0
    (0, 0, 1, -1),  # 1:  r0 -> r0*r1^-1
    (0, 1, 0, -1),  # 2:  r1 -> r1*r0^-1
    (0, 0, 1, 1),   # 3:  r0 -> r0*r1
    (1, 1, 1, -1),  # 4:  r1 -> x^-1*r1*x
    (1, 0, 2, -1),  # 5:  r0 -> y^-1*r0*y
    (1, 1, 2, -1),  # 6:  r1 -> y^-1*r1*y
    (1, 0, 1, 1),   # 7:  r0 -> x*r0*x^-1
    (1, 1, 1, 1),   # 8:  r1 -> x*r1*x^-1
    (1, 0, 2, 1),   # 9:  r0 -> y*r0*y^-1
    (1, 1, 2, 1),   # 10: r1 -> y*r1*y^-1
    (1, 0, 1, -1),  # 11: r0 -> x^-1*r0*x
)


# ---------------------------------------------------------------------------
# Fast list-based AC' move functions for dataset generation.
#
# These operate on Python lists instead of numpy arrays to avoid per-element
# overhead of numpy scalar access, np.delete, and np.pad.  The key win is
# the stack-based O(n) free reduction that replaces the scan-and-delete
# approach in envs/utils.simplify_relator (which allocates new arrays via
# np.delete/np.pad on every cancellation).
# ---------------------------------------------------------------------------


def _fast_simplify_relator(pres, offset, max_len):
    """In-place free + cyclic reduction of relator at pres[offset:offset+max_len].

    Uses a stack-based O(n) single-pass algorithm.

    Parameters:
        pres: Python list of ints (modified in-place)
        offset: start index of the relator slot
        max_len: slot size for the relator (right-zero-padded)

    Returns:
        int: new length of the relator after reduction
    """
    # Find actual word length (consecutive non-zero elements from the left)
    length = 0
    for k in range(max_len):
        if pres[offset + k] == 0:
            break
        length += 1
    else:
        length = max_len

    if length <= 1:
        return length

    # Stack-based free reduction (O(n), in-place).
    # Write position (top) is always <= read position, so in-place is safe.
    top = 0
    for r in range(length):
        val = pres[offset + r]
        if top > 0 and pres[offset + top - 1] + val == 0:
            top -= 1  # cancel adjacent inverse pair
        else:
            pres[offset + top] = val
            top += 1

    # Zero out the remainder of the slot
    for k in range(top, max_len):
        pres[offset + k] = 0
    length = top

    # Cyclic reduction: cancel matching pairs at the word boundaries
    if length > 1:
        nc = 0
        half = length // 2
        while nc < half and pres[offset + nc] + pres[offset + length - 1 - nc] == 0:
            nc += 1
        if nc > 0:
            new_len = length - 2 * nc
            for k in range(new_len):
                pres[offset + k] = pres[offset + nc + k]
            for k in range(new_len, length):
                pres[offset + k] = 0
            length = new_len

    return length


def _fast_concatenate(pres, max_len, i, j, sign, lengths):
    """In-place concatenation: r_i -> r_i * r_j^sign, with boundary free reduction.

    If the result would exceed max_len, the move is skipped (pres unchanged).

    Parameters:
        pres: Python list of ints (modified in-place)
        max_len: slot size per relator
        i, j: relator indices (0 or 1, i != j)
        sign: +1 or -1 (whether to invert r_j)
        lengths: list [len_r0, len_r1] (modified in-place)
    """
    i_off = i * max_len
    j_off = j * max_len
    len_i = lengths[i]
    len_j = lengths[j]

    if sign == 1:
        # r_j as-is — read directly from buffer (i != j, no overlap)
        acc = 0
        m = min(len_i, len_j)
        while acc < m and pres[i_off + len_i - 1 - acc] + pres[j_off + acc] == 0:
            acc += 1

        new_size = len_i + len_j - 2 * acc
        if new_size <= max_len:
            # r_i[:len_i-acc] is already in place; append r_j[acc:]
            write_pos = len_i - acc
            for k in range(len_j - acc):
                pres[i_off + write_pos + k] = pres[j_off + acc + k]
            # Zero out old tail if result is shorter
            for k in range(new_size, len_i):
                pres[i_off + k] = 0
            lengths[i] = new_size
    else:
        # r_j^{-1}: reverse and negate (needs temp buffer since order changes)
        rj_inv = [-pres[j_off + len_j - 1 - k] for k in range(len_j)]

        acc = 0
        m = min(len_i, len_j)
        while acc < m and pres[i_off + len_i - 1 - acc] + rj_inv[acc] == 0:
            acc += 1

        new_size = len_i + len_j - 2 * acc
        if new_size <= max_len:
            write_pos = len_i - acc
            for k in range(len_j - acc):
                pres[i_off + write_pos + k] = rj_inv[acc + k]
            for k in range(new_size, len_i):
                pres[i_off + k] = 0
            lengths[i] = new_size


def _fast_conjugate(pres, max_len, i, j, sign, lengths):
    """In-place conjugation: r_i -> x_j^sign * r_i * x_j^{-sign}.

    If the result would exceed max_len, the move is skipped (pres unchanged).

    Parameters:
        pres: Python list of ints (modified in-place)
        max_len: slot size per relator
        i: relator index (0 or 1)
        j: generator index (1=x, 2=y)
        sign: +1 or -1
        lengths: list [len_r0, len_r1] (modified in-place)
    """
    i_off = i * max_len
    len_i = lengths[i]

    if len_i == 0:
        return  # safety: can't conjugate empty relator

    generator = sign * j  # element to prepend

    # Check if boundary elements cancel with the conjugating generator
    start_cancel = 1 if pres[i_off] == -generator else 0
    end_cancel = 1 if pres[i_off + len_i - 1] == generator else 0

    new_size = len_i + 2 - 2 * (start_cancel + end_cancel)

    if new_size <= max_len:
        # Build result in temp list (at most len_i + 2 elements)
        result = []
        if not start_cancel:
            result.append(generator)
        for k in range(start_cancel, len_i - end_cancel):
            result.append(pres[i_off + k])
        if not end_cancel:
            result.append(-generator)

        # Write back
        for k in range(new_size):
            pres[i_off + k] = result[k]
        # Zero out old tail if result is shorter
        for k in range(new_size, len_i):
            pres[i_off + k] = 0

        lengths[i] = new_size


def _fast_ac_move(move_id, pres, max_len, lengths):
    """Apply one AC' move followed by full free + cyclic reduction. In-place.

    Parameters:
        move_id: int in [0, 11]
        pres: Python list of ints (modified in-place)
        max_len: slot size per relator
        lengths: list [len_r0, len_r1] (modified in-place)
    """
    move_type, i, j, sign = _MOVE_PARAMS[move_id]

    if move_type == 0:
        _fast_concatenate(pres, max_len, i, j, sign, lengths)
    else:
        _fast_conjugate(pres, max_len, i, j, sign, lengths)

    # Full free + cyclic simplification on both relators
    lengths[0] = _fast_simplify_relator(pres, 0, max_len)
    lengths[1] = _fast_simplify_relator(pres, max_len, max_len)


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

    Uses optimized list-based operations to avoid numpy overhead in the
    inner loop.  The hot path (move application + simplification) runs on
    Python lists, which is ~3-5x faster than the numpy-based ACMove path
    for small arrays (relator lengths < 256).

    Parameters:
        presentation: numpy array of dtype int8
        max_relator_length: int, max length each relator can take
        lengths: list of two ints, current word lengths
        n_moves: number of random moves to apply
        rng: numpy.random.Generator instance for reproducibility

    Returns:
        (presentation, lengths) after applying all moves
    """
    # Batch all random move IDs in one call (avoids n_moves individual draws)
    move_ids = rng.integers(0, 12, size=n_moves)

    # Convert to Python list for fast element access in the inner loop
    pres_list = presentation.tolist()
    lengths = list(lengths)

    for k in range(n_moves):
        _fast_ac_move(int(move_ids[k]), pres_list, max_relator_length, lengths)

    return np.array(pres_list, dtype=np.int8), lengths


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

            # Batch all random move IDs for this iteration
            move_ids = rng.integers(0, 12, size=n_moves)

            # Convert to Python list for fast inner loop
            pres_list = working_pres.tolist()
            working_lengths = list(working_lengths)

            for k in range(n_moves):
                _fast_ac_move(int(move_ids[k]), pres_list, li, working_lengths)

            # Convert back to numpy
            working_pres = np.array(pres_list, dtype=np.int8)

            # Resize back to lmax for storage
            result_lmax = resize_presentation_np(working_pres, li, lmax)
            chain_states[chain_j] = result_lmax

            # Store result
            presentations[output_idx] = result_lmax
            metadata[output_idx] = [p0_index, phase_i, chain_j]
            output_idx += 1

    return presentations, metadata


def load_progress(output_dir):
    """Load generation progress from progress.json.

    Parameters:
        output_dir: path to the output directory

    Returns:
        dict with keys: completed_shards (list of int), config (dict),
        total_elapsed (float). Returns empty defaults if file doesn't exist.
    """
    progress_path = os.path.join(output_dir, "progress.json")
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            return json.load(f)
    return {"completed_shards": [], "total_elapsed": 0.0}


def save_progress(output_dir, completed_shards, config, total_elapsed):
    """Save generation progress to progress.json.

    Parameters:
        output_dir: path to the output directory
        completed_shards: list of completed shard indices
        config: dict of generation parameters
        total_elapsed: total elapsed time in seconds
    """
    progress = {
        "completed_shards": sorted(completed_shards),
        "config": config,
        "total_elapsed": total_elapsed,
    }
    progress_path = os.path.join(output_dir, "progress.json")
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


def save_shard(shard_idx, results, output_dir, lmax):
    """Save a single shard's results to disk.

    Parameters:
        shard_idx: int, the shard index
        results: list of (presentations, metadata) tuples from workers
        output_dir: path to the output directory
        lmax: max relator length, used for array width calculation
    """
    shard_dir = os.path.join(output_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    # Count total rows in this shard
    total_rows = sum(len(pres) for pres, _ in results)

    shard_presentations = np.zeros((total_rows, 2 * lmax), dtype=np.int8)
    shard_metadata = np.zeros((total_rows, 3), dtype=np.int32)

    offset = 0
    for presentations, metadata in results:
        n = len(presentations)
        shard_presentations[offset : offset + n] = presentations
        shard_metadata[offset : offset + n] = metadata
        offset += n

    pres_path = os.path.join(shard_dir, f"shard_{shard_idx:04d}_presentations.npy")
    meta_path = os.path.join(shard_dir, f"shard_{shard_idx:04d}_metadata.npy")

    np.save(pres_path, shard_presentations)
    np.save(meta_path, shard_metadata)


def merge_shards(output_dir, n_shards, lmax):
    """Merge all shard files into single presentations.npy and metadata.npy.

    Parameters:
        output_dir: path to the output directory
        n_shards: total number of shards
        lmax: max relator length

    Returns:
        (total_presentations, total_metadata) numpy arrays
    """
    shard_dir = os.path.join(output_dir, "shards")

    # First pass: count total rows
    total_rows = 0
    for shard_idx in range(n_shards):
        pres_path = os.path.join(shard_dir, f"shard_{shard_idx:04d}_presentations.npy")
        shard_pres = np.load(pres_path)
        total_rows += len(shard_pres)

    # Allocate final arrays
    all_presentations = np.zeros((total_rows, 2 * lmax), dtype=np.int8)
    all_metadata = np.zeros((total_rows, 3), dtype=np.int32)

    # Second pass: fill
    offset = 0
    for shard_idx in range(n_shards):
        pres_path = os.path.join(shard_dir, f"shard_{shard_idx:04d}_presentations.npy")
        meta_path = os.path.join(shard_dir, f"shard_{shard_idx:04d}_metadata.npy")

        shard_pres = np.load(pres_path)
        shard_meta = np.load(meta_path)
        n = len(shard_pres)

        all_presentations[offset : offset + n] = shard_pres
        all_metadata[offset : offset + n] = shard_meta
        offset += n

    # Save merged files
    pres_path = os.path.join(output_dir, "presentations.npy")
    meta_path = os.path.join(output_dir, "metadata.npy")

    print(f"Saving merged presentations to {pres_path}...")
    np.save(pres_path, all_presentations)

    print(f"Saving merged metadata to {meta_path}...")
    np.save(meta_path, all_metadata)

    return all_presentations, all_metadata


def generate_full_dataset(args):
    """Main entry point: generate the full ~1.8M presentation dataset.

    Uses shard-based incremental saving so that progress is preserved across
    interruptions. Each shard processes a batch of initial presentations and
    saves results to disk immediately.

    Parameters:
        args: argparse.Namespace with fields:
            n_phases, n_chains, n_moves, lmax, output_dir, n_workers, seed,
            shard_size, resume, no_merge
    """
    print("Loading initial presentations...")
    initial_states = load_initial_states_from_text_file("all")
    n_presentations = len(initial_states)
    total_per_p0 = args.n_phases * args.n_chains
    total_dataset = n_presentations * total_per_p0

    # Sharding
    shard_size = args.shard_size
    n_shards = math.ceil(n_presentations / shard_size)

    print(f"Configuration:")
    print(f"  Initial presentations: {n_presentations}")
    print(f"  Phases: {args.n_phases}")
    print(f"  Chains per presentation: {args.n_chains}")
    print(f"  Moves per phase: {args.n_moves}")
    print(f"  Max relator length: {args.lmax}")
    print(f"  Total output presentations: {total_dataset:,}")
    print(f"  Workers: {args.n_workers}")
    print(f"  Seed: {args.seed}")
    print(f"  Shard size: {shard_size} presentations ({n_shards} shards)")

    config = {
        "n_phases": args.n_phases,
        "n_chains": args.n_chains,
        "n_moves": args.n_moves,
        "lmax": args.lmax,
        "seed": args.seed,
    }

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load progress for resume
    completed_shards = set()
    total_elapsed = 0.0
    if args.resume:
        progress = load_progress(args.output_dir)
        completed_shards = set(progress.get("completed_shards", []))
        total_elapsed = progress.get("total_elapsed", 0.0)
        if completed_shards:
            print(f"Resuming: {len(completed_shards)}/{n_shards} shards already complete")

    # Process shards
    for shard_idx in range(n_shards):
        if shard_idx in completed_shards:
            continue

        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, n_presentations)

        worker_args = [
            (idx, initial_states[idx], config) for idx in range(start_idx, end_idx)
        ]

        shard_start = time.time()
        print(f"\nShard {shard_idx + 1}/{n_shards} "
              f"(presentations {start_idx}-{end_idx - 1})...")

        if args.n_workers <= 1:
            results = []
            for wa in tqdm(worker_args, desc=f"Shard {shard_idx + 1}"):
                results.append(generate_dataset_for_presentation(wa))
        else:
            with Pool(processes=args.n_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(generate_dataset_for_presentation, worker_args),
                        total=len(worker_args),
                        desc=f"Shard {shard_idx + 1}",
                    )
                )

        # Save shard immediately
        save_shard(shard_idx, results, args.output_dir, args.lmax)

        shard_elapsed = time.time() - shard_start
        total_elapsed += shard_elapsed
        completed_shards.add(shard_idx)

        # Update progress
        save_progress(args.output_dir, list(completed_shards), config, total_elapsed)

        shard_rows = sum(len(pres) for pres, _ in results)
        print(f"  Saved {shard_rows:,} presentations in {shard_elapsed:.1f}s")

    print(f"\nAll shards complete in {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    # Merge shards into final output
    if not args.no_merge:
        print("\nMerging shards...")
        all_presentations, all_metadata = merge_shards(
            args.output_dir, n_shards, args.lmax
        )

        # Save config
        config_out = {
            "n_presentations": n_presentations,
            "n_phases": args.n_phases,
            "n_chains": args.n_chains,
            "n_moves": args.n_moves,
            "lmax": args.lmax,
            "seed": args.seed,
            "shard_size": shard_size,
            "n_shards": n_shards,
            "total_generated": len(all_presentations),
            "elapsed_seconds": total_elapsed,
        }
        config_path = os.path.join(args.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_out, f, indent=2)
        print(f"Saved config to {config_path}")

        # Quick validation
        pres_path = os.path.join(args.output_dir, "presentations.npy")
        print("\nValidation:")
        print(f"  Shape: {all_presentations.shape}")
        print(f"  Dtype: {all_presentations.dtype}")
        file_size_mb = os.path.getsize(pres_path) / (1024 * 1024)
        print(f"  File size: {file_size_mb:.1f} MB")
    else:
        print("Skipping merge (--no-merge). Shard files are in: "
              f"{os.path.join(args.output_dir, 'shards')}")


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
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100,
        help="Number of initial presentations per shard (default: 100)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip final merge step (keep shard files only)",
    )
    args = parser.parse_args()
    if args.n_workers is None:
        args.n_workers = os.cpu_count() or 1
    return args


if __name__ == "__main__":
    args = parse_args()
    generate_full_dataset(args)
