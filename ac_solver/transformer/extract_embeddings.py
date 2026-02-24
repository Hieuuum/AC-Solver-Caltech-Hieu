"""
Extract EOS-token embeddings from the 1190 Miller-Schupp presentations.

Loads a trained ACTransformer checkpoint, passes all 1190 presentations
through the model, and captures the final-layer hidden state at the EOS
token position for each presentation.

Outputs (saved to --output-dir):
    embeddings.npy  : shape (1190, 512), float32
    labels.npy      : shape (1190,),     int32
                      1 = GS-solved  (first 533 presentations)
                      0 = GS-unsolved (remaining 657 presentations)

Usage:
    python -m ac_solver.transformer.extract_embeddings \\
        --checkpoint ac_solver/transformer/checkpoints/model_final.pt

    # Custom output directory
    python -m ac_solver.transformer.extract_embeddings \\
        --checkpoint <path> --output-dir data/embeddings/
"""

import argparse
import os
from ast import literal_eval
from importlib import resources

import numpy as np
import torch

from ac_solver.transformer.model import ACTransformer
from ac_solver.transformer.tokenizer import TOKEN_EOS, presentation_to_tokens

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CKPT = os.path.join(_THIS_DIR, "checkpoints", "model_final.pt")
_DEFAULT_OUT = os.path.join(_THIS_DIR, "checkpoints")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_miller_schupp_presentations():
    """
    Load all 1190 Miller-Schupp presentations from the package data file.

    Returns
    -------
    presentations : list of list of int
        Each entry is a PPO-encoded presentation array (variable length:
        36, 40, 48, 56, 64, or 72 elements corresponding to
        max_relator_length 18–36).
    raw_lines : list of str
        The original string representations, used for GS-label matching.
    """
    with resources.open_text(
        "ac_solver.search.miller_schupp.data", "all_presentations.txt"
    ) as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    presentations = [literal_eval(line) for line in raw_lines]
    return presentations, raw_lines


def build_gs_labels(raw_lines):
    """
    Build binary GS-solved labels for the 1190 presentations.

    Label 1 = GS-solved, 0 = GS-unsolved.
    The first 533 entries in all_presentations.txt are GS-solved.

    Returns
    -------
    labels : np.ndarray, shape (1190,), dtype int32
    """
    with resources.open_text(
        "ac_solver.search.miller_schupp.data", "greedy_solved_presentations.txt"
    ) as f:
        gs_solved_set = {line.strip() for line in f if line.strip()}

    labels = np.array(
        [1 if line in gs_solved_set else 0 for line in raw_lines], dtype=np.int32
    )
    return labels


# ---------------------------------------------------------------------------
# Tokenization + batching
# ---------------------------------------------------------------------------


def tokenize_all(presentations):
    """
    Tokenize all presentations and return token tensors + per-sample EOS
    positions (needed to index the correct hidden state).

    Each sequence has format: [r0_tokens..., SEP, r1_tokens..., EOS]
    The EOS token is always the last element of the real sequence.

    Returns
    -------
    padded : torch.LongTensor, shape (N, max_seq_len)
        Token IDs, right-padded with TOKEN_EOS.
    eos_positions : list of int
        Index of the EOS token in each padded sequence (= seq_len - 1).
    """
    token_lists = []
    for pres in presentations:
        pres_arr = pres if isinstance(pres, list) else list(pres)
        mrl = len(pres_arr) // 2
        tokens = presentation_to_tokens(pres_arr, max_relator_length=mrl)
        token_lists.append(tokens)

    eos_positions = [len(t) - 1 for t in token_lists]
    max_len = max(len(t) for t in token_lists)

    N = len(token_lists)
    padded = torch.full((N, max_len), fill_value=TOKEN_EOS, dtype=torch.long)
    for i, tokens in enumerate(token_lists):
        padded[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)

    return padded, eos_positions


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


def extract_embeddings(model, input_ids, eos_positions, device, batch_size=256):
    """
    Run the model in eval mode and collect the EOS hidden state for each
    sample.

    Parameters
    ----------
    model : ACTransformer
    input_ids : LongTensor, shape (N, T)
    eos_positions : list of int, length N
    device : torch.device
    batch_size : int
        Number of presentations per forward pass (1190 is small enough
        to do in one pass, but this keeps GPU memory bounded).

    Returns
    -------
    embeddings : np.ndarray, shape (N, d_model), float32
    """
    model.eval()
    N = input_ids.shape[0]
    d_model = model.d_model
    embeddings = np.zeros((N, d_model), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_ids = input_ids[start:end].to(device)

            # hidden: (B, T, d_model)
            hidden = model.get_hidden_states(batch_ids)

            for local_i, global_i in enumerate(range(start, end)):
                eos_pos = eos_positions[global_i]
                embeddings[global_i] = hidden[local_i, eos_pos, :].cpu().float().numpy()

    return embeddings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load presentations ---
    print("Loading Miller-Schupp presentations...")
    presentations, raw_lines = load_miller_schupp_presentations()
    n = len(presentations)
    print(f"  Loaded {n} presentations")

    # --- Build labels ---
    labels = build_gs_labels(raw_lines)
    n_solved = int(labels.sum())
    n_unsolved = int((labels == 0).sum())
    print(f"  GS-solved: {n_solved}  |  GS-unsolved: {n_unsolved}")

    # --- Tokenize ---
    print("Tokenizing presentations...")
    input_ids, eos_positions = tokenize_all(presentations)
    seq_lens = [p + 1 for p in eos_positions]
    print(f"  Token sequence lengths: min={min(seq_lens)}, max={max(seq_lens)}, "
          f"mean={sum(seq_lens)/len(seq_lens):.1f}")
    print(f"  Padded input shape: {tuple(input_ids.shape)}")

    # --- Load model ---
    print(f"Loading model from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Run train_lm.py first to produce a trained model."
        )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = ACTransformer(
        vocab_size=6, d_model=512, n_heads=4, n_layers=8, context_length=1024
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    trained_step = ckpt.get("step", "?")
    trained_epoch = ckpt.get("epoch", "?")
    print(f"  Loaded checkpoint (epoch={trained_epoch}, step={trained_step})")

    # --- Extract embeddings ---
    print("Extracting EOS embeddings...")
    embeddings = extract_embeddings(
        model, input_ids, eos_positions, device, batch_size=args.batch_size
    )
    print(f"  Embedding matrix shape: {embeddings.shape}  (expect (1190, 512))")
    assert embeddings.shape == (n, 512), (
        f"Expected (1190, 512), got {embeddings.shape}"
    )

    # --- Save ---
    os.makedirs(args.output_dir, exist_ok=True)
    emb_path = os.path.join(args.output_dir, "embeddings.npy")
    lbl_path = os.path.join(args.output_dir, "labels.npy")
    np.save(emb_path, embeddings)
    np.save(lbl_path, labels)
    print(f"  Saved embeddings → {emb_path}")
    print(f"  Saved labels     → {lbl_path}")

    # --- Quick sanity check ---
    emb_reload = np.load(emb_path)
    lbl_reload = np.load(lbl_path)
    assert emb_reload.shape == (1190, 512)
    assert lbl_reload.shape == (1190,)
    print("\nDone. Embedding stats:")
    print(f"  mean abs value : {np.abs(embeddings).mean():.4f}")
    print(f"  std            : {embeddings.std():.4f}")
    print(f"  label counts   : {lbl_reload.sum()} solved, {(lbl_reload==0).sum()} unsolved")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract EOS embeddings from 1190 Miller-Schupp presentations"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=_DEFAULT_CKPT,
        help="Path to trained ACTransformer checkpoint (default: checkpoints/model_final.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=_DEFAULT_OUT,
        help="Directory to save embeddings.npy and labels.npy",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for model forward passes (default: 256)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
