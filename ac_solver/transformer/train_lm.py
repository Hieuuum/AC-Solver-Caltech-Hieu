"""
Transformer language model training loop for the AC conjecture.

Optimizations (all on by default):
  1. Sequence packing   — streams all tokens into a rolling buffer and yields
                          fixed-length (context_length,) chunks. Eliminates
                          padding waste; every token in every batch is real.
  2. BF16 mixed precision — torch.autocast on CUDA (no GradScaler needed for
                            BF16). ~1.7× faster on A100/H100/L4.
  3. torch.compile       — fuses ops across the forward pass. ~1.3× faster.
                            Disabled on CPU (overhead not worth it).
  4. Batch size 512      — better GPU tensor-core utilization vs. 128.

Use --no-pack to fall back to the original padded-batch mode (e.g., for
debugging or when running on CPU where packing overhead hurts).

Usage:
    python -m ac_solver.transformer.train_lm
    python -m ac_solver.transformer.train_lm --epochs 3 --batch-size 512
    python -m ac_solver.transformer.train_lm --no-pack --batch-size 128
"""

import argparse
import glob
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset

from ac_solver.transformer.model import ACTransformer
from ac_solver.transformer.tokenizer import TOKEN_EOS, presentation_to_tokens

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SHARDS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_THIS_DIR)), "dataset", "transformer_ds", "shards"
)
_DEFAULT_CKPT_DIR = os.path.join(_THIS_DIR, "checkpoints")


# ---------------------------------------------------------------------------
# Dataset: packed (default, recommended)
# ---------------------------------------------------------------------------


class PackedShardDataset(IterableDataset):
    """
    Streams raw presentations from shards, tokenizes on the fly, and
    concatenates tokens into a rolling buffer. Yields fixed-length
    (context_length + 1,) chunks split into (input, target) pairs.

    No padding: 100% of every batch is real tokens.
    Multi-worker: worker i processes shards i, i+W, i+2W, ...
    Buffer state persists across shards within a worker, so no tokens
    are lost at shard boundaries.

    Parameters
    ----------
    shard_paths : list of str
    lmax : int
        Max relator length used when shards were generated.
    context_length : int
        Number of tokens per training context (default: 1024).
    shuffle : bool
        Shuffle presentation order within each shard.
    seed : int
    """

    def __init__(
        self,
        shard_paths: list,
        lmax: int = 128,
        context_length: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.shard_paths = shard_paths
        self.lmax = lmax
        self.context_length = context_length
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            my_shards = self.shard_paths
            worker_id = 0
        else:
            my_shards = self.shard_paths[worker_info.id :: worker_info.num_workers]
            worker_id = worker_info.id

        rng = np.random.default_rng([self.seed, worker_id])
        needed = self.context_length + 1  # +1: need both input and target
        buf = []

        for shard_path in my_shards:
            pres = np.load(shard_path)
            order = (
                rng.permutation(len(pres)) if self.shuffle else np.arange(len(pres))
            )
            for idx in order:
                buf.extend(
                    presentation_to_tokens(pres[idx], max_relator_length=self.lmax)
                )
                while len(buf) >= needed:
                    chunk = torch.tensor(buf[:needed], dtype=torch.long)
                    buf = buf[needed:]
                    yield chunk[:-1], chunk[1:]
        # Trailing tokens < context_length are discarded (negligible loss)


# ---------------------------------------------------------------------------
# Dataset: padded (legacy / --no-pack fallback)
# ---------------------------------------------------------------------------


class ShardedPresentationDataset(IterableDataset):
    """
    Original padded-batch dataset. Yields variable-length (input, target)
    pairs; use with collate_padded below.
    """

    def __init__(
        self,
        shard_paths: list,
        lmax: int = 128,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.shard_paths = shard_paths
        self.lmax = lmax
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            my_shards = self.shard_paths
            worker_id = 0
        else:
            my_shards = self.shard_paths[worker_info.id :: worker_info.num_workers]
            worker_id = worker_info.id

        rng = np.random.default_rng([self.seed, worker_id])
        for shard_path in my_shards:
            pres = np.load(shard_path)
            order = (
                rng.permutation(len(pres)) if self.shuffle else np.arange(len(pres))
            )
            for idx in order:
                tokens = presentation_to_tokens(pres[idx], max_relator_length=self.lmax)
                t = torch.tensor(tokens, dtype=torch.long)
                yield t[:-1], t[1:]


def collate_padded(batch):
    """Pad variable-length pairs; target padding uses ignore_index=-100."""
    inputs, targets = zip(*batch)
    return (
        pad_sequence(inputs, batch_first=True, padding_value=TOKEN_EOS),
        pad_sequence(targets, batch_first=True, padding_value=-100),
    )


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def get_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Linear warmup → cosine decay to min_lr."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"Device: {device}")

    # Mixed precision: BF16 on CUDA (stable, no GradScaler needed).
    # On CPU: no autocast (overhead outweighs benefit for this model size).
    amp_dtype = torch.bfloat16 if use_cuda else None
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else torch.no_grad().__class__()  # no-op context
    )
    if amp_dtype:
        print(f"Mixed precision: {amp_dtype}")

    # --- Discover shards ---
    shard_paths = sorted(
        glob.glob(os.path.join(args.shards_dir, "shard_*_presentations.npy"))
    )
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in: {args.shards_dir}")
    print(f"Shards found: {len(shard_paths)}")

    lmax = np.load(shard_paths[0]).shape[1] // 2
    rows_per_shard = np.load(shard_paths[0]).shape[0]
    total_presentations = len(shard_paths) * rows_per_shard
    print(f"lmax: {lmax}  |  presentations: {total_presentations:,}")

    # --- Dataset + DataLoader ---
    use_pack = not args.no_pack
    if use_pack:
        dataset = PackedShardDataset(
            shard_paths=shard_paths,
            lmax=lmax,
            context_length=args.context_length,
            shuffle=True,
            seed=args.seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=use_cuda,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
        # Estimate steps: total_tokens / context_length / batch_size
        avg_tokens_per_pres = 108  # measured from shard stats
        total_tokens = total_presentations * avg_tokens_per_pres
        steps_per_epoch = total_tokens // args.context_length // args.batch_size
        print(
            f"Mode: PACKED  (context={args.context_length}, "
            f"est. {steps_per_epoch:,} steps/epoch)"
        )
    else:
        dataset = ShardedPresentationDataset(
            shard_paths=shard_paths,
            lmax=lmax,
            shuffle=True,
            seed=args.seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_padded,
            pin_memory=use_cuda,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
        steps_per_epoch = total_presentations // args.batch_size
        print(f"Mode: PADDED  (est. {steps_per_epoch:,} steps/epoch)")

    # --- Model ---
    model = ACTransformer(
        vocab_size=6,
        d_model=512,
        n_heads=4,
        n_layers=8,
        context_length=args.context_length,
    ).to(device)
    n_params = sum(p.numel() for p in set(model.parameters()))
    print(f"Parameters: {n_params:,}")

    # torch.compile: fuses ops across forward pass (~1.3× speedup on GPU).
    # Skipped on CPU (compile overhead not worth it for CPU-bound training).
    if args.compile and use_cuda:
        try:
            # Packed mode: always static shape (B, context_length) — no dynamic needed.
            # Padded mode: shape varies per batch, but compile still caches per shape.
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile: skipped ({e})")
    elif args.compile:
        print("torch.compile: skipped (CPU — overhead not beneficial)")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    # --- Resume from checkpoint ---
    start_step = 0
    start_epoch = 0
    resume_path = None if (args.resume or "").lower() == "none" else args.resume
    if resume_path is None:
        default_resume = os.path.join(args.checkpoints_dir, "model_final.pt")
        if os.path.exists(default_resume):
            resume_path = default_resume
    if resume_path is not None:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]
        start_epoch = ckpt["epoch"]
        print(f"  Resumed at epoch={start_epoch}, step={start_step}")

    # --- LR schedule ---
    total_steps = start_step + args.epochs * steps_per_epoch
    warmup_steps = max(1, int(0.05 * (args.epochs * steps_per_epoch)))
    print(
        f"Total steps ≈ {total_steps:,}  |  warmup: {warmup_steps:,}  |  "
        f"epochs: {args.epochs}  |  batch: {args.batch_size}"
    )

    # --- Training loop ---
    step = start_step
    running_loss = 0.0
    t0 = time.time()

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for input_ids, target_ids in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)

            lr_now = get_lr(step, warmup_steps, total_steps, args.lr, args.lr * 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            # Forward under autocast (BF16 on CUDA, FP32 on CPU)
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(input_ids)
                    loss = loss_fn(
                        logits.reshape(-1, logits.size(-1)),
                        target_ids.reshape(-1),
                    )
            else:
                logits = model(input_ids)
                loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            epoch_steps += 1
            running_loss += loss.item()
            epoch_loss += loss.item()

            if step % args.log_interval == 0:
                avg = running_loss / args.log_interval
                elapsed = time.time() - t0
                tok_per_sec = (
                    args.batch_size * args.context_length * args.log_interval / elapsed
                    if use_pack
                    else args.batch_size * 108 * args.log_interval / elapsed
                )
                print(
                    f"epoch {epoch:2d} | step {step:7d} | "
                    f"lr {lr_now:.2e} | loss {avg:.4f} | "
                    f"{tok_per_sec/1000:.1f}k tok/s | {elapsed:.1f}s"
                )
                running_loss = 0.0
                t0 = time.time()

            if step % args.save_interval == 0:
                ckpt_path = os.path.join(
                    args.checkpoints_dir, f"ckpt_step{step:07d}.pt"
                )
                torch.save(
                    {
                        "step": step,
                        "epoch": epoch,
                        "model_state_dict": (
                            model._orig_mod.state_dict()
                            if hasattr(model, "_orig_mod")
                            else model.state_dict()
                        ),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item(),
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"  [ckpt] {ckpt_path}")

        print(
            f"=== Epoch {epoch}/{start_epoch + args.epochs} | "
            f"avg loss: {epoch_loss / max(epoch_steps, 1):.4f} ==="
        )

    # --- Final model ---
    final_path = os.path.join(args.checkpoints_dir, "model_final.pt")
    torch.save(
        {
            "step": step,
            "epoch": start_epoch + args.epochs,
            "model_state_dict": (
                model._orig_mod.state_dict()
                if hasattr(model, "_orig_mod")
                else model.state_dict()
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        final_path,
    )
    print(f"\nTraining complete. Final model → {final_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ACTransformer (packed + BF16 + compile)"
    )
    parser.add_argument(
        "--shards-dir",
        type=str,
        default=_DEFAULT_SHARDS_DIR,
        help="Directory containing shard_*_presentations.npy files",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=_DEFAULT_CKPT_DIR,
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=1024,
        help="Tokens per packed context window (default: 1024)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size (default: 512, optimized for GPU)",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument(
        "--compile",
        action="store_true",
        default=True,
        help="Enable torch.compile (default: True on CUDA)",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--no-pack",
        action="store_true",
        default=False,
        help="Use padded batches instead of sequence packing (legacy mode)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from. Defaults to checkpoints/model_final.pt "
             "if it exists. Pass 'none' to force training from scratch.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
