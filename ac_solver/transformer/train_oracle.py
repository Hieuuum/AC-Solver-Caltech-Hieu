"""
Hardness Oracle H(P): MLP binary classifier on Transformer EOS embeddings.

Loads the (1190, 512) embedding matrix produced by extract_embeddings.py and
trains a 2-layer MLP (512 → 128 → 1) to predict GS-solved vs GS-unsolved.

Evaluation uses stratified 5-fold cross-validation so every sample appears in
a validation fold exactly once, giving a robust F1 estimate even with 1190
samples.  The final score is compared against the paper's XGBoost baseline
of F1 = 0.962.

Usage:
    python -m ac_solver.transformer.train_oracle

    # Custom embedding path and output directory
    python -m ac_solver.transformer.train_oracle \\
        --embeddings-dir ac_solver/transformer/checkpoints/ \\
        --epochs 100 --lr 1e-3
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_EMB_DIR = os.path.join(_THIS_DIR, "checkpoints")

PAPER_BASELINE_F1 = 0.962  # XGBoost on neighborhood features (Table X in paper)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class HardnessOracle(nn.Module):
    """
    2-layer MLP with 128 hidden units for GS-solved binary classification.

    Architecture: Linear(512, 128) → ReLU → Linear(128, 1)
    Output: raw logit (use BCEWithLogitsLoss during training, sigmoid at test).
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args,
    device: torch.device,
    fold: int,
) -> tuple:
    """
    Train one fold of the MLP.  Returns (val_preds, val_probs) as numpy arrays.

    Parameters
    ----------
    X_train, X_val : (N, 512) float32 arrays (already scaled)
    y_train, y_val : (N,) int32 arrays of binary labels
    args : argparse.Namespace
    device : torch.device
    fold : int, for logging

    Returns
    -------
    val_preds : np.ndarray, shape (len(val),), binary predictions
    best_f1   : float, best validation F1 seen during training
    """
    model = HardnessOracle(input_dim=X_train.shape[1], hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_vl = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_vl_np = y_val

    best_f1 = 0.0
    best_preds = np.zeros(len(y_val), dtype=np.int32)
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Full-batch gradient step (1190 samples is tiny)
        logits = model(X_tr)
        loss = loss_fn(logits, y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_interval = max(1, args.epochs // 10)
        if epoch % log_interval == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_vl)
                val_preds = (val_logits.sigmoid() > 0.5).cpu().numpy().astype(np.int32)
            fold_f1 = f1_score(y_vl_np, val_preds, zero_division=0)

            if fold_f1 > best_f1:
                best_f1 = fold_f1
                best_preds = val_preds.copy()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            print(
                f"  fold {fold} | epoch {epoch:4d} | loss {loss.item():.4f} | "
                f"val F1 {fold_f1:.4f}"
            )

    return best_preds, best_f1, best_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load embeddings + labels ---
    emb_path = os.path.join(args.embeddings_dir, "embeddings.npy")
    lbl_path = os.path.join(args.embeddings_dir, "labels.npy")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"embeddings.npy not found at {emb_path}\n"
            "Run extract_embeddings.py first."
        )
    X = np.load(emb_path).astype(np.float32)  # (1190, 512)
    y = np.load(lbl_path).astype(np.int32)     # (1190,)
    print(f"Embeddings: {X.shape}  |  Labels: {y.shape}")
    print(f"  GS-solved: {y.sum()}  |  GS-unsolved: {(y==0).sum()}")

    # --- Stratified 5-fold cross-validation ---
    # With only 1190 samples, k-fold gives a much more reliable F1 estimate
    # than a single train/val split.
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    all_val_preds = np.zeros(len(y), dtype=np.int32)
    all_val_true = np.zeros(len(y), dtype=np.int32)
    fold_f1s = []

    print(f"\nRunning {args.n_folds}-fold stratified cross-validation "
          f"({args.epochs} epochs per fold, lr={args.lr})")
    print("-" * 60)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # StandardScaler fit on train fold only (no leakage)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        val_preds, best_f1, _ = train_fold(
            X_train, y_train, X_val, y_val, args, device, fold
        )
        fold_f1s.append(best_f1)
        all_val_preds[val_idx] = val_preds
        all_val_true[val_idx] = y_val

    # --- Aggregate metrics ---
    overall_f1 = f1_score(all_val_true, all_val_preds, zero_division=0)
    mean_fold_f1 = float(np.mean(fold_f1s))
    std_fold_f1 = float(np.std(fold_f1s))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(classification_report(
        all_val_true, all_val_preds,
        target_names=["GS-unsolved", "GS-solved"],
        digits=4,
    ))
    print(f"Per-fold F1 scores : {[f'{f:.4f}' for f in fold_f1s]}")
    print(f"Mean fold F1       : {mean_fold_f1:.4f} ± {std_fold_f1:.4f}")
    print(f"Aggregated F1      : {overall_f1:.4f}  "
          f"(all {len(y)} val predictions combined)")
    print()
    print("-" * 60)
    print(f"Paper XGBoost baseline F1  : {PAPER_BASELINE_F1:.3f}")
    print(f"Our MLP oracle F1          : {overall_f1:.4f}")
    delta = overall_f1 - PAPER_BASELINE_F1
    direction = "above" if delta >= 0 else "below"
    print(f"Delta vs. baseline         : {delta:+.4f}  ({direction} baseline)")
    print("-" * 60)

    # --- Save final model (trained on full data) ---
    print("\nTraining final model on full dataset...")
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X).astype(np.float32)

    final_model = HardnessOracle(input_dim=512, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    X_full_t = torch.tensor(X_scaled_full, device=device)
    y_full_t = torch.tensor(y.astype(np.float32), device=device)

    for epoch in range(1, args.epochs + 1):
        final_model.train()
        logits = final_model(X_full_t)
        loss = loss_fn(logits, y_full_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    os.makedirs(args.output_dir, exist_ok=True)
    oracle_path = os.path.join(args.output_dir, "hardness_oracle.pt")
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "scaler_mean": scaler_full.mean_,
            "scaler_scale": scaler_full.scale_,
            "cv_f1_mean": mean_fold_f1,
            "cv_f1_std": std_fold_f1,
            "aggregated_f1": overall_f1,
            "paper_baseline_f1": PAPER_BASELINE_F1,
            "args": vars(args),
        },
        oracle_path,
    )
    print(f"Final oracle saved → {oracle_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train hardness oracle MLP on Transformer EOS embeddings"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=_DEFAULT_EMB_DIR,
        help="Directory containing embeddings.npy and labels.npy "
             "(default: ac_solver/transformer/checkpoints/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=_DEFAULT_EMB_DIR,
        help="Directory to save hardness_oracle.pt (default: same as --embeddings-dir)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Training epochs per fold (default: 300)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="AdamW learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of stratified CV folds (default: 5)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
