# =============================================================================
# AC-Solver: Full Training Pipeline — Google Colab
# Copy each cell block into a separate Colab cell.
# All outputs are saved to Google Drive so they survive session timeouts.
# =============================================================================


# ── CELL 0: Mount Google Drive ────────────────────────────────────────────────
# Run this first. Everything persists here across sessions.

from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = '/content/drive/MyDrive/AC-Solver-Caltech-Hieu'
print(f"Workspace: {DRIVE_ROOT}")


# ── CELL 1: Clone Repo & Install Dependencies ─────────────────────────────────
# Idempotent: skips clone if repo already exists on Drive.

import os

if not os.path.exists(DRIVE_ROOT):
    print("Cloning repo to Drive...")
    !git clone https://github.com/shehper/AC-Solver-Caltech-Hieu.git "{DRIVE_ROOT}"
else:
    print("Repo already on Drive. Pulling latest changes...")
    !git -C "{DRIVE_ROOT}" pull

# Install the package and dependencies
%cd "{DRIVE_ROOT}"
!pip install -e . --quiet
!pip install datasets huggingface_hub tqdm --quiet

print("\nInstalled packages:")
!pip show torch datasets huggingface_hub | grep -E "^(Name|Version)"


# ── CELL 2: Download Dataset from HuggingFace ─────────────────────────────────
# Streams 1.8M rows and reconstructs 12 numpy shards on Drive.
# Resume-safe: skips already-downloaded shards if cell is re-run.
#
# Expected output:
#   dataset/transformer_ds/shards/shard_0000_presentations.npy  (153600, 256) int8
#   dataset/transformer_ds/shards/shard_0000_metadata.npy       (153600, 3)  int32
#   ... (12 shards total)
#
# Estimated time: ~10–20 min depending on HuggingFace bandwidth.

%cd "{DRIVE_ROOT}"
!python -m ac_solver.transformer.download_dataset

# Verify shards
import glob
shards = sorted(glob.glob('dataset/transformer_ds/shards/shard_*_presentations.npy'))
print(f"\nShards found: {len(shards)} / 12")
for s in shards:
    size_mb = os.path.getsize(s) / 1024**2
    print(f"  {os.path.basename(s)}  ({size_mb:.1f} MB)")


# ── CELL 3: Train Transformer Language Model ──────────────────────────────────
# Trains for N epochs. Auto-resumes from model_final.pt if it already exists.
# Set EXTRA_EPOCHS to the number of NEW epochs to train (e.g. 2 to add on top).
# Checkpoints saved every 2000 steps AND a final model at the end.
#
# Saved to: ac_solver/transformer/checkpoints/
#   ├── ckpt_step0002000.pt   (~330 MB, periodic)
#   └── model_final.pt        (~330 MB)  ← always updated
#
# Expected loss:  random init ≈ 1.79  →  3 epochs ≈ 0.80  →  5 epochs ≈ 0.76
#
# Estimated time per epoch (batch=256):
#   A100: ~12–18 min  |  H100: ~5–8 min  |  L4: ~25–40 min
#
# NOTE: If you're on a T4 GPU, add --no-compile (torch.compile unstable on T4).

EXTRA_EPOCHS = 2   # ← change this to train more/fewer additional epochs

%cd "{DRIVE_ROOT}"

# Check GPU
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"GPU: {gpu}")
print(f"BF16 supported: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")

!python -m ac_solver.transformer.train_lm \
    --epochs {EXTRA_EPOCHS} \
    --batch-size 256 \
    --log-interval 50 \
    --save-interval 2000

# Verify final checkpoint
ckpt_path = 'ac_solver/transformer/checkpoints/model_final.pt'
if os.path.exists(ckpt_path):
    size_mb = os.path.getsize(ckpt_path) / 1024**2
    print(f"\n✅ model_final.pt saved ({size_mb:.1f} MB)")
else:
    print("\n❌ model_final.pt NOT found — check training logs above")


# ── CELL 4: Extract Transformer Embeddings ────────────────────────────────────
# Runs the 1190 Miller-Schupp seed presentations through the trained model
# and extracts the 512-dim EOS hidden state for each.
# Fast: < 1 minute on any GPU.
#
# Saved to: ac_solver/transformer/checkpoints/
#   ├── embeddings.npy   (1190, 512) float32  ← ~2.4 MB
#   └── labels.npy       (1190,)     int32    ← ~5 KB
#       (1 = GS-solved, 0 = GS-unsolved)

%cd "{DRIVE_ROOT}"
!python -m ac_solver.transformer.extract_embeddings

import numpy as np
emb = np.load('ac_solver/transformer/checkpoints/embeddings.npy')
lbl = np.load('ac_solver/transformer/checkpoints/labels.npy')
print(f"\nembeddings.npy : {emb.shape}  dtype={emb.dtype}")
print(f"labels.npy     : {lbl.shape}  dtype={lbl.dtype}")
print(f"GS-solved: {lbl.sum()}  |  GS-unsolved: {(lbl==0).sum()}")


# ── CELL 5: Train Hardness Oracle H(P) ───────────────────────────────────────
# Trains a 2-layer MLP (512 → 128 → 1) on the embeddings via 5-fold CV.
# Fast: < 5 minutes on any GPU (or CPU).
#
# Saved to: ac_solver/transformer/checkpoints/
#   └── hardness_oracle.pt   (< 1 MB)
#
# Target: F1 ≥ 0.962  (paper baseline: XGBoost F1 = 0.962)

%cd "{DRIVE_ROOT}"
!python -m ac_solver.transformer.train_oracle

if os.path.exists('ac_solver/transformer/checkpoints/hardness_oracle.pt'):
    print("\n✅ hardness_oracle.pt saved")
else:
    print("\n❌ hardness_oracle.pt NOT found")


# ── CELL 6: Verify All Outputs ────────────────────────────────────────────────
# Quick sanity check that everything needed for Phase 3 (RL integration) exists.

%cd "{DRIVE_ROOT}"
import os

files = {
    "Transformer checkpoint" : "ac_solver/transformer/checkpoints/model_final.pt",
    "Embeddings"             : "ac_solver/transformer/checkpoints/embeddings.npy",
    "Labels"                 : "ac_solver/transformer/checkpoints/labels.npy",
    "Hardness Oracle"        : "ac_solver/transformer/checkpoints/hardness_oracle.pt",
}

print("Phase 1 asset check:")
print("-" * 55)
all_ok = True
for name, path in files.items():
    exists = os.path.exists(path)
    size   = f"{os.path.getsize(path)/1024**2:.1f} MB" if exists else "—"
    status = "✅" if exists else "❌"
    print(f"  {status}  {name:<25} {size}")
    all_ok = all_ok and exists

print("-" * 55)
if all_ok:
    print("\n✅ All Phase 1 assets ready. Proceed to Phase 3 (RL integration).")
else:
    print("\n❌ Some assets missing — re-run the failing cell above.")
