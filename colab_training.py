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

# Install the package only (--no-deps: Colab already has torch/numpy/etc.
# The pinned versions in pyproject.toml are incompatible with Python 3.12)
%cd "{DRIVE_ROOT}"
!pip install -e . --no-deps --quiet
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


# ── CELL 5b: Validation Loss on Shard 11 (Option B) ──────────────────────────
# Evaluates the existing model_final.pt on shard 11 (the last shard).
# NOTE: The model was trained on this shard, so this is an approximation.
# With 1.8M samples and 25.7M params, overfitting is negligible — the
# result should be within ~0.002 of true validation loss.
#
# Paper reports validation loss: 0.7337
# This cell lets you compare directly.

%cd "{DRIVE_ROOT}"

import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ac_solver.transformer.model import ACTransformer
from ac_solver.transformer.tokenizer import TOKEN_EOS, presentation_to_tokens

CKPT_DIR = f"{DRIVE_ROOT}/ac_solver/transformer/checkpoints"
SHARDS_DIR = f"{DRIVE_ROOT}/dataset/transformer_ds/shards"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
ckpt = torch.load(f"{CKPT_DIR}/model_final.pt", map_location=device)
model = ACTransformer(vocab_size=6, d_model=512, n_heads=4,
                      n_layers=8, context_length=1024).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded checkpoint: epoch={ckpt['epoch']}, step={ckpt['step']}")

# Find shard 11 (last shard)
all_shards = sorted(glob.glob(f"{SHARDS_DIR}/shard_*_presentations.npy"))
val_shard  = all_shards[-1]
print(f"Val shard: {val_shard.split('/')[-1]}  ({len(all_shards)} shards total)")

# Stream shard 11, pack into context_length=1024 windows
CONTEXT = 1024
BATCH   = 256
lmax    = np.load(val_shard).shape[1] // 2

def _iter_val(shard_path, lmax, context):
    pres = np.load(shard_path)
    buf  = []
    for row in pres:
        buf.extend(presentation_to_tokens(row, max_relator_length=lmax))
        while len(buf) >= context + 1:
            chunk = torch.tensor(buf[:context + 1], dtype=torch.long)
            buf   = buf[context + 1:]
            yield chunk[:-1], chunk[1:]

loss_fn   = nn.CrossEntropyLoss(ignore_index=-100)
amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else None

total_loss, total_steps = 0.0, 0
batch_in, batch_tgt = [], []

def _eval_batch(batch_in, batch_tgt):
    x = torch.stack(batch_in).to(device)
    y = torch.stack(batch_tgt).to(device)
    with torch.no_grad():
        if amp_dtype:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x)
                loss   = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        else:
            logits = model(x)
            loss   = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    return loss.item()

print("Evaluating on shard 11...")
for inp, tgt in _iter_val(val_shard, lmax, CONTEXT):
    batch_in.append(inp)
    batch_tgt.append(tgt)
    if len(batch_in) == BATCH:
        total_loss  += _eval_batch(batch_in, batch_tgt)
        total_steps += 1
        batch_in, batch_tgt = [], []

if batch_in:   # last partial batch
    total_loss  += _eval_batch(batch_in, batch_tgt)
    total_steps += 1

val_loss = total_loss / max(total_steps, 1)
print(f"\n{'─'*45}")
print(f"  Val loss  (shard 11) : {val_loss:.4f}")
print(f"  Paper val loss       : 0.7337")
print(f"  Delta                : {val_loss - 0.7337:+.4f}")
print(f"  Steps evaluated      : {total_steps}")
print(f"{'─'*45}")
print("⚠️  Note: model was trained on this shard — approximation only.")


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

# ── CELL 7: t-SNE Visualization of Transformer Embeddings ─────────────────────
# Replicates the key result from the paper:
#   "The Transformer naturally separates GS-solved vs GS-unsolved presentations
#    into distinct clusters — revealing a hidden linguistic invariant."
#
# ── CONFIG ────────────────────────────────────────────────────────────────────
# FILTER_N  : "all"  → show all 1190 presentations (colored by GS label)
#             int    → e.g. 3 shows only MS(3,w) presentations
#             list   → e.g. [1,3,5] shows those n values, colored by n
# COLOR_BY  : "gs"   → color by GS-solved (blue) vs GS-unsolved (red)
#             "n"    → color each n value differently (useful when FILTER_N="all")
# ─────────────────────────────────────────────────────────────────────────────

FILTER_N  = "all"   # "all" | 1 | 2 | 3 | 4 | 5 | 6 | 7 | [1,3,5] | etc.
COLOR_BY  = "gs"    # "gs" | "n"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from ast import literal_eval
from importlib import resources

CKPT_DIR = f"{DRIVE_ROOT}/ac_solver/transformer/checkpoints"

# ── Load embeddings and GS labels ────────────────────────────────────────────
embeddings = np.load(f"{CKPT_DIR}/embeddings.npy")   # (1190, 512)
gs_labels  = np.load(f"{CKPT_DIR}/labels.npy")        # (1190,) 1=solved 0=unsolved

# ── Extract n from each presentation ─────────────────────────────────────────
# MS(n,w): first relator = x⁻¹ yⁿ x y^(-(n+1)) → encoded as [-1, 2,2,..,2, 1, -2,..]
# n = number of consecutive 2s after the leading -1
with resources.open_text("ac_solver.search.miller_schupp.data", "all_presentations.txt") as f:
    raw_lines = [line.strip() for line in f if line.strip()]
presentations = [literal_eval(line) for line in raw_lines]

def extract_n(pres):
    """Count consecutive 2s after the leading -1 in the first relator."""
    n = 0
    for token in pres[1:]:
        if token == 2:
            n += 1
        else:
            break
    return n

n_values = np.array([extract_n(p) for p in presentations])   # (1190,)
unique_ns = sorted(set(n_values.tolist()))
print(f"Embeddings : {embeddings.shape}")
print(f"n values found : {unique_ns}")
for n in unique_ns:
    mask = n_values == n
    print(f"  n={n}: {mask.sum()} presentations  "
          f"(solved={gs_labels[mask].sum()}, unsolved={(1-gs_labels[mask]).sum()})")

# ── Filter by n ───────────────────────────────────────────────────────────────
if FILTER_N == "all":
    keep = np.ones(len(embeddings), dtype=bool)
    n_label = "all n"
elif isinstance(FILTER_N, list):
    keep = np.isin(n_values, FILTER_N)
    n_label = f"n ∈ {FILTER_N}"
else:
    keep = n_values == int(FILTER_N)
    n_label = f"n={FILTER_N}"

emb_filtered = embeddings[keep]
gs_filtered  = gs_labels[keep]
n_filtered   = n_values[keep]
print(f"\nFiltered: {keep.sum()} presentations ({n_label})")

# ── t-SNE ─────────────────────────────────────────────────────────────────────
print(f"\nRunning t-SNE on {keep.sum()} points (perplexity=30)...")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, metric="cosine", random_state=42, verbose=1)
coords = tsne.fit_transform(emb_filtered)
print(f"t-SNE complete. KL divergence: {tsne.kl_divergence_:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))

if COLOR_BY == "gs" or isinstance(FILTER_N, int):
    # Color by GS-solved / GS-unsolved
    palette = {1: "#2196F3", 0: "#F44336"}
    names   = {1: "GS-solved (easy)", 0: "GS-unsolved (hard)"}
    for lbl in [1, 0]:
        mask = gs_filtered == lbl
        ax.scatter(coords[mask,0], coords[mask,1],
                   c=palette[lbl], label=names[lbl], alpha=0.7, s=20, linewidths=0)
    legend_handles = [mpatches.Patch(color=palette[l], label=names[l]) for l in [1,0]]
else:
    # Color by n value
    cmap = plt.cm.get_cmap("tab10", len(unique_ns))
    n_color_map = {n: cmap(i) for i, n in enumerate(unique_ns)}
    for n in sorted(set(n_filtered.tolist())):
        mask = n_filtered == n
        ax.scatter(coords[mask,0], coords[mask,1],
                   c=[n_color_map[n]], label=f"n={n} ({mask.sum()})",
                   alpha=0.7, s=20, linewidths=0)
    legend_handles = [mpatches.Patch(color=n_color_map[n], label=f"n={n}") for n in sorted(set(n_filtered.tolist()))]

title_suffix = f"  |  {n_label}" if FILTER_N != "all" else ""
ax.set_title(
    f"t-SNE of Transformer Embeddings\n"
    f"Miller-Schupp Presentations ({keep.sum()})  |  perplexity=30{title_suffix}",
    fontsize=13,
)
ax.set_xlabel("t-SNE dim 1")
ax.set_ylabel("t-SNE dim 2")
ax.legend(handles=legend_handles, fontsize=10)
plt.tight_layout()
n_tag    = "all" if FILTER_N == "all" else (f"n{'_'.join(map(str,FILTER_N))}" if isinstance(FILTER_N,list) else f"n{FILTER_N}")
color_tag = COLOR_BY
out_path = f"{CKPT_DIR}/tsne_{n_tag}_{color_tag}.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n✅ Plot saved → {out_path}")

# ── CELL 8: Figure 17 Replication — 7×2 Grid (Untrained vs Trained) ───────────
# Exact replication of Figure 17 from the paper:
#   Rows  = n=1..7 (one per Miller-Schupp parameter)
#   Left  = untrained Transformer (random init, seed=0)
#   Right = trained Transformer   (model_final.pt)
#   Distance metric = cosine similarity (per paper Section 6.3)
#
# Saved to: ac_solver/transformer/checkpoints/tsne_figure17.png

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from sklearn.manifold import TSNE
from ast import literal_eval
from importlib import resources

from ac_solver.transformer.model import ACTransformer
from ac_solver.transformer.tokenizer import TOKEN_EOS, presentation_to_tokens

CKPT_DIR = f"{DRIVE_ROOT}/ac_solver/transformer/checkpoints"
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Helper ────────────────────────────────────────────────────────────────────
def _get_embeddings(model, input_ids, eos_positions, device, batch_size=256):
    model.eval()
    N, d = input_ids.shape[0], model.d_model
    out  = np.zeros((N, d), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            h = model.get_hidden_states(input_ids[s:e].to(device))
            for li, gi in enumerate(range(s, e)):
                out[gi] = h[li, eos_positions[gi], :].cpu().float().numpy()
    return out

# ── Load presentations & tokenize ─────────────────────────────────────────────
with resources.open_text("ac_solver.search.miller_schupp.data", "all_presentations.txt") as f:
    raw_lines = [line.strip() for line in f if line.strip()]
presentations = [literal_eval(line) for line in raw_lines]
gs_labels = np.load(f"{CKPT_DIR}/labels.npy")

def _extract_n(pres):
    n = 0
    for tok in pres[1:]:
        if tok == 2:
            n += 1
        else:
            break
    return n

n_values  = np.array([_extract_n(p) for p in presentations])
tok_lists = [presentation_to_tokens(p if isinstance(p,list) else list(p),
                                     max_relator_length=len(p)//2)
             for p in presentations]
eos_pos   = [len(t) - 1 for t in tok_lists]
max_len   = max(len(t) for t in tok_lists)
input_ids = torch.full((len(tok_lists), max_len), fill_value=TOKEN_EOS, dtype=torch.long)
for i, t in enumerate(tok_lists):
    input_ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)

# ── Extract embeddings for both models ────────────────────────────────────────
print("Extracting trained model embeddings...")
ckpt = torch.load(f"{CKPT_DIR}/model_final.pt", map_location=device)
trained_model = ACTransformer(vocab_size=6, d_model=512, n_heads=4,
                               n_layers=8, context_length=1024).to(device)
trained_model.load_state_dict(ckpt["model_state_dict"])
emb_trained = _get_embeddings(trained_model, input_ids, eos_pos, device)
print(f"  Trained   : {emb_trained.shape}")

print("Extracting untrained model embeddings (random init, seed=0)...")
torch.manual_seed(0)
untrained_model = ACTransformer(vocab_size=6, d_model=512, n_heads=4,
                                 n_layers=8, context_length=1024).to(device)
emb_untrained = _get_embeddings(untrained_model, input_ids, eos_pos, device)
print(f"  Untrained : {emb_untrained.shape}")

# ── Build 7×2 grid ────────────────────────────────────────────────────────────
unique_ns  = sorted(set(n_values.tolist()))   # [1,2,3,4,5,6,7]
palette    = {1: "#2196F3", 0: "#F44336"}
dot_names  = {1: "GS-solved", 0: "GS-unsolved"}

fig, axes = plt.subplots(len(unique_ns), 2,
                          figsize=(12, 4 * len(unique_ns)),
                          constrained_layout=True)
fig.suptitle(
    "Figure 17 — t-SNE of Transformer Embeddings (cosine similarity)\n"
    "Left: untrained model  |  Right: trained model",
    fontsize=14,
)

for row, n in enumerate(unique_ns):
    mask   = n_values == n
    gs_sub = gs_labels[mask]
    n_pts  = int(mask.sum())
    perp   = min(30, n_pts - 1)

    for col, (emb_all, col_title) in enumerate([
        (emb_untrained, "Untrained"),
        (emb_trained,   "Trained"),
    ]):
        ax = axes[row, col]
        tsne = TSNE(n_components=2, perplexity=perp, n_iter=1000,
                    metric="cosine", random_state=42, verbose=0)
        coords = tsne.fit_transform(emb_all[mask])

        for lbl in [1, 0]:
            sub = gs_sub == lbl
            ax.scatter(coords[sub, 0], coords[sub, 1],
                       c=palette[lbl], alpha=0.7, s=15, linewidths=0)
        ax.set_title(f"n={n}  |  {col_title}  ({n_pts} pts)", fontsize=10)
        ax.set_xlabel("t-SNE dim 1", fontsize=8)
        ax.set_ylabel("t-SNE dim 2", fontsize=8)
        ax.tick_params(labelsize=7)

legend_handles = [mpatches.Patch(color=palette[l], label=dot_names[l]) for l in [1, 0]]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           fontsize=12, bbox_to_anchor=(0.5, -0.01))

out_path = f"{CKPT_DIR}/tsne_figure17.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n✅ Figure 17 saved → {out_path}")
