# Phase 1 — Transformer & Hardness Oracle

## Status

| Task | Status | Notes |
|---|---|---|
| Task 0: Codebase exploration | ✅ Done | PPO baseline understood |
| Task 1: Dataset inspection | ✅ Done | 12 shards, 1,827,840 rows, lmax=128 |
| Task 2: Build Transformer model | ✅ Done | `ac_solver/transformer/model.py` — 25.7M params, tied embeddings verified |
| Task 3: Write training loop | ✅ Done | `ac_solver/transformer/train_lm.py` — shard-lazy, collate-padded, cosine LR |
| Task 4: Write extract_embeddings.py | ✅ Done | `ac_solver/transformer/extract_embeddings.py` — verified end-to-end |
| Task 5: Write train_oracle.py | ✅ Done | `ac_solver/transformer/train_oracle.py` — verified end-to-end |
| **Run Task 3 training** | ⬜ Todo | Full 1.8M dataset, ~3 epochs |
| **Run Task 4 extraction** | ⬜ Todo | After checkpoint exists |
| **Run Task 5 oracle** | ⬜ Todo | After embeddings exist |

## Next Steps (in order)

### Immediate (code, no GPU needed)
- [x] Write `ac_solver/transformer/extract_embeddings.py` (Task 4)
- [x] Write `ac_solver/transformer/train_oracle.py` (Task 5)

### Then (GPU job)
- [ ] Launch `python -m ac_solver.transformer.train_lm` on full dataset
- [ ] Monitor training loss; expect it to converge toward ~1.0 (below random init log(6)=1.79)

### After training completes
- [ ] Run `python -m ac_solver.transformer.extract_embeddings` → produces `embeddings.npy` (1190, 512) + `labels.npy` (1190,)
- [ ] Run `python -m ac_solver.transformer.train_oracle` → train MLP, report F1 vs. paper baseline of 0.962

## Dataset Verified
- Shards: 12 (full ~1.8M dataset)
- Rows per shard: 153,600
- Total: 1,827,840 presentations
- lmax: 128, shape per shard: (153,600, 256) int8

---

# Phase 2 — Upload Dataset to Hugging Face

## Status

| Task | Status | Notes |
|---|---|---|
| Task 6: Setup HF account & auth | ✅ Done | `hf auth login` |
| Task 7: Create HF dataset repo | ✅ Done | `hf repo create ac-solver-dataset --repo-type dataset` |
| Task 8: Write upload script | ✅ Done | `ac_solver/transformer/upload_dataset.py` — generator-based, streams 12 shards, push_to_hub with Parquet |
| Task 9: Write dataset card | ✅ Done | `dataset/transformer_ds/README.md` — YAML frontmatter, schema table, usage examples, citation |
| Task 10: Run upload | ✅ Done | Dataset live at `huggingface.co/datasets/mhieuuu/ac-solver-dataset` |
| Task 11: Update GitHub README | ✅ Done | HF badge added to `README.md` |

## Dataset Schema (for HF card)
Each row:
- `presentation`: int8 array of shape (256,) — two relators padded to lmax=128 each
- `pres_idx`: int32 — which of the 1190 Miller-Schupp seed presentations
- `phase`: int32 — MCMC phase index (0–127)
- `chain`: int32 — which Markov chain (0–11)

## Upload Plan Details

### Task 6 — Setup
```bash
pip install huggingface_hub datasets
hf auth login   # enter HF token
```

### Task 7 — Create Repo
```bash
hf repo create ac-solver-dataset --repo-type dataset
```

### Task 8 — Upload Script
- File: `ac_solver/transformer/upload_dataset.py`
- Reads all 12 shards lazily from `dataset/transformer_ds/shards/`
- Merges presentations + metadata columns into a `datasets.Dataset`
- Pushes as Parquet shards to HF Hub (enables web dataset viewer)
- Includes `config.json` as a dataset info attachment

### Task 9 — Dataset Card
- File: `dataset/transformer_ds/README.md`
- Covers: what AC presentations are, generation process, schema, usage example, paper citation

### Task 10 — Run Upload
```bash
python -m ac_solver.transformer.upload_dataset \
    --dataset-dir dataset/transformer_ds \
    --repo-id <your-username>/ac-solver-dataset
```

### Task 11 — Update README
- Add badge: `[![Dataset on HF](https://huggingface.co/datasets/badge.svg)](...)`
- Add one-liner usage: `load_dataset("<your-username>/ac-solver-dataset")`
