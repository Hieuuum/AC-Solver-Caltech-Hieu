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
