# Step 0: Codebase Understanding Report — AC-Solver-Caltech-Hieu
> **Revision 2** — Written from scratch after hands-on file-by-file inspection of every source file.
> Supersedes the placeholder `docs/step0-codebase-report.md` that incorrectly stated the transformer was absent.

---

## Executive Summary

The repository contains **two fully implemented, but disconnected AI systems**:

1. **PPO Agent** — `ac_solver/agents/` — a CleanRL-style PPO loop with a 2-layer FFN actor/critic, ready to run.
2. **Transformer Language Model** — `ac_solver/transformer/` — an 8-layer decoder-only GPT-style model with **all weights already trained** (`model_final.pt`, 3 epochs on 1.83M presentations), **EOS embeddings already extracted** (`embeddings.npy`, shape `(1190, 512)`), and a **hardness oracle already present** (`hardness_oracle.pt`).

Connecting these two systems is the entire goal of **Proposal B**. The good news: both pipelines are complete and working. The only work is integration.

---

## 1. Repository Map

```
AC-Solver-Caltech-Hieu/
├── ac_solver/                          # Main Python package
│   ├── __init__.py                     # Exports: ACEnv, ACEnvConfig, bfs, greedy_search, train_ppo
│   ├── envs/
│   │   ├── ac_env.py                   # ACEnv (Gymnasium Env), ACEnvConfig (dataclass)
│   │   ├── ac_moves.py                 # concatenate_relators, conjugate, ACMove dispatcher (12 moves)
│   │   └── utils.py                    # is_valid, is_trivial, simplify_relator, convert/change_max_length
│   ├── agents/
│   │   ├── ppo.py                      # Entry point: parse_args → get_env → Agent → ppo_training_loop
│   │   ├── ppo_agent.py                # Agent(nn.Module): actor + critic FFNs, build_network helper
│   │   ├── training.py                 # ppo_training_loop: rollout → GAE → clipped PPO update (411 lines)
│   │   ├── environment.py              # make_env, get_env (SyncVectorEnv factory)
│   │   ├── args.py                     # All CLI hyperparameters via argparse
│   │   └── utils.py                    # load_initial_states_from_text_file
│   ├── transformer/
│   │   ├── model.py                    # ACTransformer (decoder-only GPT), CausalSelfAttention, MLP block
│   │   ├── tokenizer.py                # presentation_to_tokens, tokens_to_presentation (6-token vocab)
│   │   ├── data_generator.py           # Algorithm 6 (Appendix D): generates ~1.8M presentations
│   │   ├── train_lm.py                 # LM training loop: packed sequences, BF16, torch.compile
│   │   ├── extract_embeddings.py       # Extracts EOS-token hidden states → embeddings.npy
│   │   ├── train_oracle.py             # Hardness oracle: 2-layer MLP on EOS embeddings, 5-fold CV
│   │   ├── prepare_dataset.py          # Dataset preparation utilities
│   │   ├── download_dataset.py         # HuggingFace dataset download helper
│   │   ├── upload_dataset.py           # HuggingFace dataset upload helper
│   │   └── checkpoints/               # *** PRE-TRAINED ARTIFACTS ***
│   │       ├── model_final.pt          # Final LM checkpoint (epoch 3, verified loadable)
│   │       ├── ckpt_step0002000.pt     # Intermediate checkpoint at step 2,000
│   │       ├── ckpt_step0004000.pt     # Intermediate checkpoint at step 4,000
│   │       ├── embeddings.npy          # Pre-extracted EOS embeddings, shape (1190, 512), float32
│   │       ├── labels.npy              # GS labels: 1=solved(533), 0=unsolved(657), shape (1190,)
│   │       └── hardness_oracle.pt      # Trained MLP oracle (may need reload workaround)
│   └── search/
│       ├── greedy.py                   # Priority-queue greedy search (min-length heuristic)
│       ├── breadth_first.py            # Standard BFS (known bug in word_lengths at line 64-66)
│       └── miller_schupp/
│           ├── miller_schupp.py        # MS(n,w) presentation generator + batch search runner
│           └── data/
│               ├── all_presentations.txt           # 1190 MS presentations (GS-solved first)
│               ├── greedy_solved_presentations.txt  # 533 GS-solved presentations
│               ├── greedy_search_paths.txt          # 533 solution paths
│               └── __init__.py
├── dataset/
│   └── transformer_ds/
│       ├── config.json                 # {n_presentations:1190, n_phases:128, n_chains:12, n_moves:1000, lmax:128, total_generated:1827840}
│       ├── presentations.npy           # Merged 1.83M presentations, shape (1827840, 256), dtype int8
│       ├── metadata.npy                # Per-presentation metadata (origin_index, phase, chain)
│       ├── progress.json               # Generation progress log
│       └── shards/                     # 12 shards × ~150K presentations each
├── barcode_analysis/                   # C++ topological data analysis (Section 5 of paper)
├── notebooks/                          # 4 Jupyter notebooks
├── tests/                              # pytest tests (10 files covering env, search, PPO)
├── pyproject.toml                      # Poetry project, Python >=3.9,<3.13
├── colab_training.py                   # Google Colab training script
├── RLpath1546.txt                      # Example RL solution path for AK(3)
└── README.md
```

---

## 2. PPO Training Pipeline — End-to-End Trace

### 2a. Entry Point

**`ac_solver/agents/ppo.py` — `train_ppo()` (line 23)**

Call flow:
```
parse_args()
  └─ get_env(args)    [environment.py]
      └─ load_initial_states_from_text_file(states_type)
      └─ normalize all presentations to max_relator_length=36 → 72-dim arrays
      └─ SyncVectorEnv([make_env(presentation, args) for i in range(num_envs)])
  └─ Agent(envs, nodes_counts)    [ppo_agent.py]
  └─ Adam optimizer
  └─ ppo_training_loop(...)    [training.py]
```

### 2b. Observation Encoding

**`ac_solver/agents/environment.py` — `get_env()` (line 60)**

- When `--states-type all` (default): loads all **1190** MS presentations from `all_presentations.txt`
- When `--states-type solved`: loads only **533** GS-solved from `greedy_solved_presentations.txt`
- **Normalizes all to `max_relator_length=36`** (hardcoded line 87) → **72-dim `int8` arrays**
- Token values: `1=x, -1=x⁻¹, 2=y, -2=y⁻¹, 0=padding`

**`ac_solver/envs/ac_env.py` — `ACEnv`**
```python
observation_space = Box(low=-2, high=2, shape=(72,), dtype=int8)
action_space = Discrete(12)
```

The **raw `int8` presentation array** is fed directly as a float tensor — no embedding layer, no normalization.

### 2c. Actor & Critic Networks

**`ac_solver/agents/ppo_agent.py` — `Agent` (line 52)**

```
Critic: 72 → nodes[0] → nodes[1] → 1      (value function, Tanh activations)
Actor:  72 → nodes[0] → nodes[1] → 12     (policy logits, Tanh activations)
```

- **Default `nodes_counts=[256, 256]`** (2 hidden layers). Paper uses `[512, 512]` → pass `--nodes-counts 512 512`
- Orthogonal weight init: `std=sqrt(2)` hidden, `std=1.0` critic output, `std=0.01` actor output
- **Fully separate** actor and critic networks (no shared trunk)
- `get_action_and_value(x)` → Categorical distribution → sample or evaluate action

### 2d. The 12 AC' Moves

**`ac_solver/envs/ac_moves.py` — `ACMove()` (line 159)**

| ID | Move | Type |
|----|------|------|
| 0 | r₁ → r₁ · r₀ | Concatenation |
| 1 | r₀ → r₀ · r₁⁻¹ | Concatenation |
| 2 | r₁ → r₁ · r₀⁻¹ | Concatenation |
| 3 | r₀ → r₀ · r₁ | Concatenation |
| 4 | r₁ → x⁻¹ r₁ x | Conjugation |
| 5 | r₀ → y⁻¹ r₀ y | Conjugation |
| 6 | r₁ → y⁻¹ r₁ y | Conjugation |
| 7 | r₀ → x r₀ x⁻¹ | Conjugation |
| 8 | r₁ → x r₁ x⁻¹ | Conjugation |
| 9 | r₀ → y r₀ y⁻¹ | Conjugation |
| 10 | r₁ → y r₁ y⁻¹ | Conjugation |
| 11 | r₀ → x⁻¹ r₀ x | Conjugation |

Pattern: Even IDs (0,2,4,6,8,10) affect r₁; Odd IDs (1,3,5,7,9,11) affect r₀.
After every move, `simplify_presentation(cyclical=True)` is called. If the result exceeds `max_relator_length`, the presentation is returned **unchanged** (move is a no-op).

### 2e. Reward Function

**`ac_solver/envs/ac_env.py` — `step()` (line 95)**

```python
done = (sum(self.lengths) == 2)          # trivial iff both relators have length 1
reward = self.max_reward * done - sum(self.lengths) * (1 - done)
# max_reward = horizon_length × max_relator_length × n_gen = T × 36 × 2
```

Rewards then divided by `max_reward` (training.py:233) and clipped to `[-10, 1000]` via `TransformReward` wrapper.

**Effective post-normalization rewards:**
- Non-terminal: `−sum_of_relator_lengths / max_reward` ≈ very small negative
- Terminal (trivial): `+1.0`

This matches the paper's `-min(10, length(s_{t+1}))` / `+1000` scheme after normalization.

### 2f. Initial Presentation Sampling

**`ac_solver/agents/training.py` — lines 199–224**

**Data ordering (verified):** `all_presentations.txt` has the **533 GS-solved first** (indices 0–532), then **657 GS-unsolved** (indices 533–1189).

**Round 1 (first complete pass):** Sequential round-robin — each env gets the next unprocessed index. Every presentation is visited exactly once.

**After Round 1 (steady-state):** On each episode end for environment `i`:
```python
if len(success_record["solved"]) == 0 or (
    success_record["unsolved"]
    and random.uniform(0, 1) > args.repeat_solved_prob  # default 0.25
):
    curr_states[i] = random.choice(list(success_record["unsolved"]))   # 75% probability
else:
    curr_states[i] = random.choice(list(success_record["solved"]))     # 25% probability
```

The `success_record["solved"]` set is **built dynamically** during training, not pre-loaded. This 75/25 split matches the paper.

### 2g. Horizon Length

**Default:** `--horizon-length 2000`
**Paper:** T=200 (constant) or T∈{200,400,800,1200} (variable — not implemented in codebase).

### 2h. Training Loop

**`ac_solver/agents/training.py` — `ppo_training_loop()` (line 68)**

CleanRL-style PPO:
1. Rollout phase: `num_steps` × `num_envs` parallel envs
2. GAE: `γ=0.99` (paper: 0.999), `λ=0.95`
3. PPO update: `update_epochs=1`, `clip_coef=0.2`, `minibatches=4`
4. LR schedule: linear decay (or cosine) from `lr` to `min_lr_frac × lr`
5. Checkpoints every 100 updates to `out/<run_name>/ckpt.pt`
6. Optional W&B logging

---

## 3. Transformer Pipeline — Fully Implemented

> **Critical finding:** The placeholder report incorrectly stated the transformer was absent. It is **fully implemented, trained, and has pre-extracted embeddings ready to use.**

### 3a. Architecture

**`ac_solver/transformer/model.py` — `ACTransformer` (line 91)**

```
8-layer decoder-only transformer
  vocab_size     = 6   (x, x⁻¹, y, y⁻¹, SEP, EOS)
  d_model        = 512
  n_heads        = 4   (head_dim = 128)
  context_length = 1024
  n_layers       = 8
  Parameters     = 25,710,592  (verified)
  Tied embeddings: self.unembed.weight = self.token_emb.weight  (W_E = W_U^T)
```

Pre-norm blocks: `LayerNorm → CausalSelfAttention → residual → LayerNorm → 4×MLP(GELU) → residual`

### 3b. Tokenization

**`ac_solver/transformer/tokenizer.py`**

```
TOKEN_X     = 0  (PPO: 1)     x
TOKEN_X_INV = 1  (PPO: -1)    x⁻¹
TOKEN_Y     = 2  (PPO: 2)     y
TOKEN_Y_INV = 3  (PPO: -2)    y⁻¹
TOKEN_SEP   = 4               separator between r₀ and r₁
TOKEN_EOS   = 5               end of sequence
```

Token sequence: `[r₀_tokens..., SEP, r₁_tokens..., EOS]`
Zero-padded positions are **stripped** before tokenization — sequences have variable length.

Key API:
- `presentation_to_tokens(presentation, max_relator_length=None) → list[int]`
- `tokens_to_presentation(tokens, max_relator_length) → np.ndarray`

### 3c. Training Data Generation

**`ac_solver/transformer/data_generator.py`** — Implements Algorithm 6 (Appendix D)

For each of the 1190 MS presentations P₀:
- Run 128 phases × 12 parallel chains
- Each chain: apply 1000 random AC' moves within a gradually increasing length bound
- Save each resulting presentation

Already executed with: `n_phases=128, n_chains=12, n_moves=1000, lmax=128, seed=42`
**Total generated: 1,827,840 presentations** stored in `dataset/transformer_ds/`

### 3d. Embedding Extraction Method

**`ac_solver/transformer/extract_embeddings.py`**

**Method: EOS-token hidden state (after final LayerNorm, before unembedding)**

```python
tokens = presentation_to_tokens(pres)   # [r₀..., SEP, r₁..., EOS]
# EOS is always the LAST token in the sequence
hidden = model.get_hidden_states(input_ids)   # (B, T, 512)
embedding = hidden[i, eos_position, :]        # (512,) — the EOS hidden state
```

`model.get_hidden_states()` (line 162) returns `ln_f(last_block_output)` — post-final-LayerNorm, pre-unembed.

**Pre-extracted for all 1190 MS presentations:**
- `checkpoints/embeddings.npy`: shape `(1190, 512)`, dtype `float32` ✓
- `checkpoints/labels.npy`: shape `(1190,)`, dtype `int32`; `1=GS-solved, 0=GS-unsolved` ✓
- Verified counts: 533 solved + 657 unsolved = 1190 ✓

### 3e. Hardness Oracle

**`ac_solver/transformer/train_oracle.py` — `HardnessOracle`**

```
Linear(512, 128) → ReLU → Linear(128, 1)
Output: raw logit; sigmoid > 0.5 = "GS-solved"
```

5-fold stratified CV on the 1190 embeddings. Final model trained on full dataset and saved to `hardness_oracle.pt`.

**Known issue:** `hardness_oracle.pt` fails to load in the conda env (`numpy._core` not found in numpy 1.24.3). **Workaround:** re-run `train_oracle.py` — takes ~1 minute since embeddings are already computed.

### 3f. Checkpoint Status

| Artifact | File | Status |
|----------|------|--------|
| LM weights (final) | `checkpoints/model_final.pt` | ✓ Present (epoch 3) |
| EOS embeddings | `checkpoints/embeddings.npy` | ✓ **Fully usable**, shape (1190,512) |
| GS labels | `checkpoints/labels.npy` | ✓ **Fully usable** |
| Hardness oracle | `checkpoints/hardness_oracle.pt` | ⚠ Needs re-training in current env |

---

## 4. Dataset Inventory

| Source | Count | Format | Notes |
|--------|-------|--------|-------|
| `all_presentations.txt` | **1190** | Python list literals | GS-solved first (0–532), then unsolved |
| `greedy_solved_presentations.txt` | **533** | Same | Exact match with first 533 of all_presentations.txt |
| `greedy_search_paths.txt` | **533** | `[(action_id, total_len), ...]` | First entry = initial state; path_len = `len - 1` |
| `embeddings.npy` | **1190** × 512 | float32 | Pre-computed EOS embeddings, ready to use |
| Transformer dataset | **1,827,840** | int8 (256-dim) | 12 shards in `dataset/transformer_ds/shards/` |

**PPO-solved labels:** Not stored. Paper's 431/535 numbers come from W&B runs. Must re-run to reproduce.

---

## 5. Dependencies & Environment Setup

**Active conda environment:** `ac-solver`

```bash
conda activate ac-solver
# OR without activating:
conda run -n ac-solver python ...
```

| Package | Version |
|---------|---------|
| Python | 3.9 |
| PyTorch | **2.0.1+cu117** |
| Gymnasium | 0.28.1 |
| NumPy | 1.24.3 |
| scikit-learn | 1.3.1 |
| wandb | 0.15.3 |

**Verified import test (all pass):**
```bash
conda run -n ac-solver python -c "
import torch; print(torch.__version__)        # 2.0.1+cu117
from ac_solver.transformer.model import ACTransformer
m = ACTransformer(); print(sum(p.numel() for p in set(m.parameters())))  # 25710592
import numpy as np
np.load('ac_solver/transformer/checkpoints/embeddings.npy')  # (1190, 512) ✓
"
```

---

## 6. Integration Points for Proposal B

### What Already Exists (No Implementation Needed)

| Component | Status |
|-----------|--------|
| Transformer model code | ✓ Complete |
| Pre-trained LM weights | ✓ Available |
| Tokenizer | ✓ Complete |
| Training data (1.83M pres.) | ✓ Available |
| EOS embeddings (1190×512) | ✓ Available |
| GS labels | ✓ Available |
| Hardness oracle code | ✓ Complete (needs re-training) |
| PPO full pipeline | ✓ Complete |
| 12 AC' moves | ✓ Complete |
| 1190 MS presentations | ✓ Available |

### Approach A — Curriculum Sampler

**Injection point:** `ac_solver/agents/training.py`, lines 207–224 (the after-Round-1 sampling block)

**Replace** the current `random.choice(unsolved/solved)` logic with a `CurriculumSampler` that:
1. Pre-loads `embeddings.npy` and re-trains (or loads) the hardness oracle
2. Assigns a difficulty score `H(P)` to each of the 1190 presentations
3. Maintains a dynamic frontier window that expands based on training progress

**Secondary point:** `environment.py` lines 75–101 — reorder `initial_states` by `H` before Round 1.

**New file:** `ac_solver/agents/curriculum_sampler.py`

---

### Approach B — Auxiliary Reward Shaping (+ΔH)

**Injection point:** Gymnasium wrapper in `make_env()`, `environment.py` lines 32–56, **after** `ACEnv` is created.

```python
def thunk():
    env = ACEnv(env_config)
    env = HardnessRewardWrapper(env, oracle=oracle, transformer=model, alpha=args.reward_alpha)
    if args.norm_rewards: env = NormalizeReward(env, gamma=args.gamma)
    if args.clip_rewards: env = TransformReward(env, lambda r: np.clip(r, ...))
    return env
```

Wrapper adds `alpha * (H(s_t) - H(s_{t+1}))` to the reward each step.

**Key design decision:** Computing embedding per step is expensive. Options:
- Pre-cache embeddings for seed presentations only (fast, limited)
- Batch compute at rollout level (feasible, general)

**New file:** `ac_solver/envs/transformer_reward_wrapper.py`

---

### Approach C — Enriched State Representation

**Injection point:** Gymnasium ObservationWrapper in `make_env()`, `environment.py` lines 32–56, after `ACEnv`.

```python
def thunk():
    env = ACEnv(env_config)
    env = TransformerEmbeddingWrapper(env, transformer=model, device=device)
    ...
    return env
```

The `TransformerEmbeddingWrapper(gym.ObservationWrapper)`:
- Widens `observation_space` from `Box(72,)` to `Box(72+512,) = Box(584,)`
- On `reset()` and `step()`: computes transformer embedding of current state, concatenates to 72-dim obs

**Agent auto-adapts:** `ppo_agent.py:72` reads `input_dim = np.prod(envs.single_observation_space.shape)` → FFN input widens from 72 to **584** with zero code changes to the Agent.

**New file:** `ac_solver/envs/transformer_obs_wrapper.py`

---

### Files That Must NOT Change

| File | Reason |
|------|--------|
| `ac_solver/envs/ac_moves.py` | Mathematically precise AC' moves |
| `ac_solver/envs/utils.py` | Presentation validation |
| `ac_solver/search/` | Reference search algorithms |
| `tests/` | Regression test baseline |

---

## 7. Paper vs. Code Hyperparameter Discrepancies

| Parameter | Paper | Code Default | CLI Flag |
|-----------|-------|-------------|---------|
| Parallel actors | 28 | `num_envs=4` | `--num-envs 28` |
| Hidden units | 512×2 | `[256,256]` | `--nodes-counts 512 512` |
| Horizon T | 200 | `2000` | `--horizon-length 200` |
| γ | 0.999 | 0.99 | `--gamma 0.999` |
| lr | 1e-4 | 2.5e-4 | `--learning-rate 1e-4` |
| Rollout length | 200 | 2000 | `--num-steps 200` |
| Variable horizon | 200→400→800→1200 | Not implemented | Needs custom scheduler |

---

## 8. Important Implementation Notes

1. **The hardness oracle must be re-trained** in the current conda env before any integration. Takes ~1 minute: `conda run -n ac-solver python -m ac_solver.transformer.train_oracle`.

2. **For Approach C, the observation dtype changes** from `int8` to `float32` (can't mix in one array). The `TransformerEmbeddingWrapper` should explicitly cast the PPO obs to float32 before concatenating.

3. **For online embedding in Approaches B/C:** The transformer has 25.7M parameters and runs on GPU. For 28 parallel envs × 200 rollout steps = 5,600 forward passes per update, batching is essential. A practical pattern: collect all current observations, batch-forward through the transformer once, cache the results for the rollout.

4. **The EOS embedding is the correct embedding choice**, as confirmed by the paper's t-SNE visualization (Figure 17) demonstrating GS-solved/unsolved clustering. The `model.get_hidden_states()` method makes this easy to extract.

5. **The hardness oracle's scaler.mean_ and scaler.scale_** will need to be saved/loaded alongside the MLP weights for correct inference on new embeddings.
