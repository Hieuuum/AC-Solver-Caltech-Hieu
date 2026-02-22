# Step 0: Codebase Understanding Report — AC-Solver

## Context

This report documents a thorough, code-verified exploration of the AC-Solver repository at `/home/user/AC-Solver-Caltech-Hieu`. The goal is to fully understand the codebase before implementing Proposal B: Transformer-Guided RL for AC Trivialization, which connects the paper's disconnected PPO agent and transformer language model.

---

## 1. Repository Map

```
AC-Solver-Caltech-Hieu/
├── ac_solver/                          # Main Python package
│   ├── __init__.py                     # Exports: ACEnv, ACEnvConfig, bfs, greedy_search, train_ppo
│   ├── envs/
│   │   ├── ac_env.py                   # ACEnv (Gymnasium Env), ACEnvConfig (dataclass)
│   │   ├── ac_moves.py                 # concatenate_relators, conjugate, ACMove dispatcher
│   │   └── utils.py                    # is_valid, is_trivial, simplify_relator, convert/change_max_length
│   ├── agents/
│   │   ├── ppo.py                      # Entry point: parse → get_env → Agent → ppo_training_loop
│   │   ├── ppo_agent.py                # Agent(nn.Module): actor + critic FFNs
│   │   ├── training.py                 # ppo_training_loop: rollout → GAE → clipped PPO update
│   │   ├── environment.py              # make_env, get_env (SyncVectorEnv factory)
│   │   ├── args.py                     # All CLI hyperparameters via argparse
│   │   └── utils.py                    # load_initial_states_from_text_file
│   └── search/
│       ├── greedy.py                   # greedy_search (priority queue, min-length heuristic)
│       ├── breadth_first.py            # bfs (standard BFS)
│       └── miller_schupp/
│           ├── miller_schupp.py        # MS presentation generator + batch search runner
│           └── data/
│               ├── all_presentations.txt           # 1190 MS presentations
│               ├── greedy_solved_presentations.txt  # 533 GS-solved
│               ├── greedy_search_paths.txt          # 533 solution paths
│               └── bfs_solved_presentations.txt     # 278 BFS-solved
├── barcode_analysis/                   # C++ code for topological data analysis (Section 5)
│   ├── 5_steps_neibourhoods/           # Barcode feature computation
│   └── simplex_data_generation/        # AC-graph simplex data generation
├── notebooks/
│   ├── Checking-AC-Paths.ipynb
│   ├── Classical-Search-and-PPO-in-AC-Environment.ipynb
│   ├── Scaling-PPO-in-AC-Environment.ipynb
│   └── Stable-AK3.ipynb
├── tests/                              # pytest tests for env, search, PPO components
├── pyproject.toml                      # Poetry project, Python >=3.9,<3.12
├── RLpath1546.txt                      # Example RL solution path (1546 actions)
└── README.md
```

---

## 2. PPO Training Pipeline — End-to-End Trace

### 2a. Entry Point

**File:** `ac_solver/agents/ppo.py` — `train_ppo()` (line 23)

Flow: `parse_args()` → `get_env()` → `Agent()` → `ppo_training_loop()`

### 2b. Environment & Observations

**File:** `ac_solver/agents/environment.py` — `get_env()` (line 60)

- Loads all 1190 MS presentations via `load_initial_states_from_text_file(states_type="all")` from `ac_solver/agents/utils.py:10`
- **Normalizes all presentations to `max_relator_length=36`** (hardcoded at line 87), yielding **72-dimensional `int8` arrays** (2 relators × 36 elements each)
- Native data has variable lengths: 36 (n=1, max_rel=18), 40 (n=2, max_rel=20), ... up to 72 (n=7, max_rel=36), but all are zero-padded to 36 per relator
- Tokens: `1=x, -1=x⁻¹, 2=y, -2=y⁻¹, 0=padding`
- Creates `SyncVectorEnv` with `num_envs` parallel copies (default 4, paper uses 28)

**File:** `ac_solver/envs/ac_env.py` — `ACEnv` (line 56)
- `observation_space = Box(-2, 2, shape=(72,), dtype=int8)`
- `action_space = Discrete(12)`

### 2c. Actor & Critic Networks

**File:** `ac_solver/agents/ppo_agent.py` — `Agent` (line 52)

```
Critic: input_dim → nodes[0] → nodes[1] → 1
Actor:  input_dim → nodes[0] → nodes[1] → 12
```

- Default `nodes_counts=[256, 256]`; paper uses `[512, 512]` (pass `--nodes-counts 512 512`)
- `input_dim = 72` (from observation space)
- `tanh` activations between all layers
- Orthogonal weight init (std=sqrt(2) for hidden, 1.0 for critic output, 0.01 for actor output)
- No shared trunk — actor and critic are fully separate `nn.Sequential` modules
- The raw int8 presentation array is fed directly as a float tensor — **no embedding layer, no normalization**

### 2d. 12 AC' Moves

**File:** `ac_solver/envs/ac_moves.py` — `ACMove()` (line 159)

The complete mapping (from the docstring, verified against the code logic):

| ID | Move | Type |
|----|------|------|
| 0 | r₁ → r₁ r₀ | Concatenation |
| 1 | r₀ → r₀ r₁⁻¹ | Concatenation |
| 2 | r₁ → r₁ r₀⁻¹ | Concatenation |
| 3 | r₀ → r₀ r₁ | Concatenation |
| 4 | r₁ → x⁻¹ r₁ x | Conjugation |
| 5 | r₀ → y⁻¹ r₀ y | Conjugation |
| 6 | r₁ → y⁻¹ r₁ y | Conjugation |
| 7 | r₀ → x r₀ x⁻¹ | Conjugation |
| 8 | r₁ → x r₁ x⁻¹ | Conjugation |
| 9 | r₀ → y r₀ y⁻¹ | Conjugation |
| 10 | r₁ → y r₁ y⁻¹ | Conjugation |
| 11 | r₀ → x⁻¹ r₀ x | Conjugation |

**Pattern:** Even IDs (0,2,4,6,8,10) affect r₁; Odd IDs (1,3,5,7,9,11) affect r₀.

After each move, `simplify_presentation()` is called with `cyclical=True` (default), which applies both free and cyclic reduction. If the result would exceed `max_relator_length`, the presentation is returned unchanged.

### 2e. Reward Function

**File:** `ac_solver/envs/ac_env.py` — `step()` (line 95)

```python
done = sum(self.lengths) == 2     # trivial iff both relators have length 1
reward = self.max_reward * done - sum(self.lengths) * (1 - done)
# where max_reward = horizon_length * max_relator_length * n_gen = T * 36 * 2
```

Then in the training loop (`training.py:233`), rewards are divided by `max_reward`, and via `TransformReward` wrapper they are clipped to `[-10, 1000]`.

**Effective reward after normalization:**
- Non-terminal: `−sum_of_relator_lengths / max_reward` (very small negative)
- Terminal: `+1.0` (after rescaling)

### 2f. Initial Presentation Sampling

**File:** `ac_solver/agents/training.py` — lines 199-224

**Verified data ordering:** `all_presentations.txt` has the 533 GS-solved presentations first (indices 0–532), then the 657 GS-unsolved (indices 533–1189). Confirmed by comparing first 533 lines of `all_presentations.txt` with `greedy_solved_presentations.txt` — exact match.

**Round 1 (first pass, lines 207-208):** Round-robin — each env sequentially gets the next unprocessed index. Every presentation is visited exactly once.

**After Round 1 (lines 210-221):** On each episode end for environment `i`:
- If no PPO-solved presentations yet, OR with probability `1 - repeat_solved_prob` (= 0.75): sample uniformly from PPO-unsolved set
- With probability `repeat_solved_prob` (= 0.25): sample uniformly from PPO-solved set

The "PPO-solved" set is built dynamically during training (not pre-loaded). The 75/25 split matches the paper's stated 3/4 unsolved, 1/4 solved ratio.

### 2g. Horizon Length

**File:** `ac_solver/agents/args.py` — line 100

Default: `--horizon-length 2000` (paper uses 200 for constant-horizon; variable 200→400→800→1200 for scaling experiments). **No variable-horizon scheduling logic exists in the codebase** — it would need to be implemented.

### 2h. Training Loop

**File:** `ac_solver/agents/training.py` — `ppo_training_loop()` (line 68)

Standard CleanRL-style single-file PPO:
- Collects `num_steps` steps across `num_envs` parallel envs → `batch_size = num_steps × num_envs`
- GAE with `γ=0.99, λ=0.95` (paper uses `γ=0.999`)
- `update_epochs=1` epoch per rollout
- `clip_coef=0.2` (ε=0.2 in paper)
- Linear LR decay from `lr` to `min_lr_frac × lr` (default: decay to 0)
- Also supports cosine LR decay and warmup
- Supports both clipped loss (default) and KL-penalty loss
- Checkpoints every 100 updates to `out/<run_name>/ckpt.pt` (line 384-408)
- Logs to W&B when `--wandb-log` is set

---

## 3. Transformer Pipeline

### Critical Finding: THE TRANSFORMER IS NOT IN THIS REPOSITORY

Confirmed via:
- `grep -ri "transformer|attention|d_model|decoder|embedding|tokeniz"` across all `.py` files → **zero results**
- No `.pt` or `.pth` model weight files anywhere in the repo
- No transformer architecture code in any file or notebook
- No data generation scripts for the 1.8M training presentations

**What needs to be built from scratch:**
1. Transformer model architecture (8-layer decoder-only, d_model=512, 4 heads, context window 1024, tied embeddings)
2. Tokenization scheme (6 tokens: x, y, x⁻¹, y⁻¹, separator, EOS)
3. Training data generation (Algorithm 6 from Appendix D: apply random AC moves to the 1190 MS presentations to generate ~1.8M AC-equivalent presentations)
4. Training pipeline for next-token prediction
5. Embedding extraction (EOS token's final-layer hidden state — consistent with t-SNE analysis in Figure 17)

### Pre-trained Weights: NOT INCLUDED — must train from scratch

### Tokenization Mapping for the Transformer
The PPO environment uses `{1=x, -1=x⁻¹, 2=y, -2=y⁻¹, 0=padding}`. The transformer needs a different tokenization:
```
PPO encoding → Transformer token
  1 (x)      → token 0
 -1 (x⁻¹)    → token 1
  2 (y)      → token 2
 -2 (y⁻¹)    → token 3
 separator   → token 4  (between r₁ and r₂)
 EOS         → token 5
```

---

## 4. Dataset

| File | Count | Verified | Notes |
|------|-------|----------|-------|
| `data/all_presentations.txt` | **1190** | Yes | MS series n≤7, len(w)≤7. Ordered: GS-solved first (0-532), then GS-unsolved (533-1189) |
| `data/greedy_solved_presentations.txt` | **533** | Yes | Exact match with first 533 lines of all_presentations.txt |
| `data/greedy_search_paths.txt` | **533** | Yes | Format: `[(action_id, total_length), ...]` — path length = `len(path) - 1` (first entry is initial state with action=-1) |
| `data/bfs_solved_presentations.txt` | **278** | Yes | BFS-solved subset (≤1M nodes explored) |

**Native presentation sizes (before normalization):**
- 36 elements (max_rel=18, n=1): 340 presentations
- 40 elements (max_rel=20, n=2): 170 presentations
- 48 elements (max_rel=24, n=3): 170 presentations
- 56 elements (max_rel=28, n=4): 170 presentations
- 64 elements (max_rel=32, n=5): 170 presentations
- 72 elements (max_rel=36, n=6-7): 170 presentations

All normalized to 72 elements (max_relator_length=36) during PPO training.

**GS-unsolved:** 657 presentations = indices 533-1189 of all_presentations.txt. No separate file; derive by exclusion.

**PPO-solved labels:** NOT stored in the repo. The paper's 431/535 numbers come from W&B training logs. Must re-run PPO or access original W&B project.

**Greedy search path lengths:** Extractable from `greedy_search_paths.txt`. Path length = `len(path) - 1` since the first entry `(-1, initial_length)` represents the starting state.

---

## 5. Search Algorithms

### Greedy Search (`ac_solver/search/greedy.py`)
- Priority queue (min-heap) ordered by **total relator length** (sum of both relator lengths)
- Expands the node with minimum total length first
- Terminates when total length = 2 (trivial presentation) or node limit reached
- Default: max 10,000 nodes; MS data used 1,000,000

### BFS (`ac_solver/search/breadth_first.py`)
- Standard breadth-first search using `collections.deque`
- Same termination conditions as greedy
- **Bug at line 64-66:** `word_lengths` is recomputed from `presentation` (the initial presentation) instead of from `state` (current node). This means BFS may use incorrect word lengths. Does not affect correctness of solution detection (which checks `sum(new_word_lengths) == 2`) but may affect move application.

### Miller-Schupp Generator (`ac_solver/search/miller_schupp/miller_schupp.py`)
- `generate_miller_schupp_presentations(n, max_w_len)` generates all MS presentations for fixed n
- MS(n, w) = ⟨x, y | x⁻¹ yⁿ x = yⁿ⁺¹, x = w⟩
- Filters: w must have zero exponent sum on x; duplicates from free/cyclic reduction and cyclic permutation are removed
- `trivialize_miller_schupp_through_search()` applies a search function to a range of (n, w) values

---

## 6. Dependencies & Environment Setup

| Requirement | Version in pyproject.toml | Notes |
|-------------|--------------------------|-------|
| Python | >=3.9, <3.12 | |
| PyTorch | 2.0.1 | |
| Gymnasium | 0.28.1 | |
| NumPy | 1.24.3 | |
| wandb | 0.15.3 | |
| scikit-learn | 1.3.1 | Available for classifier training |
| scipy | 1.13.1 | |
| matplotlib | 3.7.1 | |
| tqdm | 4.65.0 | |

**Setup:** `pip install --no-deps -e .` works (pathtools build fails but isn't needed at runtime).

**Tests:** 8 test files covering environment utils, AC moves, search algorithms, PPO components. Run with `pytest tests/`.

---

## 7. Integration Points for Proposal B

### Phase 1: Hardness Oracle H(P)

**No existing infrastructure** — the transformer, tokenizer, training data generator, and classifier must all be built from scratch.

New files needed:
- `ac_solver/transformer/model.py` — Decoder-only transformer architecture
- `ac_solver/transformer/tokenizer.py` — Presentation → token sequence conversion
- `ac_solver/transformer/data_generator.py` — Algorithm 6: random AC moves to generate 1.8M training presentations
- `ac_solver/transformer/train.py` — Next-token prediction training
- `ac_solver/transformer/embeddings.py` — Extract EOS embeddings for presentations
- `ac_solver/oracle/hardness_oracle.py` — MLP classifier on top of transformer embeddings

### Approach A — Curriculum Sampler

**Primary injection point:** `ac_solver/agents/training.py:207-224`

The current sampling logic after Round 1:
```python
if not round1_complete:
    curr_states[i] = max(states_processed) + 1
else:
    if len(success_record["solved"]) == 0 or (
        success_record["unsolved"]
        and random.uniform(0, 1) > args.repeat_solved_prob
    ):
        curr_states[i] = random.choice(list(success_record["unsolved"]))
    else:
        curr_states[i] = random.choice(list(success_record["solved"]))
```

**Replace with:** A curriculum sampler that selects from a difficulty-sorted subset. Pre-compute `H(P)` for all 1190 presentations before training. Use a schedule to gradually expand the difficulty range.

**Secondary point:** `ac_solver/agents/environment.py:75-101` — reorder `initial_states` by hardness so the Round 1 round-robin pass processes easy presentations first.

### Approach B — Auxiliary Reward Shaping

**Primary injection point:** `ac_solver/envs/ac_env.py:95-113` — the `step()` method.

Add `reward += alpha * (H(prev_state) - H(new_state))` before returning. The oracle `H` must be loaded into the env at initialization.

**Alternative (cleaner):** Compute it in the rollout loop at `ac_solver/agents/training.py:154-160` where `reward` is collected, keeping the base env pure. However, this requires access to previous/current observations to compute H.

**Cleanest approach:** A Gymnasium wrapper around `ACEnv` that intercepts `step()`, computes `ΔH`, and adds the shaped reward.

### Approach C — Enriched State Representation

**Primary injection point:** `ac_solver/envs/ac_env.py:67-74` — the observation space definition.

Change `observation_space` from `Box(shape=(72,))` to `Box(shape=(72 + 512,))` and modify `step()`/`reset()` to append the 512-dim transformer embedding.

**The PPO agent input adjusts automatically:** `ac_solver/agents/ppo_agent.py:72` — `input_dim = np.prod(envs.single_observation_space.shape)` picks up the new dimension, so the FFN input layer widens to 584 with no other code changes.

**Cleanest approach:** A Gymnasium wrapper `TransformerEmbeddingWrapper(ACEnv)` that:
1. Wraps observation space to (72 + 512,)
2. On `reset()` and `step()`, computes the transformer embedding of the current presentation and concatenates it
3. Leaves the base env untouched

### Key Code That Must NOT Change
- `ac_solver/envs/ac_moves.py` — The 12 AC moves are mathematically precise
- `ac_solver/envs/utils.py` — Presentation validation and simplification
- `ac_solver/search/` — Search algorithms (used for validation only)
- `tests/` — All existing tests must continue to pass

---

## 8. Paper-vs-Code Discrepancies

| Paper says | Code default | CLI flag to match |
|-----------|-------------|-------------------|
| 28 parallel actors | `num_envs=4` | `--num-envs 28` |
| 512 hidden units, 2 layers | `nodes_counts=[256,256]` | `--nodes-counts 512 512` |
| Horizon T=200 | `horizon_length=2000` | `--horizon-length 200` |
| γ=0.999 | `gamma=0.99` | `--gamma 0.999` |
| lr=1e-4 | `lr=2.5e-4` | `--learning-rate 1e-4` |
| λ_GAE=0.95 | `gae_lambda=0.95` | Already matches |
| ε=0.2 | `clip_coef=0.2` | Already matches |
| Reward: -min(10, len) / +1000 | Clip wrapper + normalization | `--clip-rewards --min-rew -10 --max-rew 1000` (default) |

**To reproduce paper's constant-horizon baseline:**
```bash
python -m ac_solver.agents.ppo \
  --num-envs 28 \
  --nodes-counts 512 512 \
  --horizon-length 200 \
  --gamma 0.999 \
  --learning-rate 1e-4 \
  --num-steps 200 \
  --total-timesteps <TBD>
```

---

## 9. What Must Be Built (Summary)

### Must build from scratch (not in repo):
1. **Transformer model** — 8-layer decoder-only, d_model=512, 4 heads, context 1024, tied embeddings
2. **Tokenizer** — Map PPO's int encoding to 6-token vocabulary
3. **Training data generator** — Algorithm 6: random AC moves on 1190 presentations → ~1.8M examples
4. **Transformer training pipeline** — Next-token prediction with cross-entropy loss
5. **Embedding extractor** — EOS token's final-layer hidden state
6. **Hardness oracle** — 2-layer MLP classifier on 512-dim embeddings → solvability + difficulty
7. **Curriculum sampler** — Difficulty-sorted sampling for Approach A
8. **Reward shaping wrapper** — ΔH reward for Approach B
9. **State enrichment wrapper** — Concatenate embeddings for Approach C
10. **Variable horizon scheduler** — 200→400→800→1200 schedule (not in codebase)

### Already exists and can be reused:
- Full PPO pipeline (entry point, env, agent, training loop)
- AC environment with 12 moves
- Dataset of 1190 MS presentations with GS labels
- Search algorithms for validation
- Test infrastructure

---

## 10. Verification Plan

1. **Existing tests:** `pytest tests/` — all must pass unchanged
2. **PPO baseline reproduction:** Run with paper's hyperparameters, verify ~431 solved (constant T=200)
3. **Transformer training:** Verify loss convergence, then extract embeddings and check t-SNE clustering of GS-solved vs GS-unsolved (should match Figure 17)
4. **Hardness oracle:** Cross-validated accuracy on 1190 presentations (target: paper's F1=0.962 from Section 8.2)
5. **Each approach:** Compare number of presentations solved vs baseline PPO at same compute budget
