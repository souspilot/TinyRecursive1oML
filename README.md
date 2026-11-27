# Validity or diversity: GRAM on puzzles with many solutions

COL864 course project (Harshit Bisht, Ashwendh Choudhary), built on the [GRAM](https://github.com/ahn-ml/gram-release)
code. GRAM makes a tiny recursive reasoning network generative: at every transition it adds random stochastic guidance,
drawn in training from a posterior that sees the target and at test time from a learned prior, with a KL weight β. We
ask what that posterior buys on puzzles with many solutions, against baselines whose stochastic guidance never sees the
target, and what else we tried to get both validity and diversity.

**Main result.** On all five tasks GRAM gives the most distinct valid answers, a baseline without a posterior is the
most valid, and no model is both. On the test split, GRAM's diversity costs 0.9 (graph colouring) to 22 (mazes)
points of validity.

- Report: [`writeup/icml2027/paper.pdf`](writeup/icml2027/paper.pdf); further runs:
  [`writeup/icml2027/extra_results.pdf`](writeup/icml2027/extra_results.pdf).
- Every term is defined in [`writeup/glossary.md`](writeup/glossary.md). Earlier drafts are in `writeup/archive/`.
- All numbers below are one training seed. Unless marked *test*, they are on the 1000-input validation split
  (500 inputs per scored checkpoint, 100 samples each), given as **validity (%) / unique-solution validity (%) /
  distinct@100** (distinct valid answers in 100 samples, on inputs with several solutions).

**Contents:** [1. Setup](#1-setup) · [2. Datasets](#2-datasets) · [3. Models and code map](#3-models-and-code-map) ·
[4. Main study](#4-main-study) · [5. Ideas we tried](#5-ideas-we-tried) · [6. Reproducing each table](#6-reproducing-each-table)

## 1. Setup

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Python 3.12, CUDA 13.0. Every run used one NVIDIA A100 80GB, except best-of-K with gradient through all transitions on
mazes, which needs two (the 768-sequence batch is split across them; `CUDA_VISIBLE_DEVICES=0,1`). A 150k-step run takes
9–17 hours. Scripts are run from the repository root. Add `EXTRA_ARGS="wandb_offline=True"` to training commands to
log to Weights & Biases offline.

## 2. Datasets

Every task has 100k training, 1000 validation and 2000 test inputs, with no input shared between splits. Each input
stores at most 16 of its solutions (its *stored solutions*) and the exact number of solutions, so inputs with more than
16 have *unseen solutions* that were never in training. Inputs are spread over groups of solution counts: unique, 2–3,
4–15, 16–63, 64–255 and 256 or more.

| Task | `<task>` | Input | Solutions per input | Source | Builder |
|---|---|---|---|---|---|
| Sudoku | `sudoku` | 9×9 grid, 26–50 clues | 1–1024 | generated | `dataset/build_sudoku_multi_dataset.py` |
| Maze | `maze` | 15×15 grid | 1–924 shortest paths | generated | `dataset/build_maze_dataset.py` |
| Graph colouring | `graph` | 10 nodes, 3 colours | 1–9842 | generated | `dataset/build_graph_coloring_multi_dataset.py` |
| Heyawake | `heyawake` | 10×10 grid | 1–1023 | human-made (PPBench) | `dataset/build_heyawake_multi_dataset.py` |
| Nurikabe | `nurikabe` | 10×10 grid | 1–1023 | human-made (PPBench) | `dataset/build_nurikabe_multi_dataset.py` |
| N-Queens | `nqueens` | 10×10, 1–3 queens given | 1–16 | generated | `dataset/build_nqueens_dataset.py` |

N-Queens appears only in the extra results. Heyawake and nurikabe start from the human-made *base puzzles* of Pencil
Puzzle Bench (PPBench, `full_dataset.jsonl` at
[huggingface.co/datasets/bluecoconut/pencil-puzzle-bench](https://huggingface.co/datasets/bluecoconut/pencil-puzzle-bench)),
each with a unique solution. A builder *loosens* some of a base puzzle's numbers, so every original solution stays
valid and the count can only grow, then counts the solutions exactly with the answer set solver clingo. The base
puzzles are split between training, validation and test first, so no base puzzle is shared.

```bash
bash scripts/build_multi_datasets.sh sudoku maze graph nqueens          # CPU only, uses every core
hf download bluecoconut/pencil-puzzle-bench full_dataset.jsonl --repo-type dataset --local-dir ppbench
PPBENCH_SOURCE=ppbench/full_dataset.jsonl bash scripts/build_multi_datasets.sh heyawake nurikabe   # needs clingo
```

Each task is written in three layouts under `data/`; `dataset/multi_solution.py` holds the shared builder code.

| Directory | Rows | Used by |
|---|---|---|
| `data/<task>-multi` | one row per stored solution | GRAM (its target is one random stored solution per epoch) |
| `data/<task>-multi-1oml` | one row per input, with all stored solutions (`label_set`) | best-of-K, MinLoss, the hybrid and every evaluation |
| `-single`, `-unique` variants | one stored solution / unique-solution inputs only | extra results only |

## 3. Models and code map

All models share GRAM's network (2 layers, 3 high-level × 4 low-level cycles, up to 16 supervision steps, 150k
optimiser steps of 768 sequences) and differ only in where the stochastic guidance comes from and which target the loss
uses. `scripts/multi_compare.sh <task> <method>` trains any of them.

| Model | `<method>` | Stochastic guidance | Loss |
|---|---|---|---|
| GRAM | `gram` (β with `KL_COEF`, default the reference β) | posterior in training, learned prior at test | cross-entropy to a random stored solution + β·KL |
| Best-of-K | `gram_std_bestofk` | N(0, I) in training and test | K = 8 rollouts per input; only the one closest to a stored solution is trained |
| MinLoss | `gram_std_bestofk` with `ROLLOUTS=1` | N(0, I) | cross-entropy to the stored solution closest to the answer |

The reference β is 4.5 divided by the number of answer cells (Sudoku 0.05, graph colouring 0.45, maze 0.02, heyawake and
nurikabe 0.045), the rule GRAM's own settings follow.

| Where | What |
|---|---|
| `models/recursive_reasoning/gram.py` | the network and recursion (`GenerativeRecursiveReasoningModelInner.forward`), prior and posterior (`VariationalLayer`), every option (`GenerativeRecursiveReasoningModelConfig`) |
| `models/losses.py` | `ACTLossHead`: cross-entropy, KL, halting loss; best-of-K's closest-solution choice (`target_set_losses`) and the other target assignments |
| `config/arch/gram.yaml`, `config/cfg_pretrain.yaml` | default settings (constant learning rate 1e-4 after 2000 warmup steps, EMA weights) |
| `pretrain.py` | training loop |
| `dataset/puzzle_dataset.py` | data loading; repeats each input K times for best-of-K, before the batch is split across GPUs |
| `sample_predictions.py` | draws 100 samples per input from a checkpoint, in resumable shards |
| `scripts/analyze_multi_samples.py`, `evaluators/*_utils.py` | each task's checker; validity, distinct@100, unseen solutions |
| `scripts/multi_compare.sh` | trains a model: `<task> <method>` |
| `scripts/score_checkpoints.sh` | scores every second saved checkpoint on 500 validation inputs |
| `scripts/checkpoint_trends.py` | prints those scores and the *run summary* (mean of the last 4 scored checkpoints) |
| `scripts/multi_evaluate.sh` | one checkpoint on any split; `NOISE=<σ>` adds test-time state noise, `LATENT=posterior` samples GRAM's posterior |
| `scripts/report_numbers.py` | prints every single-checkpoint result (validation, test, train with unseen solutions) |

A full experiment, for example best-of-K on Sudoku:

```bash
bash scripts/multi_compare.sh sudoku gram_std_bestofk                                  # train
R=gram_std_bestofk_steps150000_seed42                                                  # run name
bash scripts/score_checkpoints.sh sudoku $R && python scripts/checkpoint_trends.py $R sudoku   # run summary
NOISE=0.5 QUERIES=500 bash scripts/multi_evaluate.sh sudoku $R val                     # state noise
bash scripts/multi_evaluate.sh sudoku $R test,train                                    # test split, unseen solutions
python scripts/report_numbers.py sudoku
```

## 4. Main study

### Validity against diversity

Run summaries on validation; *test* is the final checkpoint on the 2000 test inputs. The highest validity and
distinct@100 of each task are in bold.

| Task | GRAM (reference β) | Best-of-K | MinLoss | Test: GRAM / best-of-K / MinLoss validity |
|---|---|---|---|---|
| Sudoku | 92.5 / 97.6 / **34.0** | 99.2 / 99.3 / 10.4 | **99.5** / 99.8 / 2.5 | 89.9 / 99.5 / 99.5 |
| Graph colouring | 97.0 / 97.4 / **44.9** | **99.8** / 100.0 / 38.4 | 99.7 / 99.5 / 1.8 | 99.0 / 99.6 / 99.9 |
| Maze (GRAM at ½β, 0.01) | 67.3 / 48.2 / **35.3** | 83.3 / 75.4 / 2.2 | **88.4** / 94.1 / 1.0 | 66.7 / 88.8 / 84.1 |
| Heyawake | 78.7 / 72.0 / **33.8** | 85.3 / 81.9 / 9.7 | **96.2** / 97.6 / 1.3 | 77.0 / 81.7 / 94.4 |
| Nurikabe | 82.0 / 83.9 / **29.8** | 78.7 / 78.4 / 7.2 | **89.1** / 94.4 / 1.2 | 81.1 / 77.9 / 86.1 |

On the test split every value is within 6 points of validation and the ranking is the same on every task: GRAM is the
most diverse, MinLoss or best-of-K the most valid. On heyawake GRAM fails single-solution inputs (72.0% against
MinLoss's 97.6%) because the prior it samples at test time does not match the posterior it was trained with.

### Choosing β

| Task | ½ reference β | reference β | 2× reference β |
|---|---|---|---|
| Sudoku | 53.6 / 79.8 / 14.4 | 92.5 / 97.6 / 34.0 | 35.4 / 98.3 / 0.3 |
| Graph colouring | 71.5 / 93.4 / 31.2 | 97.0 / 97.4 / 44.9 | 86.5 / 100.0 / 2.2 |
| Maze | 67.3 / 48.2 / 35.3 | 35.9 / 97.8 / 0.8 | 33.8 / 98.6 / 0.0 |
| Heyawake | 2.2 / 1.6 / 1.8 | 78.7 / 72.0 / 33.8 | 61.0 / 95.2 / 0.5 |
| Nurikabe | 0.1 / 0.0 / 0.1 | 82.0 / 83.9 / 29.8 | 54.3 / 88.2 / 0.5 |

β works only in a narrow band: heyawake and nurikabe already collapse at 1.27× the reference β (distinct@100 0.6).
**Posterior sampling** (`LATENT=posterior`) shows which side of the band a run is on. Below it the posterior still gives
98–100% valid answers, so the prior is what fails. Above it, prior and posterior agree, because the stochastic guidance
carries nothing.

### Unseen solutions

On training inputs with more than 16 solutions (801–881 per task), the share of valid samples that are unseen
solutions, with a uniform sampler's share in brackets:

| Task | GRAM | Best-of-K | MinLoss |
|---|---|---|---|
| Sudoku | 76.5 (76.6) | 77.0 (76.8) | 77.6 (76.6) |
| Graph colouring | 84.3 (84.5) | 84.7 (84.6) | 82.6 (84.6) |
| Maze (GRAM at 0.01) | 79.2 (78.9) | 75.7 (76.5) | 68.2 (68.3) |
| Heyawake | 76.9 (77.0) | 76.0 (76.2) | 71.8 (76.1) |
| Nurikabe | 75.6 (75.3) | 74.4 (74.9) | 72.7 (73.3) |

Every model finds unseen solutions as often as a uniform sampler would; none prefers the solutions it was trained on.

## 5. Ideas we tried

Each tried to give a model without a posterior GRAM's diversity, or GRAM the baseline's validity. They are listed in
the order we tried them.

### Idea 1: test-time state noise for the baseline. Partial success

**Hypothesis:** best-of-K's answers vary little because its high-level state is nearly deterministic; noise in it
should spread the answers. **Change:** `NOISE=<σ>` in `scripts/multi_evaluate.sh` adds N(0, σ²) to the high-level state
in every transition without gradient (`--state_noise` in `sample_predictions.py`).

| Task | Best-of-K | σ = 0.5 | σ = 1 |
|---|---|---|---|
| Sudoku | 99.2 / 10.4 | 99.5 / 27.7 | 97.5 / 29.1 |
| Graph colouring | 99.8 / 38.4 | 99.6 / 39.5 | 99.4 / 40.4 |
| Maze | 83.3 / 2.2 | 88.5 / 3.2 | 89.9 / 4.4 |
| Heyawake | 85.3 / 9.7 | 85.0 / 10.7 | 83.6 / 11.7 |
| Nurikabe | 78.7 / 7.2 | 78.5 / 7.7 | 76.9 / 8.5 |

(validity / distinct@100.) Triples distinct@100 on Sudoku at no cost in validity, but adds at most 2 elsewhere; at
σ = 1.5 Sudoku validity falls to 23%.

### Idea 2: other sources of randomness. Failure

**Hypothesis:** the baseline needs randomness that persists or that it is trained with. **Changes:**
- one N(0, I) vector per sample instead of per cell (`LATENT_SCOPE=sample`);
- a random initial high-level state (STRM, method `strm_bestofk`);
- state noise during training as well (`TRAIN_NOISE=1`).

**Result:** on Sudoku every variant stays at or below about 28 distinct@100 at 99% validity (one vector per sample:
8.9; STRM: 16.8, 26.1 with state noise). Training noise moves where validity breaks but not the best point. One vector
per sample lowers validity on graph colouring (47.7%).

### Idea 3: train every rollout, not only the closest. Failure

**Hypothesis:** best-of-K trains one rollout in eight. Training all of them, each towards its own target, should
spread the rollouts over the stored solutions. **Changes:** method `gram_std_nearest` (each rollout towards its own
closest stored solution) and `gram_std_balanced` (each stored solution gets at most ⌈8 / stored⌉ rollouts, closest
pairs first; a rollout sent elsewhere has weight `FORCED_WEIGHT=0.3`), in `ACTLossHead` and `balanced_assignment`.

**Result (Sudoku):** nearest 99.3 / 100.0 / 4.6 and balanced 99.2 / 99.8 / 3.3, against best-of-K's 10.4. Balanced
kept half its rollouts on another solution to the end, yet its answers did not spread. The network cannot tell from
the stochastic guidance which solution a rollout is meant for, so it learns to ignore the guidance. Best-of-K's
diversity comes from the rollouts it leaves untrained. With one vector per sample (idea 2) as well: 4.1 on Sudoku,
and lower validity on heyawake (80.0) and nurikabe (56.8).

### Idea 4: hybrid GRAM. Failure

**Hypothesis** (after the conditional VAE of Sohn et al., 2015): GRAM fails at test time because its prior is never
used in training. Training a share of rows with the prior's stochastic guidance and MinLoss should close that gap.
**Change:** method `gram_hybrid` with `PRIOR_ROWS` (default 0.25); the KL term covers only the posterior rows
(`hybrid` in `gram.py` and `losses.py`).

| Task | GRAM | Hybrid, 25% prior rows | 50% | 25%, ½β |
|---|---|---|---|---|
| Heyawake | 78.7 / 72.0 / 33.8 | 79.9 / 92.3 / 0.9 | 92.3 / 95.3 / 1.2 | 91.3 / 88.7 / 6.2 |
| Nurikabe | 82.0 / 83.9 / 29.8 | 71.5 / 90.4 / 0.9 | 85.0 / 94.2 / 1.2 | 0.0 / 0.0 / 0.0 |
| Sudoku | 92.5 / 97.6 / 34.0 | 62.7 / 99.6 / 1.8 | | |
| Graph colouring | 97.0 / 97.4 / 44.9 | 99.3 / 99.5 / 2.1 | | |

With 0% prior rows the hybrid reproduces GRAM (heyawake 79.0 / 71.9 / 33.5). With any prior rows, diversity collapses.
The rows trained with MinLoss teach the shared decoder to ignore the stochastic guidance, and even posterior sampling
then gives 0.9 distinct@100 on heyawake. The posterior rows are then trained towards a random stored solution with
nothing to say which, so on Sudoku they produce mixtures of solutions that are invalid.

### Idea 5: gradient through all transitions. Partial success

**Hypothesis:** in training, only the last transition of a supervision step has gradient, and the only stochastic
guidance on that path is drawn per cell just before the output, where it cannot choose a whole solution. Keeping the
gradient through every transition lets earlier guidance shape the whole answer. **Change:** `FULL_GRAD=1` in
`scripts/multi_compare.sh` (`grad_all_transitions` in `gram.py`), with best-of-K.

| Task | Best-of-K | + gradient through all transitions |
|---|---|---|
| Sudoku | 99.2 / 99.3 / 10.4 | 99.5 / 99.7 / **17.0** |
| Maze (40k steps) | 83.3 / 75.4 / 2.2 | 81.2 / 68.8 / **3.6** |
| Heyawake (to step 95k) | 85.3 / 81.9 / 9.7 | **88.6 / 90.4** / 5.1 |
| Nurikabe (σ = 0.5, step 87k) | 78.5 / 79.3 / 7.7 | **85.0 / 86.9** / 5.4 |
| Graph colouring | 99.8 / 100.0 / 38.4 | running |

It is the first change to training that raised best-of-K's diversity at no cost in validity (Sudoku and mazes). On the
human-made puzzles it makes the baseline a better solver instead (+3 to +7 validity, +8 unique-solution validity) with
fewer distinct answers. With state noise σ = 1 on Sudoku it reaches 28.1, the same limit as idea 1. Rows marked with a
step are from runs still in progress on 27 November 2026; this is the starting point for further work.

## 6. Reproducing each table

| Report table | README section | Runs |
|---|---|---|
| Main results, frontier figure | 4, validity against diversity | `gram`, `gram_std_bestofk`, `ROLLOUTS=1`, each scored with `score_checkpoints.sh`; noise rows with `NOISE=0.5` / `1`; test with `multi_evaluate.sh ... test` |
| β table | 4, choosing β | `KL_COEF=<½, 1, 2 × reference> bash scripts/multi_compare.sh <task> gram` |
| Posterior sampling | 4, choosing β | `LATENT=posterior QUERIES=500 bash scripts/multi_evaluate.sh <task> gram_kl<β>_steps150000_seed42 val` |
| Unseen solutions | 4, unseen solutions | `bash scripts/multi_evaluate.sh <task> <run> train`, read with `report_numbers.py` |
| Extra results | 5 | the method and option named in each idea |

Run names follow the method and options, for example `gram_kl0.045_steps150000_seed42`,
`gram_std_bestofk_k1_steps150000_seed42` (MinLoss), `gram_std_bestofk_fullgrad_steps150000_seed42` and
`gram_hybrid_pr0.25_kl0.045_steps150000_seed42`; each script's header lists its options. The code from before the
project was narrowed to these models is at the git tag `pre-prune`.
