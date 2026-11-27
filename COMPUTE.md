# Compute

Approximate cost of every kind of job in the [README](README.md), measured from our training and evaluation logs.

**Hardware:** NVIDIA A100 80GB GPUs on two clusters, in a conda environment with Python 3.12 and CUDA 13.0.

**How to read the times:**
- Times are wall-clock for one run.
- They vary by about 20% with how busy the node was.
- "150k steps" runs are 151,041 to 153,125 optimiser steps, depending on the task. Each step is 768 sequences.

## Training (`scripts/multi_compare.sh`)

| Job | GPUs | Speed (steps/s) | 150k steps |
|---|---|---|---|
| GRAM, best-of-K, MinLoss, hybrid: graph colouring | 1 | 4.3 | ~10 h |
| the same: Sudoku | 1 | 3.5 | ~12 h |
| the same: heyawake, nurikabe | 1 | 2.7–2.9 | ~15 h |
| the same: maze | 1 | 1.3 | ~31 h |
| Gradient through all transitions (`FULL_GRAD=1`): graph colouring | 1 | 3.8 | ~11 h |
| the same: Sudoku | 1 | 2.9 | ~15 h |
| the same: heyawake, nurikabe | 1 | 2.5 | ~17 h |
| the same: heyawake, nurikabe | 2 | ~4.7 | ~9 h |
| the same: maze | **2** (out of memory on one) | 2.3 | ~18 h (our run: 40k steps, ~5 h) |

- **More GPUs:** with `CUDA_VISIBLE_DEVICES=0,1`, the 768-sequence batch is split across the two GPUs. Training is otherwise unchanged; for best-of-K, all rollouts of an input stay on the same GPU.
- **Memory:** one 80 GB GPU is enough for every other job.
- **Untimed runs:** the extra-results runs (STRM, nearest, balanced, one vector per sample, training noise) cost about the same as best-of-K on the same task. We didn't time them separately.

## Evaluation

| Job | Inputs × samples | GPUs | Time |
|---|---|---|---|
| One checkpoint on validation (`multi_evaluate.sh ... val`, with or without `NOISE` / `LATENT=posterior`) | 500 × 100 | 1 | 10–20 min; maze ~30 min |
| Checkpoint sweep (`score_checkpoints.sh`, every second checkpoint) | 10 checkpoints | 1 | ~2–3 h; maze ~5 h |
| Test split (`multi_evaluate.sh ... test`) | 2000 × 100 | 1 | ~1 h; maze ~2 h |
| Training split, unseen solutions (`multi_evaluate.sh ... train`) | 2000 × 100 | 1 | ~1 h; maze ~2 h |
| Reading results (`checkpoint_trends.py`, `report_numbers.py`) | | none | seconds |

## Datasets

`scripts/build_multi_datasets.sh` runs on CPU only and uses every core. It needs no GPU.

| Builder | Extra requirement |
|---|---|
| Heyawake, nurikabe | `clingo` and the Pencil Puzzle Bench file |
| Sudoku, maze, graph colouring, N-Queens | none |

## The whole project

- **About 70 training runs** of 150k steps, all with one seed:
  - GRAM at 3–6 values of β per task;
  - best-of-K and MinLoss on every task;
  - the ideas in the README's section 5;
  - N-Queens.
- **Each run** was followed by a checkpoint sweep and a few single-checkpoint evaluations.
- **In all:** on the order of 1,000 A100-hours.
- **To reproduce one row** of the main table: about 15 GPU-hours of training plus 3 of evaluation (maze: 31 + 7).
