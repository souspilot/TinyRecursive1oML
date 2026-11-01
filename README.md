# Generative Recursive Reasoning (GRAM)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-blue)](https://neurips.cc/) [![arXiv](https://img.shields.io/badge/arXiv-2605.19376-b31b1b)](https://arxiv.org/abs/2605.19376) [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://ahn-ml.github.io/gram-website/)

Author: Junyeob Baek*, Mingyu Jo*, Minsu Kim, Mengye Ren, Yoshua Bengio, Sungjin Ahn

This is official codebase for the paper: "Generative Recursive Reasoning".

### Abstract
How should future neural reasoning systems implement extended computation? Recursive Reasoning Models (RRMs) offer a promising alternative to autoregressive sequence extension by performing iterative latent-state refinement with shared transition functions. Yet existing RRMs are largely deterministic, following a single latent trajectory and converging to a single prediction. We introduce Generative Recursive reAsoning Models (GRAM), a framework that turns recursive latent reasoning into probabilistic multi-trajectory computation. GRAM models reasoning as a stochastic latent trajectory, enabling multiple hypotheses, alternative solution strategies, and inference-time scaling through both recursive depth and parallel trajectory sampling. This yields a latent-variable generative model supporting conditional reasoning via p(y|x) and, with fixed or absent inputs, unconditional generation via p(x). Trained with amortized variational inference, GRAM improves over deterministic recurrent and recursive baselines on structured reasoning and multi-solution constraint satisfaction tasks, while demonstrating an unconditional generation capability.

## Overview
<div align="center">
<img src="misc/fig1.png" width="100%" alt="GRAM Overview">
</div>


### Requirements

- Python 3.10 (or similar)
- Cuda 12.6.0 (or similar)

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 # install torch based on your cuda version
pip install -r requirements.txt # install requirements
pip install adam-atan2-pytorch # provides the adam_atan2_pytorch module imported by pretrain.py
wandb login YOUR-LOGIN # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
```

### Dataset Preparation

```bash
# ARC-AGI-1
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

## Note: You cannot train on both ARC-AGI-1 and ARC-AGI-2 and evaluate them both because ARC-AGI-2 training data contains some ARC-AGI-1 eval data

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples, 1000 augments

# Sudoku Generation (blank board -> valid solution)
python dataset/build_sudoku_generation_dataset.py --output-dir data/sudoku-generation-50k --train-size 50000 --test-size 10000

# N-Queens
python dataset/build_nqueens_dataset.py --n 8 --remove-counts 5 6 7 --output-dir data/nqueens-8
python dataset/build_nqueens_dataset.py --n 10 --remove-counts 7 8 9 --output-dir data/nqueens-10

# Graph Coloring
python dataset/build_graph_coloring_dataset.py --config-name graph-coloring-8 --output-dir data/graph-coloring-8
python dataset/build_graph_coloring_dataset.py --config-name graph-coloring-10 --output-dir data/graph-coloring-10

# MNIST
python dataset/build_mnist_dataset.py --output-dir data/mnist_2 --vocab-levels 2 --test-vocab-levels 256 --threshold 0.5
```

## Experiments

```bash
# ARC-AGI-1 (4 H-100 GPUs, ~3 days)
bash scripts/arc1.sh

# ARC-AGI-2 (4 H-100 GPUs, ~3 days)
bash scripts/arc2.sh

# Sudoku-Extreme (8 RTX 4090 GPUs, < 2 hours)
bash scripts/sudoku.sh

# Sudoku Generation (8 RTX 4090 GPUs)
bash scripts/sudoku_generation.sh

# N-Queens (8 RTX 4090 GPUs, ~1 hour, ~3 hours)
bash scripts/nqueens_8.sh
bash scripts/nqueens_10.sh

# Graph Coloring (8 RTX 4090 GPUs, ~2 hours, ~6 hours)
bash scripts/graph_coloring_8.sh
bash scripts/graph_coloring_10.sh

# MNIST (8 RTX 4090 GPUs, ~16 hours)
bash scripts/mnist.sh
```

## One-of-Many Learning (1oML) comparison on N-Queens

`scripts/nqueens_compare.sh` compares GRAM and vanilla TRM against TRM trained with the solution-multiplicity-aware objectives of [Nandwani et al., ICLR 2021](https://arxiv.org/abs/2008.11990). All methods are evaluated on the same test split: one row per unique input, checked by the N-Queens verifier. Metrics include `exact_accuracy_os` and `exact_accuracy_ms` for one-solution and multi-solution queries.

```bash
# Datasets: --one-of-many stores every solution of an input in label_set
python dataset/build_nqueens_dataset.py --n 8 --remove-counts 5 6 7 --output-dir data/nqueens-8-1oml --one-of-many
python dataset/build_nqueens_dataset.py --n 8 --remove-counts 5 6 7 --output-dir data/nqueens-8-unique --one-of-many --train-unique-only

bash scripts/nqueens_compare.sh 8 gram         # GRAM on every (input, solution) pair
bash scripts/nqueens_compare.sh 8 trm          # 1oML "Naive": TRM on every (input, solution) pair
bash scripts/nqueens_compare.sh 8 trm_random   # one fixed random solution per input
bash scripts/nqueens_compare.sh 8 trm_unique   # single-solution inputs only
bash scripts/nqueens_compare.sh 8 trm_minloss  # train towards the closest solution
bash scripts/nqueens_compare.sh 8 trm_iexplr   # sample the target by model likelihood
bash scripts/nqueens_compare.sh 8 trm_selectr  # learned target selector, started from trm_minloss's final checkpoint
```

To spread the runs over several GPU jobs, start the same worker in each job. Workers share one queue, so each run is trained once, and SelectR starts as soon as MinLoss has finished:

```bash
bash scripts/nqueens_worker.sh 10   # in every job; claims live in checkpoints/Nqueens-10-compare/.claims/
```

If a job is killed (e.g. at walltime), continue its run from the latest checkpoint, with optimizer state, step and data order restored:

```bash
RESUME=1 bash scripts/nqueens_compare.sh 10 gram
```

Once the runs finish, evaluate them all and print a comparison table (also saved to `results/nqueens_8/summary.md`):

```bash
bash scripts/nqueens_eval_all.sh 8   # GRAM: 100 samples; TRM is deterministic, so 1 sample
```

Runs that are still training are skipped, and runs already evaluated are not redone, so rerun it as more runs finish.

The objective is set with `arch.loss.one_of_many` (`naive | minloss | iexplr | selectr`) and works with any dataset built with `--one-of-many`. It requires a model that does not condition on the labels, so it is not available for GRAM. SelectR differs from the paper in two ways: the selector G_φ is a small transformer rather than an NLM, and it sees the current soft prediction instead of a lagged copy of the model (the paper's `copyitr = 1`).

Every training run also saves its end-of-training weights to `checkpoints/<project>/<run>/final/checkpoint.pt`.

### Hidden-constraint experiment

Tests when a model's set of answers matters more than one committed answer. Each finished comparison run is sampled
on the test split (GRAM: 100 samples per query; TRM: 1, plus a temperature-sampled MinLoss baseline), then scored on
an unseen requirement that only some of each query's solutions meet, on accuracy vs number of samples (majority
vote, verifier, coverage), and on accuracy by number of solutions and given queens:

```bash
bash scripts/hidden_constraints.sh 10   # results/hidden_constraints_10/summary.md and figures
```

Sampling is saved in shards (`samples/`), so rerunning after an interruption continues where it stopped, and it uses
every visible GPU. The analysis (`scripts/analyze_nqueens_samples.py`) runs on CPU from the saved samples.

## Mazes with several shortest paths

A second domain for the same comparison, chosen so that test answers can't be memorized and solution sets can be
large without being listed. 15x15 mazes have between 1 and 924 shortest paths; their number, and the share
through each cell, are counted exactly (`evaluators/maze_utils.py`), and any shortest path counts as correct.

```bash
# Datasets (a few minutes on a multi-core node): data/maze-15, -1oml, -single, -unique share one test split
python dataset/build_maze_dataset.py --size 15 --num-train 100000 --num-test 2000 --output-dir data/maze-15

# Training (~50k steps per run; GRAM, gram_single, trm, trm_minloss, trm_iexplr, trm_random, trm_unique)
bash scripts/maze_worker.sh                    # in each GPU job; or: bash scripts/maze_compare.sh <method>
RESUME=1 bash scripts/maze_compare.sh gram     # continue a run killed at walltime

# Sampling and analysis (hidden constraints, accuracy vs samples, even sampling): results/maze_15/summary.md
bash scripts/maze_hidden_constraints.sh
```

Noisy TRM with set matching (`trm_setmatch`): each maze gets several rollouts (`ROLLOUTS`, default 4), each with its
own random vector fed to every recursion step (`arch.noise_dim`), and the rollouts are matched one to one to the
maze's stored paths, so the model learns to use the noise to choose a solution. `trm_noise_minloss` is the ablation
without matching. Both keep the ~50k-step budget (the batch still holds 768 sequences) and are sampled 100 times.

```bash
bash scripts/maze_compare.sh trm_setmatch        # watch train/rollout_distinct: 1/ROLLOUTS means the noise is ignored
bash scripts/maze_compare.sh trm_noise_minloss
```

GRAM with a per-maze KL budget (`gram_budget`): one KL weight can't suit both mazes with one shortest path (the
latent should carry nothing) and mazes with hundreds (it must pick one). With `arch.kl_budget=b`, a maze's posterior
KL is free up to b·ln(number of shortest paths) nats, the information needed to pick one path, and weighted by
`kl_coef` above it; the prior always learns to follow the posterior. It needs `num_solutions` at training time only.

```bash
bash scripts/maze_compare.sh gram_budget                # KL_COEF=0.05, KL_BUDGET=1.0 by default
KL_BUDGET=3 bash scripts/maze_compare.sh gram_budget    # run name gram_budget_b3_seed42
```

### Deciding one step at a time (walk decoding)

A model trained on one random path per maze (`trm`) learns the share of shortest paths through each cell, but
marking every cell at once blends paths. `walk_decode.py` instead walks from the start, one step at a time,
choosing each step by the model's probabilities and feeding the steps taken back as input. With exact shares
this samples shortest paths uniformly; the model is re-run only at uncertain steps (about one pass per fork).
`--mode start` moves the start token to the walker (works with any maze model); `--mode hint` marks the walked cells
as PATH, for models trained with path hints (`trm_hints`). `--choice greedy` gives one deterministic answer.

```bash
bash scripts/maze_compare.sh trm_hints                  # TRM trained with partial paths revealed in its inputs
torchrun --nproc-per-node 1 walk_decode.py --checkpoint checkpoints/Maze-15-compare/trm_seed42/final/checkpoint.pt \
  --data data/maze-15-1oml --out samples/maze_15/trm_seed42_walk --num_samples 100 --mode start --choice sample
python scripts/maze_marginals.py --data data/maze-15-1oml --out results/maze_15/marginals \
  --runs trm_seed42=checkpoints/Maze-15-compare/trm_seed42/final/checkpoint.pt   # probabilities vs exact shares
```

`sample_predictions.py` also saves each sample's halting logit (the model's own estimate that it is right; the
analysis reports best-of-K picked by it), and takes `--latent_noise` (Gaussian noise on the latent state at each
recursion step, as in the Probabilistic Tiny Recursive Model) and `--halt_steps` (more recursion at test time).

All of these experiments run from one queue, one worker per GPU, in any number of jobs:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/maze_h2_smoke.sh     # a few minutes: checks the pipeline end to end
CUDA_VISIBLE_DEVICES=0 bash scripts/maze_h2_worker.sh 2>&1 | tee -a logs/maze_h2_worker_$(hostname)_0.log
```

## Evaluation

```bash
# bash scripts/eval_pass_k.sh <checkpoint_dir> <num_samples>
bash scripts/eval_pass_k.sh checkpoints/<project_name>/<run_name>/step_<N> 100
```

Set `--data_paths` in `scripts/eval_pass_k.sh` to the target dataset (default: `data/nqueens-8`). `<num_samples>` must be one of 1, 2, 5, 10, 20, 100, 250, 500, 1000. Results are printed and saved to `eval_results_*.txt` in the checkpoint directory.

## Reference

```bibtex
@article{baek2026generative,
  title={Generative recursive reasoning},
  author={Baek, Junyeob and Jo, Mingyu and Kim, Minsu and Ren, Mengye and Bengio, Yoshua and Ahn, Sungjin},
  journal={arXiv preprint arXiv:2605.19376},
  year={2026}
}
```


This code is based on the Hierarchical Reasoning Model [code](https://github.com/sapientinc/HRM) and the Tiny Reasoning Model [code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).
