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

The objective is set with `arch.loss.one_of_many` (`naive | minloss | iexplr | selectr`) and works with any dataset built with `--one-of-many`. It requires a model that does not condition on the labels, so it is not available for GRAM. SelectR differs from the paper in two ways: the selector G_φ is a small transformer rather than an NLM, and it sees the current soft prediction instead of a lagged copy of the model (the paper's `copyitr = 1`).

Every training run also saves its end-of-training weights to `checkpoints/<project>/<run>/final/checkpoint.pt`.

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
