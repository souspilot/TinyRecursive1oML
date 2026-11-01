#!/bin/bash
# Quick end-to-end check of the H2 code on one GPU (a few minutes), before starting scripts/maze_h2_worker.sh.
# Writes only to samples/maze_15_smoke and results/maze_15_smoke.
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/maze_h2_smoke.sh
set -eo pipefail
runs=checkpoints/Maze-15-compare
out=samples/maze_15_smoke
rm -rf ${out} results/maze_15_smoke
run() { torchrun --nproc-per-node 1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 "$@"; }

echo "== 1. Training loader with path hints"
python - <<'PY'
import numpy as np
from dataset.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from evaluators.maze_utils import PATH
ds = PuzzleDataset(PuzzleDatasetConfig(seed=0, dataset_paths=["data/maze-15"], global_batch_size=768, test_set_mode=False,
                                       epochs_per_iter=1, rank=0, num_replicas=1, path_hints=0.75), split="train")
for i, (_, batch, _) in zip(range(3), ds):
    x, y = batch["inputs"].numpy(), batch["labels"].numpy()
    hinted = (x == PATH)
    assert (y[hinted] == PATH).all(), "hints must be target path cells"
    print(f"batch {i}: {hinted.any(1).mean():.2f} of inputs hinted, {hinted.sum(1).mean():.1f} hint cells on average")
PY

echo "== 2. Walk decoding (start and hint conditioning) on naive TRM, 32 mazes"
run walk_decode.py --checkpoint ${runs}/trm_seed42/final/checkpoint.pt --data data/maze-15-1oml --out ${out}/walk_start \
  --num_samples 8 --mode start --choice sample --max_queries 32 --shard_size 16
run walk_decode.py --checkpoint ${runs}/trm_seed42/final/checkpoint.pt --data data/maze-15-1oml --out ${out}/walk_hint \
  --num_samples 2 --mode hint --choice greedy --max_queries 32 --shard_size 16

echo "== 3. Sampling with latent noise, more recursion steps and halting logits (MinLoss), 32 mazes"
run sample_predictions.py --checkpoint ${runs}/trm_minloss_seed42/final/checkpoint.pt --data data/maze-15-1oml \
  --out ${out}/noise --num_samples 8 --latent_noise 1.0 --halt_steps 20 --max_queries 32 --shard_size 16

echo "== 4. Per-cell probabilities vs exact shares (naive TRM, full test split)"
python scripts/maze_marginals.py --data data/maze-15-1oml --out results/maze_15_smoke/marginals \
  --runs trm_seed42=${runs}/trm_seed42/final/checkpoint.pt

echo "== 5. Analysis of the smoke samples"
python scripts/analyze_maze_samples.py --data data/maze-15-1oml --samples ${out} --out results/maze_15_smoke \
  | grep -A6 "Accuracy vs number\|Compute and"
echo "== Smoke test passed"
