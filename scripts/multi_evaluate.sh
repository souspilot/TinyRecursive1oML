#!/bin/bash
# Evaluate a run trained with scripts/multi_compare.sh: draw SAMPLES answers per input from the model, each with a
# fresh random latent (sample_predictions.py), then score them all with scripts/analyze_multi_samples.py (validity,
# diversity against a uniform sampler, error anatomy and, on the train split, held-out solutions).
#
# Usage: bash scripts/multi_evaluate.sh <maze|nqueens|sudoku|graph> <run> [splits]
#   e.g. bash scripts/multi_evaluate.sh sudoku gram_std_bestofk_seed42            (test and train, one model load)
#        bash scripts/multi_evaluate.sh graph gram_std_bestofk_seed42 test        (test only)
#
# splits: test, train or test,train (default). SAMPLES per input (default 100). QUERIES limits the training inputs
# (default 2000, the first ones, which are in random order; test always uses all). STEP picks a checkpoint other than
# the final one (e.g. STEP=41660 for checkpoints/<project>/<run>/step_41660). GPUS for torchrun (default 1). BATCH
# sets queries x samples per forward pass (default: scaled to the sequence length to fit an 80GB GPU; lower it to
# share a GPU with a training run).
# Samples go to samples/<dataset>-<split>/<run>[_step<STEP>], results to results/<dataset>-<split>. Finished shards
# and up-to-date analyses are reused, so rerunning after a failure only redoes what is missing.
set -e
dataset=$1
run=$2
splits=${3:-test,train}
case $dataset in
  maze)    data=data/maze-15-1oml project=Maze-15-compare ;;
  nqueens) data=data/nqueens-10-s16-1oml project=Nqueens-multi ;;
  sudoku)  data=data/sudoku-multi-1oml project=Sudoku-multi ;;
  graph)   data=data/graph-coloring-multi-1oml project=Graph-coloring-multi ;;
  *) echo "Unknown dataset: ${dataset} (maze, nqueens, sudoku, graph)"; exit 1 ;;
esac
step_dir=final
name=${run}
if [ -n "${STEP}" ]; then step_dir=step_${STEP}; name=${run}_step${STEP}; fi
checkpoint=checkpoints/${project}/${run}/${step_dir}/checkpoint.pt
[ -f "${checkpoint}" ] || { echo "No checkpoint at ${checkpoint}"; ls "checkpoints/${project}/${run}" 2>/dev/null; exit 1; }

torchrun --nproc-per-node ${GPUS:-1} --master-port $((29500 + RANDOM % 1000)) sample_predictions.py \
  --checkpoint "${checkpoint}" --data "${data}" --split "${splits}" --num_samples ${SAMPLES:-100} \
  --out "samples/${dataset}-{split}/${name}" --max_train_queries ${QUERIES:-2000} ${BATCH:+--effective_batch_size ${BATCH}}
for split in ${splits//,/ }; do
  python scripts/analyze_multi_samples.py --task ${dataset} --data "${data}" --samples "samples/${dataset}-${split}" \
    --out "results/${dataset}-${split}"
done
