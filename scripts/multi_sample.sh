#!/bin/bash
# Sample a trained run on one of the five multi-solution datasets and score it with scripts/analyze_multi_samples.py.
#
# Usage: bash scripts/multi_sample.sh <maze|nqueens|sudoku|graph|sat> <run> [test|train]
#   e.g. bash scripts/multi_sample.sh sudoku gram_prior_bestofk_seed42
#        bash scripts/multi_sample.sh graph gram_prior_bestofk_seed42 train   (held-out solutions on training inputs)
#
# SAMPLES per input (default 100). QUERIES limits the inputs (default: all 2000 test inputs; the first 2000 training
# inputs for train, which are in random order). STEP picks a checkpoint other than the final one (e.g. STEP=41660 for
# checkpoints/<project>/<run>/step_41660). GPUS for torchrun (default 1). BATCH sets queries x samples per forward
# pass (default 15360; lower it to share a GPU with a training run).
# Samples go to samples/<dataset>-<split>/<run>[_step<STEP>], results to results/<dataset>-<split>.
set -e
dataset=$1
run=$2
split=${3:-test}
case $dataset in
  maze)    data=data/maze-15-1oml project=Maze-15-compare ;;
  nqueens) data=data/nqueens-10-s16-1oml project=Nqueens-multi ;;
  sudoku)  data=data/sudoku-multi-1oml project=Sudoku-multi ;;
  graph)   data=data/graph-coloring-multi-1oml project=Graph-coloring-multi ;;
  sat)     data=data/sat-16-wide-1oml project=Sat-16-wide ;;
  *) echo "Unknown dataset: ${dataset} (maze, nqueens, sudoku, graph, sat)"; exit 1 ;;
esac
step_dir=final
name=${run}
if [ -n "${STEP}" ]; then step_dir=step_${STEP}; name=${run}_step${STEP}; fi
checkpoint=checkpoints/${project}/${run}/${step_dir}/checkpoint.pt
[ -f "${checkpoint}" ] || { echo "No checkpoint at ${checkpoint}"; ls "checkpoints/${project}/${run}" 2>/dev/null; exit 1; }

queries=${QUERIES:-}
[ "${split}" = train ] && queries=${QUERIES:-2000}
out=samples/${dataset}-${split}/${name}
torchrun --nproc-per-node ${GPUS:-1} --master-port $((29500 + RANDOM % 1000)) sample_predictions.py \
  --checkpoint "${checkpoint}" --data "${data}" --split "${split}" --num_samples ${SAMPLES:-100} --out "${out}" \
  ${queries:+--max_queries ${queries}} ${BATCH:+--effective_batch_size ${BATCH}}
python scripts/analyze_multi_samples.py --task ${dataset} --data "${data}" --samples "samples/${dataset}-${split}" \
  --out "results/${dataset}-${split}"
