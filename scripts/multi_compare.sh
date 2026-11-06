#!/bin/bash
# Train one method on one of the five multi-solution datasets (built by scripts/build_multi_datasets.sh), with the
# methods, step budget and options of scripts/maze_compare.sh (see there: gram, gram_prior_bestofk, trm, trm_minloss,
# ..., and RESUME, SEED, STEPS, KL_COEF, ROLLOUTS). Checkpoints go to checkpoints/<Dataset>-multi/<run>.
#
# Usage: bash scripts/multi_compare.sh <maze|nqueens|sudoku|graph|sat> <method>
#   e.g. bash scripts/multi_compare.sh sudoku gram_prior_bestofk     (prior-only GRAM, best of 8 prior samples)
set -e
dataset=$1
shift
case $dataset in
  maze)    export ROOT=data/maze-15 PROJECT=Maze-15-compare ;;  # the maze runs so far, so GRAM baselines line up
  nqueens) export ROOT=data/nqueens-10-s16 PROJECT=Nqueens-multi ;;
  sudoku)  export ROOT=data/sudoku-multi PROJECT=Sudoku-multi ;;
  graph)   export ROOT=data/graph-coloring-multi PROJECT=Graph-coloring-multi ;;
  sat)
    export ROOT=data/sat-16-wide PROJECT=Sat-16-wide
    # Learned positions, so the 3-token clause slots have fixed embeddings (as in scripts/sat_compare.sh)
    export EXTRA_ARGS="arch.pos_encodings=learned ${EXTRA_ARGS}" ;;
  *) echo "Unknown dataset: ${dataset} (maze, nqueens, sudoku, graph, sat)"; exit 1 ;;
esac
if [ ! -f "${ROOT}-1oml/train/dataset.json" ]; then
  echo "${ROOT}-1oml is missing: build it with bash scripts/build_multi_datasets.sh ${dataset}"
  exit 1
fi
exec bash "$(dirname "$0")/maze_compare.sh" "$@"
