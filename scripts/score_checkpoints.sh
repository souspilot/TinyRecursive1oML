#!/bin/bash
# Score a run's saved checkpoints on the validation split, to see how validity and diversity change during training.
# Read the result with scripts/checkpoint_trends.py.
#
# Usage: bash scripts/score_checkpoints.sh <maze|nqueens|sudoku|graph> <run>
#
# Env: EVERY scores every EVERY-th checkpoint counted back from the last, so the last is always included (default 2);
# QUERIES validation inputs (default 500); SAMPLES per input (default 100). A failed checkpoint is reported and skipped,
# and rerunning only redoes what is missing.
set -o pipefail
dataset=$1
run=$2
case $dataset in
  maze)    project=Maze-multi ;;
  nqueens) project=Nqueens-multi ;;
  sudoku)  project=Sudoku-multi ;;
  graph)   project=Graph-coloring-multi ;;
  *) echo "Unknown dataset: ${dataset} (maze, nqueens, sudoku, graph)"; exit 1 ;;
esac
every=${EVERY:-2}
steps=$(ls -d checkpoints/${project}/${run}/step_* | sed 's/.*step_//' | grep -E '^[0-9]+$' | sort -nr \
  | awk -v e=${every} 'NR % e == 1 || e == 1' | sort -n)
[ -n "${steps}" ] || { echo "No step checkpoints for ${run} in ${project}"; exit 1; }
echo "${dataset}: scoring steps" ${steps}
for step in ${steps}; do
  # Only the load line: each evaluation reprints the tables of every earlier one
  STEP=${step} QUERIES=${QUERIES:-500} bash scripts/multi_evaluate.sh ${dataset} ${run} val \
    | awk '/Loaded/' || echo "FAILED ${dataset} step ${step}"
done
