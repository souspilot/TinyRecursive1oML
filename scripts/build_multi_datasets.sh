#!/bin/bash
# Build the five multi-solution datasets of the project (see scripts/multi_compare.sh), skipping any that exist.
# Every dataset stores at most 16 solutions per training input and keeps the exact count, so inputs with more
# solutions have held-out ones (for testing whether a model produces solutions it never saw).
#
#   maze      15x15 mazes, 1 to 924 shortest paths          (dataset/build_maze_dataset.py)
#   nqueens   10x10 N-Queens, 1 to 3 queens given           (dataset/build_nqueens_dataset.py, capped at 16)
#   sudoku    9x9 Sudoku, 26 to 50 clues, up to 1024 sols    (dataset/build_sudoku_multi_dataset.py)
#   graph     3-colouring of 10-node graphs, 1 to 9842 sols  (dataset/build_graph_coloring_multi_dataset.py)
#   sat       random 3-SAT, 16 variables, clause density 1 to 4.5 (dataset/build_sat_dataset.py)
#
# Usage: bash scripts/build_multi_datasets.sh [dataset ...]     (default: all five; CPU only, uses every core)
set -e
cd "$(dirname "$0")/.."
datasets=${*:-maze nqueens sudoku graph sat}

built() { [ -f "$1/test/dataset.json" ] && echo "$1 exists, skipping"; }

for d in ${datasets}; do
  case $d in
    maze)
      built data/maze-15-1oml || python dataset/build_maze_dataset.py --size 15 --num-train 100000 --num-test 2000 --output-dir data/maze-15 ;;
    nqueens)
      # Three builds with the same seed share the split and the stored solutions
      base=data/nqueens-10-s16
      built ${base} || python dataset/build_nqueens_dataset.py --n 10 --remove-counts 7 8 9 --max-stored-solutions 16 --output-dir ${base}
      built ${base}-1oml || python dataset/build_nqueens_dataset.py --n 10 --remove-counts 7 8 9 --max-stored-solutions 16 --output-dir ${base}-1oml --one-of-many
      built ${base}-unique || python dataset/build_nqueens_dataset.py --n 10 --remove-counts 7 8 9 --max-stored-solutions 16 --output-dir ${base}-unique --one-of-many --train-unique-only ;;
    sudoku)
      built data/sudoku-multi-1oml || python dataset/build_sudoku_multi_dataset.py --num-train 100000 --num-test 2000 --output-dir data/sudoku-multi ;;
    graph)
      built data/graph-coloring-multi-1oml || python dataset/build_graph_coloring_multi_dataset.py --num-train 100000 --num-test 2000 --output-dir data/graph-coloring-multi ;;
    sat)
      built data/sat-16-wide-1oml || python dataset/build_sat_dataset.py --num-vars 16 --num-train 100000 --num-test 2000 --alpha-min 1.0 --output-dir data/sat-16-wide ;;
    *) echo "Unknown dataset: $d (maze, nqueens, sudoku, graph, sat)"; exit 1 ;;
  esac
done
echo "Done: ${datasets}"
