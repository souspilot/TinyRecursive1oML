#!/bin/bash
# Compare GRAM and TRM with one-of-many (1oML) objectives on random 3-SAT with 16 variables
# (dataset/build_sat_dataset.py), with the same methods, step budget and options as scripts/maze_compare.sh.
# Positions are learned rather than RoPE, so the 3-token clause slots have fixed embeddings. Training-time
# accuracy only counts the (up to 16) stored solutions; check samples with evaluators/sat_utils.py.
#
# Usage: bash scripts/sat_compare.sh <method>     (e.g. trm, trm_random, trm_unique, trm_minloss, trm_iexplr,
#                                                  gram with KL_COEF set)
#
# Datasets (build once, a few minutes on a multi-core node):
#   python dataset/build_sat_dataset.py --num-vars 16 --num-train 100000 --num-test 2000 --output-dir data/sat-16
export ROOT=${ROOT:-data/sat-16}
export PROJECT=${PROJECT:-Sat-16-compare}
export EXTRA_ARGS="arch.pos_encodings=learned ${EXTRA_ARGS}"
exec bash "$(dirname "$0")/maze_compare.sh" "$@"
