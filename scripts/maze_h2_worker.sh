#!/bin/bash
# Run the H2 experiments (step-by-step decoding) and the evaluation-only tests on one GPU, from a shared queue.
#
# Usage: start one worker per GPU, in any number of jobs:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/maze_h2_worker.sh 2>&1 | tee -a logs/maze_h2_worker_$(hostname)_0.log
#
# Workers take tasks in order. A task is claimed atomically (samples/maze_15/.h2_claims/<task>), so each runs once;
# a task whose checkpoint isn't trained yet is skipped and retried later, so decoding the hint-trained model starts
# as soon as its training finishes on another GPU. Interrupted tasks release their claim and resume where they
# stopped: training continues from its last checkpoint (RESUME), sampling and decoding from their saved shards.
# A failed task is marked failed and not retried (remove its claim to retry). The last worker to finish runs the
# analysis (results/maze_15/summary.md) and the per-cell calibration check (results/maze_15/marginals/).
#
# Tasks, in priority order:
#   training: trm_hints (TRM with path hints), GRAM at KL 0.015 for 100k steps (is the trade-off undertraining?)
#   step 0:   per-cell PATH probability vs exact path shares, for the existing TRM models
#   H2:       walk decoding (walk_decode.py) of naive TRM (sample x100, greedy), controls (Random, MinLoss), and the
#             hint-trained model (hint and start conditioning)
#   pending:  standard sampling of every finished run (scripts/maze_hidden_constraints.sh: KL budget, seed 1, ...)
#   H1/H4:    PTRM-style latent noise on MinLoss and I-ExplR, with halting-logit selection; GRAM resampled with
#             halting logits saved
#   H6:       twice the recursion steps at test time (depth scaling)
set -o pipefail

runs_dir=checkpoints/Maze-15-compare
data=data/maze-15-1oml
samples=samples/maze_15
claims=${samples}/.h2_claims
mkdir -p "${claims}" logs

# The finished checkpoint of a run, or nothing if it is still training
finished_ckpt() {
  local run_dir=${runs_dir}/$1/
  if [ -f "${run_dir}final/checkpoint.pt" ]; then echo "${run_dir}final/checkpoint.pt"; return; fi
  local last expected step
  last=$(ls -d "${run_dir}"step_*/ 2>/dev/null | sort -V | tail -n 1)
  expected=$(python scripts/expected_final_step.py "${run_dir}all_config.yaml" 2>/dev/null)
  step=$(basename "${last%/}"); step=${step#step_}
  if [ -n "${last}" ] && [ -n "${expected}" ] && [ "${step}" -ge "${expected}" ]; then echo "${last}checkpoint.pt"; fi
}

# train <run> <env...> -- <maze_compare.sh method>: resumes from the last checkpoint if there is one
train() {
  local run=$1; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  [ -n "$(finished_ckpt "${run}")" ] && return 0
  # A previous attempt may have left a finished-training claim of maze_compare.sh; ours supersedes it
  rm -rf "${runs_dir}/.claims/${run}"
  if ls -d "${runs_dir}/${run}"/step_*/ >/dev/null 2>&1; then envs+=(RESUME=1); fi
  env "${envs[@]}" NPROC=1 bash scripts/maze_compare.sh "$@"
}

sample() {  # sample <run> <out name> <num samples> [sample_predictions.py options]
  local run=$1 name=$2 num=$3; shift 3
  torchrun --nproc-per-node 1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 sample_predictions.py \
    --checkpoint "$(finished_ckpt "${run}")" --data ${data} --out "${samples}/${name}" --num_samples "${num}" \
    --effective_batch_size ${EFFECTIVE_BATCH_SIZE:-7680} --shard_size 256 "$@"
}

walk() {  # walk <run> <out name> <num samples> <mode> <choice> [walk_decode.py options]
  local run=$1 name=$2 num=$3 mode=$4 choice=$5; shift 5
  torchrun --nproc-per-node 1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 walk_decode.py \
    --checkpoint "$(finished_ckpt "${run}")" --data ${data} --out "${samples}/${name}" --num_samples "${num}" \
    --mode "${mode}" --choice "${choice}" "$@"
}

marginals() {  # marginals <out dir> <run>...: the runs that have finished
  local out=$1; shift
  local specs=() run ckpt
  for run in "$@"; do
    ckpt=$(finished_ckpt "${run}")
    [ -n "${ckpt}" ] && specs+=("${run}=${ckpt}")
  done
  python scripts/maze_marginals.py --data ${data} --out "${out}" --runs "${specs[@]}"
}

# name | runs that must have finished training (comma-separated, - for none) | command
tasks=(
  "train_trm_hints|-|train trm_hints_seed42 -- trm_hints"
  "train_gram_kl0.015_100k|-|train gram_kl0.015_steps100000_seed42 KL_COEF=0.015 STEPS=100000 -- gram"
  "marginals_existing|trm_seed42,trm_random_seed42,trm_minloss_seed42,trm_iexplr_seed42|marginals results/maze_15/marginals trm_seed42 trm_random_seed42 trm_minloss_seed42 trm_iexplr_seed42"
  "walk_trm|trm_seed42|walk trm_seed42 trm_seed42_walk 100 start sample"
  "walk_trm_greedy|trm_seed42|walk trm_seed42 trm_seed42_walkgreedy 1 start greedy"
  "walk_random|trm_random_seed42|walk trm_random_seed42 trm_random_seed42_walk 20 start sample"
  "walk_minloss|trm_minloss_seed42|walk trm_minloss_seed42 trm_minloss_seed42_walk 20 start sample"
  "walk_minloss_greedy|trm_minloss_seed42|walk trm_minloss_seed42 trm_minloss_seed42_walkgreedy 1 start greedy"
  "sample_pending|-|NPROC=1 bash scripts/maze_hidden_constraints.sh"
  "marginals_hints|trm_hints_seed42,trm_seed42|marginals results/maze_15/marginals_hints trm_hints_seed42 trm_seed42"
  "walk_hints|trm_hints_seed42|walk trm_hints_seed42 trm_hints_seed42_walkhint 100 hint sample"
  "walk_hints_greedy|trm_hints_seed42|walk trm_hints_seed42 trm_hints_seed42_walkhintgreedy 1 hint greedy"
  "walk_hints_start|trm_hints_seed42|walk trm_hints_seed42 trm_hints_seed42_walk 100 start sample"
  "sample_hints|trm_hints_seed42|sample trm_hints_seed42 trm_hints_seed42 1"
  "noise_minloss_1.0|trm_minloss_seed42|sample trm_minloss_seed42 trm_minloss_seed42_noise1.0 100 --latent_noise 1.0"
  "noise_minloss_0.3|trm_minloss_seed42|sample trm_minloss_seed42 trm_minloss_seed42_noise0.3 100 --latent_noise 0.3"
  "noise_iexplr_1.0|trm_iexplr_seed42|sample trm_iexplr_seed42 trm_iexplr_seed42_noise1.0 100 --latent_noise 1.0"
  "gram_selfq|gram_kl0.015_seed42|sample gram_kl0.015_seed42 gram_kl0.015_seed42_q 100"
  "depth_minloss|trm_minloss_seed42|sample trm_minloss_seed42 trm_minloss_seed42_depth32 1 --halt_steps 32"
  "depth_iexplr|trm_iexplr_seed42|sample trm_iexplr_seed42 trm_iexplr_seed42_depth32 1 --halt_steps 32"
  "depth_trm|trm_seed42|sample trm_seed42 trm_seed42_depth32 1 --halt_steps 32"
  "depth_gram|gram_kl0.015_seed42|sample gram_kl0.015_seed42 gram_kl0.015_seed42_depth32 100 --halt_steps 32"
  "sample_gram_100k|gram_kl0.015_steps100000_seed42|sample gram_kl0.015_steps100000_seed42 gram_kl0.015_steps100000_seed42 100"

  # Round 2: equal training (TRM baselines at 100k steps, as GRAM), GRAM on to 200k steps and its seed 1 to 100k.
  # All continue a finished run (CONTINUE), so they cost only the extra steps.
  "cont_gram_200k|gram_kl0.015_steps100000_seed42|train gram_kl0.015_steps200000_seed42 CONTINUE=gram_kl0.015_steps100000_seed42 KL_COEF=0.015 STEPS=200000 -- gram"
  "cont_minloss_100k|trm_minloss_seed42|train trm_minloss_steps100000_seed42 CONTINUE=trm_minloss_seed42 STEPS=100000 -- trm_minloss"
  "cont_iexplr_100k|trm_iexplr_seed42|train trm_iexplr_steps100000_seed42 CONTINUE=trm_iexplr_seed42 STEPS=100000 -- trm_iexplr"
  "cont_trm_100k|trm_seed42|train trm_steps100000_seed42 CONTINUE=trm_seed42 STEPS=100000 -- trm"
  "cont_gram_seed1_100k|gram_kl0.015_seed1|train gram_kl0.015_steps100000_seed1 CONTINUE=gram_kl0.015_seed1 KL_COEF=0.015 SEED=1 STEPS=100000 -- gram"
  "walk_iexplr_greedy|trm_iexplr_seed42|walk trm_iexplr_seed42 trm_iexplr_seed42_walkgreedy 1 start greedy"
  "sample_minloss_100k|trm_minloss_steps100000_seed42|sample trm_minloss_steps100000_seed42 trm_minloss_steps100000_seed42 1"
  "sample_iexplr_100k|trm_iexplr_steps100000_seed42|sample trm_iexplr_steps100000_seed42 trm_iexplr_steps100000_seed42 1"
  "sample_trm_100k|trm_steps100000_seed42|sample trm_steps100000_seed42 trm_steps100000_seed42 1"
  "walk_trm_100k|trm_steps100000_seed42|walk trm_steps100000_seed42 trm_steps100000_seed42_walk 100 start sample"
  "walk_trm_100k_greedy|trm_steps100000_seed42|walk trm_steps100000_seed42 trm_steps100000_seed42_walkgreedy 1 start greedy"
  "walk_minloss_100k_greedy|trm_minloss_steps100000_seed42|walk trm_minloss_steps100000_seed42 trm_minloss_steps100000_seed42_walkgreedy 1 start greedy"
  "walk_iexplr_100k_greedy|trm_iexplr_steps100000_seed42|walk trm_iexplr_steps100000_seed42 trm_iexplr_steps100000_seed42_walkgreedy 1 start greedy"
  "walk_gram_100k|gram_kl0.015_steps100000_seed42|walk gram_kl0.015_steps100000_seed42 gram_kl0.015_steps100000_seed42_walk 20 start sample"
  "sample_gram_200k|gram_kl0.015_steps200000_seed42|sample gram_kl0.015_steps200000_seed42 gram_kl0.015_steps200000_seed42 100"
  "sample_gram_seed1_100k|gram_kl0.015_steps100000_seed1|sample gram_kl0.015_steps100000_seed1 gram_kl0.015_steps100000_seed1 100"
)

current=""
trap 'if [ -n "${current}" ]; then rm -rf "${claims}/${current}"; fi; exit 130' INT TERM

while true; do
  for task in "${tasks[@]}"; do
    IFS='|' read -r name needs command <<< "${task}"
    [ -d "${claims}/${name}" ] && continue
    if [ "${needs}" != "-" ]; then
      ready=1
      for run in ${needs//,/ }; do [ -z "$(finished_ckpt "${run}")" ] && ready=0; done
      [ ${ready} -eq 0 ] && continue
    fi
    mkdir "${claims}/${name}" 2>/dev/null || continue
    current=${name}
    echo "$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-all} started $(date '+%F %T')" > "${claims}/${name}/owner"
    echo "== $(date '+%F %T') ${name}: ${command}"
    if eval "${command}" 2>&1 | tee -a "logs/maze_h2_${name}.log"; then
      touch "${claims}/${name}/finished"
      echo "== $(date '+%F %T') ${name} finished"
    else
      touch "${claims}/${name}/failed"
      echo "== $(date '+%F %T') ${name} FAILED (see logs/maze_h2_${name}.log; remove ${claims}/${name} to retry)"
    fi
    current=""
    continue 2  # back to the top: higher-priority tasks may have become ready
  done

  # Nothing to start: wait for prerequisites, or finish once every task has been run
  unfinished=0
  for task in "${tasks[@]}"; do
    name=${task%%|*}
    if [ ! -f "${claims}/${name}/finished" ] && [ ! -f "${claims}/${name}/failed" ]; then unfinished=1; fi
  done
  if [ ${unfinished} -eq 0 ]; then
    # One analysis per version of the task list, so adding tasks later analyzes again
    analysis=analysis_${#tasks[@]}
    if mkdir "${claims}/${analysis}" 2>/dev/null; then
      echo "== $(date '+%F %T') All tasks done; analyzing"
      python scripts/analyze_maze_samples.py --data ${data} --samples ${samples} --out results/maze_15 \
        2>&1 | tee -a logs/maze_h2_analysis.log
      python scripts/analyze_walk_failures.py --data ${data} --samples ${samples} --out results/maze_15 \
        2>&1 | tee -a logs/maze_h2_analysis.log
      touch "${claims}/${analysis}/finished"
    fi
    echo "== $(date '+%F %T') Nothing left to do"
    exit 0
  fi
  echo "== $(date '+%F %T') Waiting for runs still training or tasks running elsewhere"
  sleep 600
done
