#!/bin/bash
# Sample and analyze the maze comparison runs (scripts/maze_compare.sh) on the shared test split.
#
# Usage: bash scripts/maze_hidden_constraints.sh
#
# 1. Samples every finished run in checkpoints/Maze-15-compare/ on the test split of data/maze-15-1oml:
#    Stochastic runs (GRAM, and noisy TRM: trm_setmatch, trm_noise_minloss) get GRAM_SAMPLES samples per maze
#    (default 100); other TRM runs, deterministic, get 1. Runs listed in
#    TEMPERATURE_RUNS (default trm_minloss_seed42) are also sampled GRAM_SAMPLES times at TEMPERATURE (default 1.0).
# 2. Analyzes all samples into results/maze_15/ (scripts/analyze_maze_samples.py).
#
# Resumable: samples are saved in shards, so rerunning after an interruption continues where it stopped. Uses
# every visible GPU (or NPROC); runs still training are skipped and picked up by a later rerun. Lower
# EFFECTIVE_BATCH_SIZE (mazes x samples per forward pass) if the GPU runs out of memory.
set -o pipefail

runs_dir=checkpoints/Maze-15-compare
data=data/maze-15-1oml
samples_dir=samples/maze_15
results_dir=results/maze_15
gram_samples=${GRAM_SAMPLES:-100}
temperature=${TEMPERATURE:-1.0}
temperature_runs=${TEMPERATURE_RUNS:-trm_minloss_seed42}
nproc=${NPROC:-$(nvidia-smi -L | wc -l)}
mkdir -p logs

sample() {
  local name=$1 ckpt=$2 num_samples=$3 temp=$4
  echo "== $(date '+%F %T') Sampling ${name}: ${num_samples} sample(s) per maze, temperature ${temp}"
  torchrun --nproc-per-node ${nproc} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 sample_predictions.py \
    --checkpoint "${ckpt}" --data "${data}" --out "${samples_dir}/${name}" \
    --num_samples ${num_samples} --temperature ${temp} \
    --effective_batch_size ${EFFECTIVE_BATCH_SIZE:-7680} --shard_size ${SHARD_SIZE:-256} \
    2>&1 | tee -a "logs/maze_15_sampling_${name}.log"
}

failed=()
for run_dir in "${runs_dir}"/*/; do
  run=$(basename "${run_dir}")

  # The finished model: final/, or a last step_* that completed the run
  if [ -f "${run_dir}final/checkpoint.pt" ]; then
    ckpt=${run_dir}final/checkpoint.pt
  else
    last=$(ls -d "${run_dir}"step_*/ 2>/dev/null | sort -V | tail -n 1)
    expected=$(python scripts/expected_final_step.py "${run_dir}all_config.yaml" 2>/dev/null)
    step=$(basename "${last%/}")
    step=${step#step_}
    if [ -z "${last}" ] || [ -z "${expected}" ] || [ "${step}" -lt "${expected}" ]; then
      echo "== ${run}: not finished, skipping"
      continue
    fi
    ckpt=${last}checkpoint.pt
  fi

  if [[ ${run} == gram* || ${run} == trm_setmatch* || ${run} == trm_noise* ]]; then
    sample "${run}" "${ckpt}" ${gram_samples} 0 || failed+=("${run}")
  else
    sample "${run}" "${ckpt}" 1 0 || failed+=("${run}")
  fi
  if [[ " ${temperature_runs} " == *" ${run} "* ]]; then
    sample "${run}_T${temperature}" "${ckpt}" ${gram_samples} ${temperature} || failed+=("${run}_T${temperature}")
  fi
done

echo
python scripts/analyze_maze_samples.py --data "${data}" --samples "${samples_dir}" --out "${results_dir}" \
  || failed+=("analysis")

if [ ${#failed[@]} -gt 0 ]; then
  echo "== Failed: ${failed[*]}"
  exit 1
fi
