#!/bin/bash
# Hidden-constraint experiment on the N-Queens comparison runs (scripts/nqueens_compare.sh).
#
# Usage: bash scripts/hidden_constraints.sh <8|10>
#
# 1. Samples every finished run in checkpoints/Nqueens-<N>-compare/ on the test split of data/nqueens-<N>-1oml:
#    GRAM runs get GRAM_SAMPLES samples per query (default 100); TRM runs, deterministic, get 1. Runs listed in
#    TEMPERATURE_RUNS (default trm_minloss_seed42) are also sampled GRAM_SAMPLES times at TEMPERATURE (default 1.0),
#    as a baseline that makes a single-answer model stochastic.
# 2. Analyzes all samples into results/hidden_constraints_<N>/ (summary.md, per-run JSON, figures).
#
# Resumable: samples are saved in shards, so rerunning after an interruption (e.g. walltime) continues where it
# stopped. Runs on every visible GPU (or NPROC); runs still training are skipped and picked up by a later rerun.
# Lower EFFECTIVE_BATCH_SIZE (queries x samples per forward pass) if the GPU runs out of memory.
set -o pipefail

n=$1
case $n in
  8|10) ;;
  *) echo "Board size must be 8 or 10"; exit 1 ;;
esac

runs_dir=checkpoints/Nqueens-${n}-compare
data=data/nqueens-${n}-1oml
samples_dir=samples/nqueens_${n}
results_dir=results/hidden_constraints_${n}
gram_samples=${GRAM_SAMPLES:-100}
temperature=${TEMPERATURE:-1.0}
temperature_runs=${TEMPERATURE_RUNS:-trm_minloss_seed42}
nproc=${NPROC:-$(nvidia-smi -L | wc -l)}
mkdir -p logs

sample() {
  local name=$1 ckpt=$2 num_samples=$3 temp=$4
  echo "== $(date '+%F %T') Sampling ${name}: ${num_samples} sample(s) per query, temperature ${temp}"
  torchrun --nproc-per-node ${nproc} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 sample_predictions.py \
    --checkpoint "${ckpt}" --data "${data}" --out "${samples_dir}/${name}" \
    --num_samples ${num_samples} --temperature ${temp} \
    --effective_batch_size ${EFFECTIVE_BATCH_SIZE:-15360} --shard_size ${SHARD_SIZE:-512} \
    2>&1 | tee -a "logs/hidden_constraints_${n}_${name}.log"
}

failed=()
for run_dir in "${runs_dir}"/*/; do
  run=$(basename "${run_dir}")

  # The finished model: final/, or for runs saved before final/ existed, a last step_* that completed the run
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

  if [[ ${run} == gram* ]]; then
    sample "${run}" "${ckpt}" ${gram_samples} 0 || failed+=("${run}")
  else
    sample "${run}" "${ckpt}" 1 0 || failed+=("${run}")
  fi
  if [[ " ${temperature_runs} " == *" ${run} "* ]]; then
    sample "${run}_T${temperature}" "${ckpt}" ${gram_samples} ${temperature} || failed+=("${run}_T${temperature}")
  fi
done

echo
python scripts/analyze_nqueens_samples.py --data "${data}" --samples "${samples_dir}" --out "${results_dir}" \
  || failed+=("analysis")

if [ ${#failed[@]} -gt 0 ]; then
  echo "== Failed: ${failed[*]}"
  exit 1
fi
