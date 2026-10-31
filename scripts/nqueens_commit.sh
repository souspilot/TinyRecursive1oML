#!/bin/bash
# Decode the finished 10x10 N-Queens comparison runs one queen at a time (commit_decode.py), then analyze every
# run in samples/nqueens_10 (one-shot samples from scripts/hidden_constraints.sh included).
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/nqueens_commit.sh 2>&1 | tee -a logs/nqueens_commit.log
#
# Tests whether the one-of-many failure of naive training (1% exact with all cells decided at once) is a decoding
# failure: naive and Random TRM decoded by committing queens, against MinLoss, I-ExplR and GRAM. Resumable (saved
# shards are kept); runs that haven't finished training are skipped. Results: results/nqueens_10_commit/summary.md.
set -o pipefail
runs_dir=checkpoints/Nqueens-10-compare
data=data/nqueens-10-1oml
samples=samples/nqueens_10
mkdir -p logs

finished_ckpt() {
  local run_dir=${runs_dir}/$1/
  if [ -f "${run_dir}final/checkpoint.pt" ]; then echo "${run_dir}final/checkpoint.pt"; return; fi
  local last expected step
  last=$(ls -d "${run_dir}"step_*/ 2>/dev/null | sort -V | tail -n 1)
  expected=$(python scripts/expected_final_step.py "${run_dir}all_config.yaml" 2>/dev/null)
  step=$(basename "${last%/}"); step=${step#step_}
  if [ -n "${last}" ] && [ -n "${expected}" ] && [ "${step}" -ge "${expected}" ]; then echo "${last}checkpoint.pt"; fi
}

failed=()
decode() {  # decode <run> <out name> <num samples> [commit_decode.py options]
  local run=$1 name=$2 num=$3; shift 3
  local ckpt
  ckpt=$(finished_ckpt "${run}")
  if [ -z "${ckpt}" ]; then echo "== ${run}: no finished checkpoint, skipping ${name}"; return; fi
  echo "== $(date '+%F %T') ${name}"
  torchrun --nproc-per-node 1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 commit_decode.py \
    --checkpoint "${ckpt}" --data ${data} --out "${samples}/${name}" --num_samples "${num}" "$@" || failed+=("${name}")
}

decode trm_seed42 trm_seed42_commitgreedy 1 --choice greedy
decode trm_seed42 trm_seed42_commit 100 --choice sample
decode trm_random_seed42 trm_random_seed42_commitgreedy 1 --choice greedy
decode trm_random_seed42 trm_random_seed42_commit 20 --choice sample
decode trm_minloss_seed42 trm_minloss_seed42_commitgreedy 1 --choice greedy
decode trm_iexplr_seed42 trm_iexplr_seed42_commitgreedy 1 --choice greedy
decode trm_unique_seed42 trm_unique_seed42_commitgreedy 1 --choice greedy
decode trm_seed42 trm_seed42_commitrandom 20 --choice sample --order random
decode gram_seed42 gram_seed42_commit 20 --choice sample
for seed in 1; do
  decode trm_minloss_seed${seed} trm_minloss_seed${seed}_commitgreedy 1 --choice greedy
  decode trm_iexplr_seed${seed} trm_iexplr_seed${seed}_commitgreedy 1 --choice greedy
done

echo "== $(date '+%F %T') Analyzing"
python scripts/analyze_nqueens_samples.py --data ${data} --samples ${samples} --out results/nqueens_10_commit || failed+=("analysis")
if [ ${#failed[@]} -gt 0 ]; then
  echo "== Failed: ${failed[*]}"
  exit 1
fi
