#!/bin/bash
# Sample the finished 3-SAT comparison runs (scripts/sat_compare.sh) every way we decode N-Queens and mazes, then
# analyze them all against the exact solution sets (scripts/analyze_sat_samples.py -> results/sat_16/summary.md).
#
# Usage (one worker per GPU; workers share the task list through claims, so start as many as there are GPUs):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/sat_sampling.sh
#
# Per model: one shot (100 samples: GRAM's latent, the others' deterministic answer), test-time diversity (noise on
# the initial latent plus random equivalent formulas: variables renamed and negated, clauses reordered), and
# commit-and-re-run decoding (sat_commit_decode.py), greedy and sampled. Sampled commit decoding never forces a value
# (--certain 1.01), so its log_q gives Knuth's estimate of each formula's solution count. Tasks whose run hasn't
# finished training are skipped (rerun later). Resumable: saved shards are kept.
set -o pipefail
project=${PROJECT:-Sat-16-compare}
data=${DATA:-data/sat-16-1oml}
samples=${SAMPLES:-samples/sat_16}
results=${RESULTS:-results/sat_16}
gram=gram_kl${GRAM_KL:-0.015}_seed42  # GRAM_KL: the KL_COEF its run was trained with
batch="--effective_batch_size ${EFFECTIVE_BATCH_SIZE:-7680}"  # As mazes, of similar length; twice this overflows 80 GB
claims=${samples}/.claims
mkdir -p logs ${claims}

ckpt() {  # ckpt <run>: the final checkpoint, or nothing if the run hasn't finished
  local f=checkpoints/${project}/$1/final/checkpoint.pt
  [ -f "${f}" ] && echo "${f}"
}

launch() {  # launch <script> <checkpoint> <out name> [options]
  local script=$1 checkpoint=$2 name=$3; shift 3
  torchrun --nproc-per-node 1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 ${script} \
    --checkpoint "${checkpoint}" --data ${data} --out "${samples}/${name}" "$@"
}

tasks=(  # <script> <run> <out name> [options]
  "sample_predictions.py trm_seed42 trm_seed42 --num_samples 1"
  "sample_predictions.py trm_minloss_seed42 trm_minloss_seed42 --num_samples 1"
  "sample_predictions.py trm_iexplr_seed42 trm_iexplr_seed42 --num_samples 1"
  "sample_predictions.py ${gram} ${gram} --num_samples 100 ${batch}"
  "sample_predictions.py trm_seed42 trm_seed42_views_init1 --num_samples 100 --sat_views --init_noise 1 ${batch}"
  "sample_predictions.py trm_minloss_seed42 trm_minloss_seed42_views_init1 --num_samples 100 --sat_views --init_noise 1 ${batch}"
  "sample_predictions.py trm_iexplr_seed42 trm_iexplr_seed42_views_init1 --num_samples 100 --sat_views --init_noise 1 ${batch}"
  "sample_predictions.py ${gram} ${gram}_views --num_samples 100 --sat_views ${batch}"
  "sat_commit_decode.py trm_seed42 trm_seed42_commitgreedy --num_samples 1 --choice greedy"
  "sat_commit_decode.py trm_minloss_seed42 trm_minloss_seed42_commitgreedy --num_samples 1 --choice greedy"
  "sat_commit_decode.py trm_iexplr_seed42 trm_iexplr_seed42_commitgreedy --num_samples 1 --choice greedy"
  "sat_commit_decode.py ${gram} ${gram}_commitgreedy --num_samples 1 --choice greedy"
  "sat_commit_decode.py trm_seed42 trm_seed42_commit --num_samples 20 --choice sample --certain 1.01"
  "sat_commit_decode.py trm_minloss_seed42 trm_minloss_seed42_commit --num_samples 20 --choice sample --certain 1.01"
)

failed=()
skipped=0
for i in "${!tasks[@]}"; do
  read -r -a t <<< "${tasks[$i]}"
  name=${t[2]}
  [ -e "${claims}/${name}.done" ] && continue
  checkpoint=$(ckpt "${t[1]}")
  if [ -z "${checkpoint}" ]; then skipped=$((skipped + 1)); continue; fi
  mkdir "${claims}/${name}" 2>/dev/null || continue  # another worker has it
  trap 'rmdir "${claims}/${name}" 2>/dev/null; exit 130' INT TERM
  echo "== $(date '+%F %T') ${name} (GPU ${CUDA_VISIBLE_DEVICES:-unset})"
  if launch "${t[0]}" "${checkpoint}" "${name}" "${t[@]:3}" > "logs/sat_sampling_${name}.log" 2>&1; then
    touch "${claims}/${name}.done"
  else
    failed+=("${name}")
  fi
  rmdir "${claims}/${name}"
  trap - INT TERM
done

[ ${skipped} -gt 0 ] && echo "== Skipped ${skipped} tasks whose run hasn't finished training"
if [ ${#failed[@]} -gt 0 ]; then echo "== Failed (see logs/sat_sampling_<name>.log): ${failed[*]}"; exit 1; fi
# Analyze whatever is complete; with several workers, the last one to finish gives the full summary
echo "== $(date '+%F %T') Analyzing"
python scripts/analyze_sat_samples.py --data ${data} --samples ${samples} --out ${results}
