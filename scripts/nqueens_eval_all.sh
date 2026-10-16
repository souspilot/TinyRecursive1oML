#!/bin/bash
# Run pass@k evaluation on every N-Queens comparison run (scripts/nqueens_compare.sh) and summarize them.
#
# Usage: bash scripts/nqueens_eval_all.sh <8|10> [num_samples]
#
# Each run in checkpoints/Nqueens-<N>-compare/ is evaluated from its final/ checkpoint, or its last step_*
# for runs saved before final/ existed, on the shared test split data/nqueens-<N>-1oml. GRAM gets
# num_samples samples (default 100). TRM evaluates deterministically (fixed initial state, argmax), so its
# runs use 1 sample: more would all be identical. Results go to results/nqueens_<N>/<run>.json, logs to
# logs/eval_nqueens_<N>_<run>.log, and the comparison table to results/nqueens_<N>/summary.md.
#
# Runs without final/ whose last checkpoint is short of the run's planned last step are still training (or
# stopped early) and are skipped; set ALLOW_UNFINISHED=1 to evaluate them anyway.
# Runs already evaluated are skipped; set FORCE=1 to redo them. Lower EFFECTIVE_BATCH_SIZE
# (queries x samples per forward pass, default 15360) if the GPU runs out of memory.
set -o pipefail

n=$1
samples=${2:-100}
case $n in
  8|10) ;;
  *) echo "Board size must be 8 or 10"; exit 1 ;;
esac

runs_dir=checkpoints/Nqueens-${n}-compare
results_dir=results/nqueens_${n}
mkdir -p logs "${results_dir}"
nproc=${NPROC:-$(nvidia-smi -L | wc -l)}

failed=()
for run_dir in "${runs_dir}"/*/; do
  run=$(basename "${run_dir}")

  if [ -f "${run_dir}final/checkpoint.pt" ]; then
    ckpt=${run_dir}final
  else
    ckpt=$(ls -d "${run_dir}"step_*/ 2>/dev/null | sort -V | tail -n 1)
    ckpt=${ckpt%/}
    if [ -z "${ckpt}" ]; then
      echo "== ${run}: no checkpoints, skipping"
      continue
    fi
    step=$(basename "${ckpt}")
    step=${step#step_}
    expected=$(python scripts/expected_final_step.py "${run_dir}all_config.yaml" 2>/dev/null)
    if [ -z "${expected}" ]; then
      echo "== ${run}: no final/ checkpoint and can't tell if it finished, using step_${step}"
    elif [ "${step}" -lt "${expected}" ] && [ -z "${ALLOW_UNFINISHED}" ]; then
      echo "== ${run}: at step_${step} of ${expected}, still training or stopped early, skipping (ALLOW_UNFINISHED=1 to evaluate)"
      continue
    else
      echo "== ${run}: no final/ checkpoint, using step_${step} (planned last step ${expected})"
    fi
  fi

  result=${results_dir}/${run}.json
  if [ -f "${result}" ] && [ -z "${FORCE}" ]; then
    echo "== ${run}: already evaluated, skipping (FORCE=1 to redo)"
    continue
  fi

  k=${samples}
  [[ ${run} == trm* ]] && k=1

  # The evaluator writes nqueens_results.json into the checkpoint directory; drop any stale copy first
  rm -f "${ckpt}/nqueens_results.json"
  echo "== $(date '+%F %T') Evaluating ${run} from ${ckpt} with ${k} sample(s)"
  if torchrun --nproc-per-node ${nproc} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 eval_pass_at_k.py \
       --checkpoint_path "${ckpt}" \
       --num_samples ${k} \
       --data_paths "['data/nqueens-${n}-1oml']" \
       --max_h_cycles 16 \
       --effective_batch_size ${EFFECTIVE_BATCH_SIZE:-15360} \
       2>&1 | tee "logs/eval_nqueens_${n}_${run}.log" \
     && [ -f "${ckpt}/nqueens_results.json" ]; then
    python -c "import json, sys; json.dump({'run': sys.argv[1], 'checkpoint': sys.argv[2], 'num_samples': int(sys.argv[3]), 'metrics': json.load(open(sys.argv[4]))}, open(sys.argv[5], 'w'), indent=2)" \
      "${run}" "${ckpt}" ${k} "${ckpt}/nqueens_results.json" "${result}"
  else
    echo "== ${run} evaluation FAILED, see logs/eval_nqueens_${n}_${run}.log"
    failed+=("${run}")
  fi
done

echo
python scripts/summarize_nqueens.py "${results_dir}"

if [ ${#failed[@]} -gt 0 ]; then
  echo "== Failed: ${failed[*]}"
  exit 1
fi
