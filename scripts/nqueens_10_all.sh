#!/bin/bash
# Build the 10-queens datasets and run every method of scripts/nqueens_compare.sh in sequence.
# Each run logs to logs/nqueens_10_<method>.log. A finished run leaves logs/nqueens_10_<method>.done,
# so rerunning this script after an interruption skips completed methods. A failed run is reported
# at the end without stopping the others.
#
# Usage: bash scripts/nqueens_10_all.sh
set -o pipefail

mkdir -p logs

build() {
  local out_dir=$1; shift
  if [ -f "${out_dir}/test/dataset.json" ]; then
    echo "== ${out_dir} exists, skipping build"
    return 0
  fi
  echo "== Building ${out_dir}"
  python dataset/build_nqueens_dataset.py --n 10 --remove-counts 7 8 9 --output-dir "${out_dir}" "$@" 2>&1 \
    | tee "logs/build_$(basename "${out_dir}").log"
}

build data/nqueens-10 || exit 1
build data/nqueens-10-1oml --one-of-many || exit 1
build data/nqueens-10-unique --one-of-many --train-unique-only || exit 1

failed=()
for method in gram trm trm_random trm_unique trm_minloss trm_iexplr trm_selectr; do
  log=logs/nqueens_10_${method}.log
  if [ -f "logs/nqueens_10_${method}.done" ]; then
    echo "== ${method} already done, skipping"
    continue
  fi

  if [ "${method}" = trm_selectr ]; then
    if [ ! -f logs/nqueens_10_trm_minloss.done ]; then
      echo "== Skipping trm_selectr: trm_minloss has not finished"
      failed+=("${method}")
      continue
    fi
    # Start from the last checkpoint of the MinLoss run
    export INIT_CKPT=$(ls -d checkpoints/Nqueens-10-compare/trm_minloss_seed${SEED:-42}/step_*/checkpoint.pt 2>/dev/null | sort -V | tail -n 1)
    if [ -z "${INIT_CKPT}" ]; then
      echo "== Skipping trm_selectr: no trm_minloss checkpoint found"
      failed+=("${method}")
      continue
    fi
  fi

  echo "== $(date '+%F %T') Starting ${method} (log: ${log})"
  if bash scripts/nqueens_compare.sh 10 "${method}" 2>&1 | tee "${log}"; then
    touch "logs/nqueens_10_${method}.done"
    echo "== $(date '+%F %T') Finished ${method}"
  else
    echo "== $(date '+%F %T') ${method} FAILED, see ${log}"
    failed+=("${method}")
  fi
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "== Failed: ${failed[*]}"
  exit 1
fi
echo "== All 10-queens runs finished"
