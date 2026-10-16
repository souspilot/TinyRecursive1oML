#!/bin/bash
# Train N-Queens comparison runs (scripts/nqueens_compare.sh) one after another until none are left.
# Start one worker per GPU job: workers share the queue through the claims nqueens_compare.sh makes, so
# each run is trained by exactly one job. SelectR is taken only once MinLoss has finished, by whichever
# worker is free first, and starts from MinLoss's last checkpoint.
#
# Usage: bash scripts/nqueens_worker.sh <8|10>
#
# Runs whose checkpoint directory exists without a claim were started before claims existed (or crashed
# then) and are skipped. Each run logs to logs/nqueens_<N>_<method>.<job id>.log.
set -o pipefail

n=$1
case $n in
  8|10) ;;
  *) echo "Board size must be 8 or 10"; exit 1 ;;
esac

for data in data/nqueens-${n} data/nqueens-${n}-1oml data/nqueens-${n}-unique; do
  if [ ! -f "${data}/test/dataset.json" ]; then
    echo "Missing ${data}: build the datasets first (see scripts/nqueens_compare.sh)"
    exit 1
  fi
done

seed=${SEED:-42}
runs_dir=checkpoints/Nqueens-${n}-compare
claims=${runs_dir}/.claims
job=${PBS_JOBID:-$(hostname).$$}
mkdir -p logs

# Longest runs first, and MinLoss early so SelectR can start sooner
methods="trm gram trm_minloss trm_iexplr trm_random trm_unique trm_selectr"
failed=()

while true; do
  started=0
  for method in ${methods}; do
    run=${method}_seed${seed}
    [ -d "${claims}/${run}" ] && continue
    [[ " ${failed[*]} " == *" ${method} "* ]] && continue
    if [ -d "${runs_dir}/${run}" ]; then
      echo "== [${job}] ${method}: ${runs_dir}/${run} exists without a claim (started outside the workers, or crashed), skipping"
      continue
    fi

    if [ "${method}" = trm_selectr ]; then
      [ -f "${claims}/trm_minloss_seed${seed}/finished" ] || continue
      minloss_dir=${runs_dir}/trm_minloss_seed${seed}
      if [ -f "${minloss_dir}/final/checkpoint.pt" ]; then
        export INIT_CKPT=${minloss_dir}/final/checkpoint.pt
      else
        export INIT_CKPT=$(ls -d ${minloss_dir}/step_*/checkpoint.pt 2>/dev/null | sort -V | tail -n 1)
      fi
    fi

    log=logs/nqueens_${n}_${method}.${job}.log
    echo "== $(date '+%F %T') [${job}] Starting ${method} (log: ${log})"
    bash scripts/nqueens_compare.sh ${n} ${method} 2>&1 | tee "${log}"
    status=$?
    case ${status} in
      0) echo "== $(date '+%F %T') [${job}] Finished ${method}" ;;
      3) echo "== [${job}] ${method} was claimed by another job first" ;;
      *) echo "== $(date '+%F %T') [${job}] ${method} FAILED, see ${log}"; failed+=("${method}") ;;
    esac
    started=1
    # Rescan from the top, so a run that just became available (SelectR) is taken in order
    break
  done
  [ ${started} = 0 ] && break
done

waiting=""
if [ -d "${claims}/trm_minloss_seed${seed}" ] && [ ! -f "${claims}/trm_minloss_seed${seed}/finished" ] \
   && [ ! -d "${claims}/trm_selectr_seed${seed}" ]; then
  waiting=" (SelectR waits for MinLoss, which another job is training; that job takes SelectR next)"
fi
echo "== [${job}] Nothing left to start${waiting}"
if [ ${#failed[@]} -gt 0 ]; then
  echo "== [${job}] Failed: ${failed[*]}"
  exit 1
fi
