#!/bin/bash
# Train maze comparison runs (scripts/maze_compare.sh) one after another until none are left.
# Start one worker per GPU job: workers share the queue through the claims maze_compare.sh makes, so each run
# is trained by exactly one job.
#
# Usage: bash scripts/maze_worker.sh
#
# Runs whose checkpoint directory exists without a claim (crashed, or killed at walltime) are skipped; continue
# them with RESUME=1 bash scripts/maze_compare.sh <method>. Each run logs to logs/maze_15_<method>.<job id>.log.
set -o pipefail

for data in data/maze-15 data/maze-15-1oml data/maze-15-single data/maze-15-unique; do
  if [ ! -f "${data}/test/dataset.json" ]; then
    echo "Missing ${data}: build the datasets first (see scripts/maze_compare.sh)"
    exit 1
  fi
done

seed=${SEED:-42}
runs_dir=checkpoints/Maze-15-compare
claims=${runs_dir}/.claims
job=${PBS_JOBID:-$(hostname).$$}
mkdir -p logs

# GRAM steps are slower, so those runs go first
methods=${METHODS:-"gram gram_single trm trm_minloss trm_iexplr trm_random trm_unique"}
failed=()

while true; do
  started=0
  for method in ${methods}; do
    run=${method}_seed${seed}
    [ -d "${claims}/${run}" ] && continue
    [[ " ${failed[*]} " == *" ${method} "* ]] && continue
    if [ -d "${runs_dir}/${run}" ]; then
      echo "== [${job}] ${method}: ${runs_dir}/${run} exists without a claim (crashed or interrupted), skipping"
      continue
    fi

    log=logs/maze_15_${method}.${job}.log
    echo "== $(date '+%F %T') [${job}] Starting ${method} (log: ${log})"
    bash scripts/maze_compare.sh ${method} 2>&1 | tee "${log}"
    status=$?
    case ${status} in
      0) echo "== $(date '+%F %T') [${job}] Finished ${method}" ;;
      3) echo "== [${job}] ${method} was claimed by another job first" ;;
      *) echo "== $(date '+%F %T') [${job}] ${method} FAILED, see ${log}"; failed+=("${method}") ;;
    esac
    started=1
    break
  done
  [ ${started} = 0 ] && break
done

echo "== [${job}] Nothing left to start"
if [ ${#failed[@]} -gt 0 ]; then
  echo "== [${job}] Failed: ${failed[*]}"
  exit 1
fi
