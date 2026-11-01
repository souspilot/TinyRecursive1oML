#!/bin/bash
# Diversity without a latent variable: sample deterministic TRMs with noise on the initial latent (--init_noise)
# and/or the 8 board symmetries (--symmetry), against GRAM's latent sampling, at 100 samples per input.
#
# Usage (one worker per GPU; workers share the task list through claims, so start as many as there are GPUs):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/diversity_sampling.sh
#
# Why: scripts/tie_breaking.py found that the recursion commits to one solution within its first supervision step,
# so noise after it (--latent_noise, PTRM-style) gives MinLoss no diversity (maze coverage 8.7, 1 path), while noise
# on the initial latent (4 seeds: 2.8 distinct paths, coverage 16) and symmetric views (8 views: 5.9 paths,
# coverage 30, GRAM 28.8) do. Resumable (shards are kept). When every task is done, the last worker analyzes both
# sample directories with the usual scripts (results/maze_15/summary.md, results/nqueens_10_diversity/summary.md).
set -o pipefail
claims=samples/.diversity_claims
mkdir -p logs ${claims}

ckpt() {  # ckpt <project> <run>: the final checkpoint, else the last step's
  local d=checkpoints/$1/$2
  if [ -f "$d/final/checkpoint.pt" ]; then echo "$d/final/checkpoint.pt"; else echo "$(ls -d "$d"/step_*/ | sort -V | tail -1)checkpoint.pt"; fi
}

sample() {  # sample <maze|nqueens> <run> <out name> [sample_predictions.py options]
  local task=$1 run=$2 name=$3; shift 3
  local project data samples
  if [ "${task}" = maze ]; then project=Maze-15-compare; data=data/maze-15-1oml; samples=samples/maze_15
  else project=Nqueens-10-compare; data=data/nqueens-10-1oml; samples=samples/nqueens_10_diversity; fi
  torchrun --nproc-per-node 1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 sample_predictions.py \
    --checkpoint "$(ckpt ${project} "${run}")" --data ${data} --out "${samples}/${name}" --num_samples 100 \
    --effective_batch_size ${EFFECTIVE_BATCH_SIZE:-7680} --shard_size 256 "$@"
}

tasks=(
  "maze trm_minloss_seed42 trm_minloss_seed42_init1 --init_noise 1"
  "maze trm_minloss_seed42 trm_minloss_seed42_sym_init1 --symmetry --init_noise 1"
  "maze trm_iexplr_seed42 trm_iexplr_seed42_init1 --init_noise 1"
  "maze trm_iexplr_seed42 trm_iexplr_seed42_sym_init1 --symmetry --init_noise 1"
  "maze gram_kl0.015_steps100000_seed42 gram_kl0.015_steps100000_seed42_sym --symmetry"
  "maze trm_seed42 trm_seed42_sym_init1 --symmetry --init_noise 1"
  "maze trm_minloss_seed42 trm_minloss_seed42_init3 --init_noise 3"
  "nqueens trm_minloss_seed42 trm_minloss_seed42_init1 --init_noise 1"
  "nqueens trm_minloss_seed42 trm_minloss_seed42_sym_init1 --symmetry --init_noise 1"
  "nqueens trm_iexplr_seed42 trm_iexplr_seed42_sym_init1 --symmetry --init_noise 1"
  "nqueens trm_unique_seed42 trm_unique_seed42_sym_init1 --symmetry --init_noise 1"
  "nqueens gram_seed42 gram_seed42 "
  "nqueens gram_seed42 gram_seed42_sym --symmetry"
)

failed=()
for i in "${!tasks[@]}"; do
  read -r -a t <<< "${tasks[$i]}"
  name="${t[0]}_${t[2]}"
  [ -e "${claims}/${name}.done" ] && continue
  mkdir "${claims}/${name}" 2>/dev/null || continue  # another worker has it
  trap 'rmdir "${claims}/${name}" 2>/dev/null; exit 130' INT TERM
  echo "== $(date '+%F %T') ${name} (GPU ${CUDA_VISIBLE_DEVICES:-unset})"
  if sample "${t[@]}" > "logs/diversity_${name}.log" 2>&1; then touch "${claims}/${name}.done"; else failed+=("${name}"); fi
  rmdir "${claims}/${name}"
  trap - INT TERM
done

if [ ${#failed[@]} -gt 0 ]; then echo "== Failed (see logs/diversity_<name>.log): ${failed[*]}"; exit 1; fi
done_count=$(ls ${claims}/*.done 2>/dev/null | wc -l)
if [ "${done_count}" -eq "${#tasks[@]}" ] && mkdir "${claims}/analysis" 2>/dev/null; then
  echo "== $(date '+%F %T') Analyzing"
  python scripts/analyze_maze_samples.py --data data/maze-15-1oml --samples samples/maze_15 --out results/maze_15
  python scripts/analyze_nqueens_samples.py --data data/nqueens-10-1oml --samples samples/nqueens_10_diversity \
    --out results/nqueens_10_diversity
else
  echo "== No tasks left for this worker (${done_count}/${#tasks[@]} done)"
fi
