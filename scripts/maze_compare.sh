#!/bin/bash
# Compare GRAM and TRM with one-of-many (1oML) objectives on 15x15 mazes with several shortest paths
# (dataset/build_maze_dataset.py). Every run trains for about STEPS steps (default 50000) in 20 evaluation rounds,
# so methods with smaller training sets get more epochs, and all are evaluated on the same test split.
# A prediction counts as correct when it marks any shortest path (evaluators/maze_utils.py).
#
# Usage: bash scripts/maze_compare.sh <method>
#   gram          GRAM on one random stored path per maze each epoch (up to 16 stored per maze)
#   gram_single   GRAM on one fixed path per maze
#   gram_budget   GRAM with a per-maze KL budget: KL up to KL_BUDGET * ln(#shortest paths) nats is free (default 1,
#                 enough to pick one path), KL above it is weighted by KL_COEF (default 0.05 here)
#   trm           TRM on one random stored path per maze each epoch   (1oML "Naive")
#   trm_random    TRM on one fixed random path per maze               (1oML "Random")
#   trm_unique    TRM on mazes with a unique shortest path only       (1oML "Unique")
#   trm_minloss   TRM trained towards the closest stored path         (1oML "MinLoss")
#   trm_iexplr    TRM sampling stored paths by model likelihood       (1oML "I-ExplR")
#   trm_setmatch  noisy TRM: ROLLOUTS rollouts per maze (default 4), each with its own random vector
#                 (NOISE_DIM, default 32), matched one to one to the maze's stored paths
#   trm_noise_minloss  same noisy rollouts, each trained towards its closest stored path (ablation: no matching)
#   When ROLLOUTS or NOISE_DIM is set, run names include it (e.g. trm_setmatch_k8_nd64_seed42).
#
# RESUME=1 continues an interrupted run from its latest checkpoint. SEED sets the seed (default 42).
# KL_COEF sets GRAM's KL weight (default 0.07, untuned for mazes); when set, GRAM run names include it
# (e.g. gram_kl0.01_seed42), so a KL sweep doesn't collide with the default runs. Likewise KL_BUDGET
# (e.g. gram_budget_b0.5_seed42). STEPS sets the training length (default ~50000); when set, run names include it
# (e.g. gram_kl0.015_steps100000_seed42).
# WANDB_OFFLINE=1 logs to W&B offline (for nodes without internet); upload later with `wandb sync`.
#
# Datasets (build once):
#   python dataset/build_maze_dataset.py --size 15 --num-train 100000 --num-test 2000 --output-dir data/maze-15
set -e

method=$1
steps=${STEPS:-50000}
root=data/maze-15
gram_args="arch=gram arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 arch.kl_balance_alpha=0.8 arch.kl_coef=${KL_COEF:-0.07}"
trm_args="arch=trm arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4"
case $method in
  gram)        data=${root}        arch_args="${gram_args}" ;;
  gram_single) data=${root}-single arch_args="${gram_args}" ;;
  gram_budget)
    data=${root}
    arch_args="${gram_args/arch.kl_coef=${KL_COEF:-0.07}/arch.kl_coef=${KL_COEF:-0.05}} arch.kl_budget=${KL_BUDGET:-1.0}" ;;
  trm)         data=${root}        arch_args="${trm_args}" ;;
  trm_random)  data=${root}-single arch_args="${trm_args}" ;;
  trm_unique)  data=${root}-unique arch_args="${trm_args}" ;;
  trm_minloss) data=${root}-1oml   arch_args="${trm_args} arch.loss.one_of_many=minloss" ;;
  trm_iexplr)  data=${root}-1oml   arch_args="${trm_args} arch.loss.one_of_many=iexplr" ;;
  trm_setmatch|trm_noise_minloss)
    rollouts=${ROLLOUTS:-4}
    objective=setmatch
    [ "${method}" = trm_noise_minloss ] && objective=minloss
    data=${root}-1oml
    arch_args="${trm_args} arch.noise_dim=${NOISE_DIM:-32} arch.loss.one_of_many=${objective} +rollouts=${rollouts}" ;;
  *) echo "Unknown method: ${method}"; exit 1 ;;
esac

# 20 evaluation rounds of eval_interval epochs, about ${steps} steps of 768 sequences in total. With several
# rollouts per maze, each step covers 768 / rollouts mazes, so the same steps take more epochs.
rollouts=${rollouts:-1}
mazes_per_step=$(( 768 / rollouts ))
groups=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['total_groups'])" "${data}/train/dataset.json")
eval_interval=$(( (steps * mazes_per_step / 20 + groups / 2) / groups ))
[ ${eval_interval} -lt 1 ] && eval_interval=1
epochs=$(( eval_interval * 20 ))
if [ ${rollouts} -gt 1 ]; then
  # Epochs are few here, so rounding to exactly 20 rounds would miss the step budget by several percent:
  # take the epoch count closest to it, evaluated about every 1/20 of it (largest divisor up to that)
  epochs=$(( (steps * mazes_per_step + groups / 2) / groups ))
  eval_interval=$(( epochs / 20 ))
  [ ${eval_interval} -lt 1 ] && eval_interval=1
  while [ $(( epochs % eval_interval )) -ne 0 ]; do eval_interval=$(( eval_interval - 1 )); done
fi

run=${method}_seed${SEED:-42}
if [ -n "${KL_COEF}" ] && [[ ${method} == gram* ]]; then
  run=${method}_kl${KL_COEF}_seed${SEED:-42}
fi
if [ -n "${KL_BUDGET}" ] && [ "${method}" = gram_budget ]; then
  run=${run%_seed*}_b${KL_BUDGET}_seed${SEED:-42}
fi
if [ ${rollouts} -gt 1 ] && { [ -n "${ROLLOUTS}" ] || [ -n "${NOISE_DIM}" ]; }; then
  run=${method}${ROLLOUTS:+_k${ROLLOUTS}}${NOISE_DIM:+_nd${NOISE_DIM}}_seed${SEED:-42}
fi
if [ -n "${STEPS}" ]; then
  run=${run%_seed*}_steps${STEPS}_seed${SEED:-42}
fi
runs_dir=checkpoints/Maze-15-compare

[ -n "${WANDB_OFFLINE}" ] && arch_args="${arch_args} wandb_offline=True"

if [ -n "${RESUME}" ]; then
  if [ -f "${runs_dir}/${run}/final/checkpoint.pt" ]; then
    echo "${runs_dir}/${run} has already finished (final/ exists), nothing to resume"
    exit 1
  fi
  resume_ckpt=$(ls -d ${runs_dir}/${run}/step_*/checkpoint.pt 2>/dev/null | sort -V | tail -n 1)
  if [ -z "${resume_ckpt}" ]; then
    echo "No checkpoint to resume from in ${runs_dir}/${run}"
    exit 1
  fi
  echo "Resuming from ${resume_ckpt}"
  arch_args="${arch_args} +load_checkpoint=${resume_ckpt} +resume=True"
fi

# Claim the run so no other job trains it at the same time. mkdir is atomic, so only one job can win.
# The claim is kept (marked finished) when training succeeds, and released if it fails or is interrupted.
claim=${runs_dir}/.claims/${run}
mkdir -p "$(dirname "${claim}")"
if ! mkdir "${claim}" 2>/dev/null; then
  if [ -f "${claim}/finished" ]; then
    echo "${run} has already finished. To train it again, remove ${claim}"
  else
    owner=$(cat "${claim}/owner" 2>/dev/null)
    echo "${run} is claimed by another job (${owner:-claimed just now}). If that job died, remove ${claim}"
  fi
  exit 3
fi
echo "$(hostname) job=${PBS_JOBID:-none} started $(date '+%F %T')" > "${claim}/owner"
trap 'rm -rf "${claim}"; exit 130' INT TERM

echo "${run}: ${data}, ${epochs} epochs (evaluation every ${eval_interval}) for ~${steps} steps"
nproc=${NPROC:-$(nvidia-smi -L | wc -l)}
if ! torchrun --nproc-per-node ${nproc} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
${arch_args} \
seed=${SEED:-42} \
global_batch_size=768 \
data_paths="[${data}]" \
data_paths_test="[${root}-1oml]" \
evaluators="[]" \
epochs=${epochs} eval_interval=${eval_interval} \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
+project_name=Maze-15-compare \
+run_name=${run} ema=True \
checkpoint_every_eval=True; then
  rm -rf "${claim}"
  exit 1
fi
touch "${claim}/finished"
