#!/bin/bash
# Compare GRAM, vanilla TRM, and TRM trained with one-of-many (1oML, https://arxiv.org/abs/2008.11990)
# objectives on N-Queens. Every method is evaluated on the same test split: one row per unique input,
# scored by the N-Queens verifier, with accuracy split into one-solution (_os) and multi-solution (_ms) queries.
#
# Usage: bash scripts/nqueens_compare.sh <8|10> <method>
#   gram         GRAM on every (input, solution) pair
#   trm          vanilla TRM on every (input, solution) pair  (1oML "Naive")
#   trm_random   TRM on one fixed random solution per input   (1oML "Random")
#   trm_unique   TRM on single-solution inputs only           (1oML "Unique")
#   trm_minloss  TRM trained towards the closest solution     (1oML "MinLoss")
#   trm_iexplr   TRM sampling solutions by model likelihood   (1oML "I-ExplR")
#   trm_selectr  TRM with a learned target selector           (1oML "SelectR"); starts from the trm_minloss run's
#                final checkpoint, or the checkpoint in INIT_CKPT if set
#
# Datasets (build once):
#   python dataset/build_nqueens_dataset.py --n 8 --remove-counts 5 6 7 --output-dir data/nqueens-8
#   python dataset/build_nqueens_dataset.py --n 8 --remove-counts 5 6 7 --output-dir data/nqueens-8-1oml --one-of-many
#   python dataset/build_nqueens_dataset.py --n 8 --remove-counts 5 6 7 --output-dir data/nqueens-8-unique --one-of-many --train-unique-only
# (for N=10 use --n 10 --remove-counts 7 8 9)
set -e

n=$1
method=$2

case $n in
  8)  epochs=${EPOCHS:-5000} eval_interval=${EVAL_INTERVAL:-50} kl_coef=0.07 ;;
  10) epochs=${EPOCHS:-1000} eval_interval=${EVAL_INTERVAL:-5} kl_coef=0.045 ;;
  *)  echo "Board size must be 8 or 10"; exit 1 ;;
esac

trm_args="arch=trm arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4"
case $method in
  gram)        data=data/nqueens-${n}        arch_args="arch=gram arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 arch.kl_balance_alpha=0.8 arch.kl_coef=${kl_coef}" ;;
  trm)         data=data/nqueens-${n}        arch_args="${trm_args}" ;;
  trm_random)  data=data/nqueens-${n}-1oml   arch_args="${trm_args}" ;;
  trm_unique)  data=data/nqueens-${n}-unique arch_args="${trm_args}" ;;
  trm_minloss) data=data/nqueens-${n}-1oml   arch_args="${trm_args} arch.loss.one_of_many=minloss" ;;
  trm_iexplr)  data=data/nqueens-${n}-1oml   arch_args="${trm_args} arch.loss.one_of_many=iexplr" ;;
  trm_selectr)
    minloss_dir=checkpoints/Nqueens-${n}-compare/trm_minloss_seed${SEED:-42}
    if [ -z "${INIT_CKPT}" ]; then
      INIT_CKPT=${minloss_dir}/final/checkpoint.pt
      if [ ! -f "${INIT_CKPT}" ]; then
        # Runs from before final/ was saved: use their last step
        INIT_CKPT=$(ls -d ${minloss_dir}/step_*/checkpoint.pt 2>/dev/null | sort -V | tail -n 1)
      fi
    fi
    if [ ! -f "${INIT_CKPT}" ]; then
      echo "SelectR starts from a MinLoss model, but no checkpoint was found in ${minloss_dir}. Run trm_minloss first, or set INIT_CKPT."
      exit 1
    fi
    echo "SelectR starting from ${INIT_CKPT}"
    data=data/nqueens-${n}-1oml
    arch_args="${trm_args} arch.loss.one_of_many=selectr +load_checkpoint=${INIT_CKPT}" ;;
  *) echo "Unknown method: ${method}"; exit 1 ;;
esac

# Claim the run so no other job trains it at the same time. mkdir is atomic, so only one job can win.
# The claim is kept (marked finished) when training succeeds, and released if it fails or is interrupted.
run=${method}_seed${SEED:-42}
claim=checkpoints/Nqueens-${n}-compare/.claims/${run}
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

nproc=${NPROC:-$(nvidia-smi -L | wc -l)}
if ! torchrun --nproc-per-node ${nproc} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
${arch_args} \
seed=${SEED:-42} \
global_batch_size=768 \
data_paths="[${data}]" \
data_paths_test="[data/nqueens-${n}-1oml]" \
evaluators="[]" \
epochs=${epochs} eval_interval=${eval_interval} \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
+project_name=Nqueens-${n}-compare \
+run_name=${run} ema=True \
checkpoint_every_eval=True; then
  rm -rf "${claim}"
  exit 1
fi
touch "${claim}/finished"
