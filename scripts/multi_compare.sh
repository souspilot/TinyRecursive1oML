#!/bin/bash
# Train one method on one dataset (built by scripts/build_multi_datasets.sh). All methods share the backbone below,
# train for about STEPS steps, and are evaluated on the same test split.
#
# Usage: KL_COEF=0.1 bash scripts/multi_compare.sh <maze|nqueens|sudoku|graph> gram
#        bash scripts/multi_compare.sh <maze|nqueens|sudoku|graph> gram_std_bestofk
#
#   gram              GRAM: posterior latents, one random stored solution per input each epoch. KL_COEF (beta) is required
#   gram_std_bestofk  Same backbone with N(0, I) latents, so the label never reaches the latent (no posterior, no KL,
#                     no beta): ROLLOUTS samples per input (default 8), and only the one closest to a stored solution
#                     trains. Same 768 sequences per step, so 768 / ROLLOUTS inputs per step
#
# STEPS (default 150000), SEED (default 42), RESUME=1 (continue an interrupted run from its latest checkpoint),
# EXTRA_ARGS (appended to the pretrain.py arguments). Checkpoints go to checkpoints/<project>/<run>.
set -e
dataset=$1
method=$2
steps=${STEPS:-150000}
seed=${SEED:-42}
case $dataset in
  maze)    root=data/maze-multi project=Maze-multi ;;
  nqueens) root=data/nqueens-multi project=Nqueens-multi ;;
  sudoku)  root=data/sudoku-multi project=Sudoku-multi ;;
  graph)   root=data/graph-coloring-multi project=Graph-coloring-multi ;;
  *) echo "Unknown dataset: ${dataset} (maze, nqueens, sudoku, graph)"; exit 1 ;;
esac

backbone="arch=gram arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4"
case $method in
  gram)
    [ -n "${KL_COEF}" ] || { echo "gram needs KL_COEF"; exit 1; }
    data=${root}
    rollouts=1
    arch_args="${backbone} arch.kl_balance_alpha=0.8 arch.kl_coef=${KL_COEF}"
    run=gram_kl${KL_COEF}_steps${steps}_seed${seed} ;;
  gram_std_bestofk)
    data=${root}-1oml
    rollouts=${ROLLOUTS:-8}
    arch_args="${backbone} arch.train_latent=prior arch.prior_noise=standard arch.loss.one_of_many=bestofk +rollouts=${rollouts}"
    run=${method}${ROLLOUTS:+_k${ROLLOUTS}}_steps${steps}_seed${seed} ;;
  *) echo "Unknown method: ${method} (gram, gram_std_bestofk)"; exit 1 ;;
esac
[ -z "${EXTRA_ARGS}" ] || arch_args="${arch_args} ${EXTRA_ARGS}"

if [ ! -f "${data}/train/dataset.json" ] || [ ! -f "${root}-1oml/test/dataset.json" ]; then
  echo "${data} is missing: build it with bash scripts/build_multi_datasets.sh ${dataset}"
  exit 1
fi

# The epoch count closest to the step budget, evaluated about every 1/20 of it (largest divisor up to that)
inputs_per_step=$(( 768 / rollouts ))
groups=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['total_groups'])" "${data}/train/dataset.json")
epochs=$(( (steps * inputs_per_step + groups / 2) / groups ))
eval_interval=$(( epochs / 20 ))
[ ${eval_interval} -lt 1 ] && eval_interval=1
while [ $(( epochs % eval_interval )) -ne 0 ]; do eval_interval=$(( eval_interval - 1 )); done

run_dir=checkpoints/${project}/${run}
if [ -n "${RESUME}" ]; then
  [ ! -f "${run_dir}/final/checkpoint.pt" ] || { echo "${run_dir} has already finished"; exit 1; }
  resume_ckpt=$(ls -d ${run_dir}/step_*/checkpoint.pt 2>/dev/null | sort -V | tail -n 1)
  [ -n "${resume_ckpt}" ] || { echo "No checkpoint to resume from in ${run_dir}"; exit 1; }
  echo "Resuming from ${resume_ckpt}"
  arch_args="${arch_args} +load_checkpoint=${resume_ckpt} +resume=True"
elif [ -e "${run_dir}" ]; then
  echo "${run_dir} already exists. Use RESUME=1, or remove it to train again"
  exit 1
fi

echo "${run}: ${data}, ${epochs} epochs (evaluation every ${eval_interval}) for ~${steps} steps"
# One process per visible GPU (nvidia-smi lists every GPU of the job, ignoring CUDA_VISIBLE_DEVICES)
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then nproc=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .); else nproc=$(nvidia-smi -L | wc -l); fi
exec torchrun --nproc-per-node ${nproc} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
${arch_args} \
seed=${seed} \
global_batch_size=768 \
data_paths="[${data}]" \
data_paths_test="[${root}-1oml]" \
evaluators="[]" \
epochs=${epochs} eval_interval=${eval_interval} \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
+project_name=${project} \
+run_name=${run} ema=True \
checkpoint_every_eval=True
