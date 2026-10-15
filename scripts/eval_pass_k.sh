#!/bin/bash
CKPT_DIR=$1
NUM_SAMPLES=${2:-100}
nproc=${NPROC:-$(nvidia-smi -L | wc -l)}
torchrun --nproc-per-node ${nproc} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 eval_pass_at_k.py \
--checkpoint_path "$CKPT_DIR" \
--num_samples $NUM_SAMPLES \
--data_paths "['data/nqueens-8']" \
--max_h_cycles 16 \
--visualize
