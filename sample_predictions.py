"""Sample predictions from a checkpoint on a test split and save them in resumable shards.

Each shard holds every sample for a contiguous range of test queries and is written atomically. Shards are
divided between processes (torchrun ranks), and a restarted job skips shards that already exist, so an
interrupted run only loses the shards in progress. Sampling is seeded by the queries in each batch, so the
result doesn't depend on how many processes or jobs produced it.

Usage:
    torchrun --nproc-per-node 2 sample_predictions.py --checkpoint checkpoints/<project>/<run>/final/checkpoint.pt \\
        --data data/nqueens-10-1oml --num_samples 100 --out samples/nqueens_10/<run>

Output: <out>/meta.json and <out>/shard_<i>.npz with `query_start` and `preds` (queries x samples x seq_len, uint8).
"""
import argparse
import json
import math
import os

import numpy as np
import torch

# Checkpoints pickle these classes from pretrain.py run as __main__; importing them here lets torch.load find them
from pretrain import PretrainConfig, ArchConfig, LossConfig, EvaluatorConfig  # noqa: F401
from dataset.common import PuzzleDatasetMetadata
from models.ema import EMAHelper
from models.losses import IGNORE_LABEL_ID
from utils.functions import load_model_class


def load_split(data_path: str, split: str = "test"):
    split_dir = os.path.join(data_path, split)
    with open(os.path.join(split_dir, "dataset.json")) as f:
        metadata = PuzzleDatasetMetadata(**json.load(f))
    data = {}
    for field in ("inputs", "labels", "puzzle_identifiers", "puzzle_indices", "label_set", "num_solutions"):
        path = os.path.join(split_dir, f"all__{field}.npy")
        if os.path.exists(path):
            data[field] = np.load(path, mmap_mode="r")
    return data, metadata


def make_batch(data, metadata: PuzzleDatasetMetadata, start: int, end: int):
    """The batch PuzzleDataset would collate for test examples [start, end)."""
    puzzle_ids = np.searchsorted(data["puzzle_indices"], np.arange(start, end), side="right") - 1
    batch = {
        "inputs": data["inputs"][start:end],
        "labels": data["labels"][start:end],
        "puzzle_identifiers": data["puzzle_identifiers"][puzzle_ids],
    }
    for field in ("label_set", "num_solutions"):
        if field in data:
            batch[field] = data[field][start:end]
    batch = {k: torch.from_numpy(np.array(v, dtype=np.int32)) for k, v in batch.items()}
    if metadata.ignore_label_id is not None:
        for field in ("labels", "label_set"):
            if field in batch:
                batch[field][batch[field] == metadata.ignore_label_id] = IGNORE_LABEL_ID
    return batch


def load_model(checkpoint_path: str, metadata: PuzzleDatasetMetadata, temperature: float, device: str):
    """Build the model with its loss head and load its EMA weights, as eval_pass_at_k.py does."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
        temperature=temperature,
    )
    with torch.device(device):
        model = load_model_class(config.arch.name)(model_cfg)
        model = load_model_class(config.arch.loss.name)(model, **config.arch.loss.__pydantic_extra__)

    strip = lambda state: {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(strip(checkpoint["model"]), assign=True)
    if "ema" in checkpoint:
        ema = EMAHelper()
        ema.register(model)
        ema.load_state_dict(strip(checkpoint["ema"]))
        ema.ema(model)

    if temperature > 0 and not hasattr(model.model.config, "temperature"):
        print(f"Warning: {config.arch.name} has no temperature setting, so --temperature is ignored")
    return model.to(device).eval(), checkpoint.get("step")


@torch.inference_mode()
def sample(model, batch, num_samples: int, device: str) -> torch.Tensor:
    """Run the model to completion on num_samples copies of each query. Returns (queries, samples, seq_len)."""
    expanded = {k: v.to(device).repeat_interleave(num_samples, dim=0) for k, v in batch.items()}
    with torch.device(device):
        carry = model.initial_carry(expanded)
    while True:
        carry, _, _, outputs, all_finish = model(carry=carry, batch=expanded, return_keys={"preds"})
        if all_finish:
            break
    return outputs["preds"].view(batch["inputs"].shape[0], num_samples, -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Dataset directory; its test split is sampled")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0, help="Per-cell sampling temperature (TRM); 0 = argmax")
    parser.add_argument("--shard_size", type=int, default=512, help="Test queries per shard")
    parser.add_argument("--effective_batch_size", type=int, default=15360, help="Queries x samples per forward pass")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    data, metadata = load_split(args.data)
    num_queries = len(data["inputs"])
    num_shards = math.ceil(num_queries / args.shard_size)

    # The settings that determine the samples; a rerun with different ones must not mix with existing shards
    meta = dict(checkpoint=args.checkpoint, data=args.data, num_samples=args.num_samples, temperature=args.temperature,
                shard_size=args.shard_size, seed=args.seed, num_queries=num_queries, num_shards=num_shards)
    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            existing = json.load(f)
        mismatched = {k: (existing.get(k), v) for k, v in meta.items() if existing.get(k) != v}
        if mismatched:
            raise ValueError(f"{args.out} holds samples made with other settings (existing, requested): {mismatched}. "
                             f"Use another --out or delete it.")
    else:
        with open(f"{meta_path}.tmp{rank}", "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(f"{meta_path}.tmp{rank}", meta_path)

    shard_path = lambda i: os.path.join(args.out, f"shard_{i:05d}.npz")
    todo = [i for i in range(rank, num_shards, world_size) if not os.path.exists(shard_path(i))]
    print(f"[rank {rank}/{world_size}] {args.out}: {len(todo)} of {len(range(rank, num_shards, world_size))} shards to sample")
    if not todo:
        return

    model, step = load_model(args.checkpoint, metadata, args.temperature, device)
    print(f"[rank {rank}] Loaded {args.checkpoint} (step {step}) on {device}")
    queries_per_pass = max(1, args.effective_batch_size // args.num_samples)

    for done, shard in enumerate(todo, 1):
        start, end = shard * args.shard_size, min((shard + 1) * args.shard_size, num_queries)
        preds = np.empty((end - start, args.num_samples, metadata.seq_len), dtype=np.uint8)
        for batch_start in range(start, end, queries_per_pass):
            batch_end = min(batch_start + queries_per_pass, end)
            torch.manual_seed(args.seed * 1_000_003 + batch_start)
            batch = make_batch(data, metadata, batch_start, batch_end)
            preds[batch_start - start:batch_end - start] = sample(model, batch, args.num_samples, device).cpu().numpy()

        tmp_path = f"{shard_path(shard)}.tmp{rank}"
        with open(tmp_path, "wb") as f:
            np.savez(f, query_start=start, preds=preds)
        os.replace(tmp_path, shard_path(shard))
        print(f"[rank {rank}] shard {shard} (queries {start}-{end - 1}) saved, {done}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
