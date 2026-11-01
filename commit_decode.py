"""Decode N-Queens one decision at a time: commit queens as extra givens and re-run the model.

The test inputs are partial boards: some queens are given, and every other row needs one. Each pass, the model's
per-cell probability of a queen is read row by row (normalized over the row's columns). Every row whose most
likely column is near-certain (normalized probability >= --certain) gets its queen; then one undecided row gets
a queen sampled from its column probabilities (--choice sample) or at its most likely column (--choice greedy).
The committed queens become givens and the model runs again, until every row has a queen. With exact conditional
probabilities, this samples the input's solutions uniformly (self-reducibility). The decoder uses only the
model's outputs: no attack checks, which would be a solver.

The undecided row is the most confident one (--order confident: decide the easiest choice first, as in
adaptive masked-diffusion decoding) or a random one (--order random).

Output, in the format of sample_predictions.py (scripts/analyze_nqueens_samples.py reads it): <out>/meta.json and
<out>/shard_<i>.npz with `query_start`, `preds` (queries x samples x n*n, uint8) and `passes` (model passes per
sample). Resumable and split across torchrun ranks by shard.

Usage:
    torchrun --nproc-per-node 1 commit_decode.py --checkpoint checkpoints/Nqueens-10-compare/trm_seed42/final/checkpoint.pt \\
        --data data/nqueens-10-1oml --out samples/nqueens_10/trm_seed42_commit --num_samples 100
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from evaluators.nqueens_utils import find_all_solutions
from sample_predictions import load_split, load_model
from walk_decode import model_predictor

QUEEN = 2


def oracle_predictor(device: str):
    """Exact share of the solutions extending each board that put a queen on each cell (for testing)."""
    def predict(inputs: torch.Tensor) -> torch.Tensor:
        n = int(round(inputs.shape[1] ** 0.5))
        out = []
        for row in inputs.cpu().numpy():
            solutions = find_all_solutions(row.reshape(n, n), n, max_solutions=10**6)
            out.append(np.mean([s.ravel() == QUEEN for s in solutions], 0) if solutions else np.zeros(n * n))
        return torch.tensor(np.stack(out), dtype=torch.float32, device=device)
    return predict


def commit(predict, boards: torch.Tensor, num_samples: int, choice: str, order: str, certain: float,
           generator: torch.Generator):
    """Fill every queen-less row of each board. boards: (Q, n, n) tokens. Returns preds (Q, S, n*n), passes (Q, S)."""
    q, n, _ = boards.shape
    device = boards.device
    b = boards.repeat_interleave(num_samples, 0).clone()
    w = len(b)
    passes = torch.zeros(w, dtype=torch.int32, device=device)
    for _ in range(n):
        empty = ~(b == QUEEN).any(2)  # (W, n) rows still without a queen
        active = empty.any(1)
        if not active.any():
            break
        idx = active.nonzero().squeeze(1)
        probs = predict(b[idx].flatten(1)).view(-1, n, n)
        passes[idx] += 1
        empty = empty[idx]
        total = probs.sum(2, keepdim=True)
        # A row the model gives no queen mass: every column equally likely
        weights = torch.where(total > 1e-12, probs / total.clamp_min(1e-12), torch.full_like(probs, 1.0 / n))
        top, best = weights.max(2)
        sure = empty & (top >= certain)

        # One undecided row per pass gets a real choice
        undecided = empty & ~sure
        if order == "confident":
            score = torch.where(undecided, top, torch.full_like(top, -1.0))
        else:
            score = torch.where(undecided, torch.rand(top.shape, generator=generator, device=device), torch.full_like(top, -1.0))
        row = score.argmax(1)
        has_choice = undecided.any(1)
        row_weights = weights[torch.arange(len(idx), device=device), row]
        if choice == "sample":
            col = torch.multinomial(row_weights, 1, generator=generator).squeeze(1)
        else:
            col = row_weights.argmax(1)

        boards_idx = b[idx]
        sure_rows, sure_cols = sure.nonzero(as_tuple=True)
        boards_idx[sure_rows, sure_cols, best[sure_rows, sure_cols]] = QUEEN
        chosen = has_choice.nonzero().squeeze(1)
        boards_idx[chosen, row[chosen], col[chosen]] = QUEEN
        b[idx] = boards_idx
    return b.view(q, num_samples, n * n), passes.view(q, num_samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="Model checkpoint (omit with --oracle)")
    parser.add_argument("--oracle", action="store_true", help="Use exact solution shares instead of a model (tests the decoder)")
    parser.add_argument("--data", required=True, help="N-Queens dataset directory; its test split is decoded")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--choice", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--order", choices=("confident", "random"), default="confident")
    parser.add_argument("--certain", type=float, default=0.99, help="Normalized probability treated as a forced queen")
    parser.add_argument("--shard_size", type=int, default=512, help="Test queries per shard")
    parser.add_argument("--model_batch", type=int, default=8192, help="Inputs per model forward pass")
    parser.add_argument("--max_queries", type=int, default=None, help="Decode only the first queries (for quick checks)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    assert args.oracle or args.checkpoint, "--checkpoint is required unless --oracle"

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    data, metadata = load_split(args.data)
    num_queries = len(data["inputs"]) if args.max_queries is None else min(args.max_queries, len(data["inputs"]))
    num_shards = math.ceil(num_queries / args.shard_size)
    n = int(round(metadata.seq_len ** 0.5))

    meta = dict(decoder="commit", checkpoint=args.checkpoint, oracle=args.oracle, data=args.data,
                num_samples=args.num_samples, choice=args.choice, order=args.order, certain=args.certain,
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
    print(f"[rank {rank}/{world_size}] {args.out}: {len(todo)} of {len(range(rank, num_shards, world_size))} shards to decode")
    if not todo:
        return

    if args.oracle:
        predict = oracle_predictor(device)
    else:
        model, step = load_model(args.checkpoint, metadata, 0.0, device)
        print(f"[rank {rank}] Loaded {args.checkpoint} (step {step}) on {device}")
        predict = model_predictor(model, args.model_batch, device, token=QUEEN)

    for done, shard in enumerate(todo, 1):
        start, end = shard * args.shard_size, min((shard + 1) * args.shard_size, num_queries)
        boards = torch.from_numpy(np.array(data["inputs"][start:end], dtype=np.int64)).view(-1, n, n).to(device)
        generator = torch.Generator(device=device).manual_seed(args.seed * 1_000_003 + start)
        preds, passes = commit(predict, boards, args.num_samples, args.choice, args.order, args.certain, generator)

        tmp_path = f"{shard_path(shard)}.tmp{rank}"
        with open(tmp_path, "wb") as f:
            np.savez(f, query_start=start, preds=preds.cpu().numpy().astype(np.uint8), passes=passes.cpu().numpy())
        os.replace(tmp_path, shard_path(shard))
        print(f"[rank {rank}] shard {shard} (queries {start}-{end - 1}) saved, {done}/{len(todo)}, "
              f"{passes.float().mean():.2f} model passes per sample", flush=True)


if __name__ == "__main__":
    main()
