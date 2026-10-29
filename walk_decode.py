"""Decode maze paths one decision at a time: walk from the start, feeding each committed step back to the model.

A model's per-cell probability of PATH estimates the share of shortest paths through each cell. Sampling a path
one step at a time, from the shares conditioned on the steps taken so far, samples shortest paths uniformly
(self-reducibility); taking the most likely step gives one deterministic answer. The walker re-runs the model
only where its choice is uncertain (largest normalized step probability below --certain), so a maze costs about
one model pass per fork taken. The decoder uses only the model's outputs and which neighbouring cells are open:
no distances, which would give the answer away.

Conditioning (--mode):
  start  the start token is moved to the walker's cell, and the cells walked so far are left open. Shortest
         paths from there to the goal are exactly the continuations, so any maze model works without retraining.
  hint   the cells walked so far are marked PATH in the input (models trained with path_hints).

Output, in the format of sample_predictions.py (so scripts/analyze_maze_samples.py reads it): <out>/meta.json and
<out>/shard_<i>.npz with `query_start`, `preds` (queries x samples x seq_len, uint8) and `passes` (model passes
per sample). Resumable and split across torchrun ranks by shard.

Usage:
    torchrun --nproc-per-node 1 walk_decode.py --checkpoint checkpoints/Maze-15-compare/trm_seed42/final/checkpoint.pt \\
        --data data/maze-15-1oml --out samples/maze_15/trm_seed42_walk --num_samples 100 --mode start --choice sample
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from evaluators.maze_utils import PAD, WALL, OPEN, START, GOAL, PATH, ShortestPaths
from sample_predictions import load_split, load_model

MOVES = torch.tensor([(-1, 0), (1, 0), (0, -1), (0, 1)])


def model_predictor(model, model_batch: int, device: str):
    """Returns predict(inputs (N, seq_len)) -> (N, seq_len) probability of PATH, running the model to completion."""
    one_of_many = getattr(model, "one_of_many", "naive") != "naive"

    @torch.inference_mode()
    def predict(inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for i in range(0, len(inputs), model_batch):
            x = inputs[i:i + model_batch].to(device=device, dtype=torch.int32)
            batch = {"inputs": x, "labels": x.clone(), "puzzle_identifiers": torch.zeros(len(x), dtype=torch.int32, device=device)}
            if one_of_many:
                batch["label_set"] = x.clone().unsqueeze(1)  # unused at inference, but the loss head expects it
            with torch.device(device):
                carry = model.initial_carry(batch)
            while True:
                carry, _, _, outputs, all_finish = model(carry=carry, batch=batch, return_keys={"logits"})
                if all_finish:
                    break
            out.append(torch.softmax(outputs["logits"].float(), dim=-1)[..., PATH])
        return torch.cat(out)
    return predict


def oracle_predictor(device: str):
    """Exact shares of shortest paths through each cell (start mode only): for testing the decoder."""
    def predict(inputs: torch.Tensor) -> torch.Tensor:
        n = int(round(inputs.shape[1] ** 0.5))
        shares = [ShortestPaths(row.reshape(n, n)).share.ravel() for row in inputs.cpu().numpy()]
        return torch.tensor(np.stack(shares), dtype=torch.float32, device=device)
    return predict


def walk(predict, grids: torch.Tensor, num_samples: int, mode: str, choice: str, certain: float,
         generator: torch.Generator):
    """Walk num_samples paths per maze. grids: (Q, n, n) tokens. Returns preds (Q, S, n*n) and passes (Q, S)."""
    q, n, _ = grids.shape
    device = grids.device
    g = grids.repeat_interleave(num_samples, 0)
    w = len(g)
    rows = torch.arange(w, device=device)
    moves = MOVES.to(device)
    passable = (g != WALL) & (g != PAD)
    goal = g == GOAL
    pos = (g == START).flatten(1).float().argmax(1)
    pos = torch.stack([pos // n, pos % n], 1)
    visited = g == START
    walked = torch.zeros_like(visited)  # PATH cells committed so far
    probs = torch.zeros(w, n, n, device=device)
    have = torch.zeros(w, dtype=torch.bool, device=device)  # probs computed at some point
    fresh = torch.zeros(w, dtype=torch.bool, device=device)  # probs conditioned on the current prefix
    done = torch.zeros(w, dtype=torch.bool, device=device)
    passes = torch.zeros(w, dtype=torch.int32, device=device)

    for _ in range(n * n):
        if done.all():
            break
        nr, nc = pos[:, :1] + moves[:, 0], pos[:, 1:] + moves[:, 1]  # (W, 4)
        inside = (nr >= 0) & (nr < n) & (nc >= 0) & (nc < n)
        nr, nc = nr.clamp(0, n - 1), nc.clamp(0, n - 1)
        at = lambda grid: grid[rows[:, None], nr, nc] & inside

        # Next to the goal: every shortest continuation steps onto it
        reached = ~done & at(goal).any(1)
        done |= reached
        options = at(passable & ~visited & ~goal)

        def weights():
            wt = probs[rows[:, None], nr, nc] * options
            total = wt.sum(1)
            return wt, total, wt.max(1).values / total.clamp_min(1e-12)

        wt, total, top = weights()
        stuck = ~done & ~options.any(1)  # dead end: no open cell left
        done |= stuck
        need = ~done & (~have | (~fresh & ((top < certain) | (total <= 1e-12))))
        if need.any():
            idx = need.nonzero().squeeze(1)
            x = g[idx].clone()
            if mode == "start":
                x[x == START] = OPEN
                x[torch.arange(len(idx), device=device), pos[idx, 0], pos[idx, 1]] = START
            else:
                x[walked[idx]] = PATH
            probs[idx] = predict(x.flatten(1)).view(-1, n, n)
            have[idx], fresh[idx] = True, True
            passes[idx] += 1
            wt, total, top = weights()

        active = ~done
        uniform = options.float()  # if the model puts no mass on any open cell, pick one at random
        wt = torch.where((total <= 1e-12)[:, None], uniform, wt)
        uncertain = active & (top < certain)
        pick = wt.argmax(1)
        if choice == "sample" and uncertain.any():
            sampled = torch.multinomial(wt[uncertain], 1, generator=generator).squeeze(1)
            pick[uncertain] = sampled
        step = pos + moves[pick]
        pos = torch.where(active[:, None], step, pos)
        visited[rows[active], pos[active, 0], pos[active, 1]] = True
        walked[rows[active], pos[active, 0], pos[active, 1]] = True
        fresh &= ~uncertain  # after a real choice the probs no longer condition on the walked prefix

    preds = torch.where(walked, torch.full_like(g, PATH), g)
    return preds.view(q, num_samples, n * n), passes.view(q, num_samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="Model checkpoint (omit with --oracle)")
    parser.add_argument("--oracle", action="store_true", help="Use exact path shares instead of a model (tests the decoder)")
    parser.add_argument("--data", required=True, help="Maze dataset directory; its test split is decoded")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--mode", choices=("start", "hint"), default="start")
    parser.add_argument("--choice", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--certain", type=float, default=0.99, help="Normalized step probability treated as a forced move")
    parser.add_argument("--halt_steps", type=int, default=None, help="Override the model's recursion steps per pass")
    parser.add_argument("--shard_size", type=int, default=256, help="Test mazes per shard")
    parser.add_argument("--model_batch", type=int, default=4096, help="Inputs per model forward pass")
    parser.add_argument("--max_queries", type=int, default=None, help="Decode only the first mazes (for quick checks)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    assert args.oracle or args.checkpoint, "--checkpoint is required unless --oracle"
    assert not (args.oracle and args.mode == "hint"), "the oracle supports --mode start only"

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

    meta = dict(decoder="walk", checkpoint=args.checkpoint, oracle=args.oracle, data=args.data, num_samples=args.num_samples,
                mode=args.mode, choice=args.choice, certain=args.certain, halt_steps=args.halt_steps,
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
        if args.halt_steps is not None:
            model.model.config.halt_max_steps = args.halt_steps
        print(f"[rank {rank}] Loaded {args.checkpoint} (step {step}) on {device}")
        predict = model_predictor(model, args.model_batch, device)

    for done, shard in enumerate(todo, 1):
        start, end = shard * args.shard_size, min((shard + 1) * args.shard_size, num_queries)
        grids = torch.from_numpy(np.array(data["inputs"][start:end], dtype=np.int64)).view(-1, n, n).to(device)
        generator = torch.Generator(device=device).manual_seed(args.seed * 1_000_003 + start)
        preds, passes = walk(predict, grids, args.num_samples, args.mode, args.choice, args.certain, generator)

        tmp_path = f"{shard_path(shard)}.tmp{rank}"
        with open(tmp_path, "wb") as f:
            np.savez(f, query_start=start, preds=preds.cpu().numpy().astype(np.uint8), passes=passes.cpu().numpy())
        os.replace(tmp_path, shard_path(shard))
        print(f"[rank {rank}] shard {shard} (mazes {start}-{end - 1}) saved, {done}/{len(todo)}, "
              f"{passes.float().mean():.2f} model passes per sample", flush=True)


if __name__ == "__main__":
    main()
