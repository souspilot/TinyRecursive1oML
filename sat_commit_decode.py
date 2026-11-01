"""Decode 3-SAT one variable at a time: commit values as extra givens and re-run the model (as commit_decode.py does
for N-Queens).

Each pass reads the model's probability that each unassigned variable is true, normalized over true and false.
Every variable whose likelier value is near-certain (probability >= --certain) gets it; then one undecided variable
gets a value sampled from its probability (--choice sample) or its likelier value (--choice greedy). Committed values
become givens (POS(i) / NEG(i) in the variable's slot) and the model runs again, until every variable is set. With
exact conditional probabilities and --certain above 1, this samples the formula's solutions uniformly
(self-reducibility, Jerrum, Valiant and Vazirani 1986). The decoder uses only the model's outputs: it never checks
clauses, which would be a solver.

It also records log_q, the log probability with which the sample was drawn (forced commits count as probability 1).
Since E[valid / q] over samples is the number of solutions the sampler can reach, mean(valid * exp(-log_q)) is
Knuth's estimate of the solution count (scripts/analyze_sat_samples.py reports it). It is exact for every sample
when the probabilities are exact (--oracle checks this).

Output, in the format of sample_predictions.py: <out>/meta.json and <out>/shard_<i>.npz with `query_start`, `preds`
(queries x samples x seq_len, uint8), `passes` (model passes per sample) and `log_q`. Resumable and split across
torchrun ranks by shard.

Usage:
    torchrun --nproc-per-node 1 sat_commit_decode.py --checkpoint checkpoints/Sat-16-compare/trm_seed42/final/checkpoint.pt \\
        --data data/sat-16-1oml --out samples/sat_16/trm_seed42_commit --num_samples 100
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from evaluators import sat_utils
from sample_predictions import load_split, load_model
from walk_decode import model_predictor


def oracle_predictor(n: int, device: str):
    """Exact share of the solutions extending each partial assignment that set each variable true (for testing).
    Returns (N, seq_len, 2) probabilities of FALSE and TRUE, 0.5 / 0.5 where nothing extends the assignment."""
    assignments = sat_utils.Assignments(n)

    def predict(inputs: torch.Tensor) -> torch.Tensor:
        out = np.full((len(inputs), inputs.shape[1], 2), 0.5, dtype=np.float32)
        for r, row in enumerate(inputs.cpu().numpy()):
            variables, negated, givens = sat_utils.decode(row, n)
            solutions = assignments.solutions(variables, negated, {i: bool(g) for i, g in enumerate(givens) if g >= 0})
            if len(solutions):
                share = assignments.values(solutions).mean(0)
                out[r, :n, 0], out[r, :n, 1] = 1 - share, share
        return torch.from_numpy(out).to(device)
    return predict


def commit(predict, inputs: torch.Tensor, n: int, num_samples: int, choice: str, order: str, certain: float,
           generator: torch.Generator):
    """Assign every free variable of each formula. inputs: (Q, seq_len) tokens.
    Returns preds (Q, S, seq_len), passes (Q, S) and log_q (Q, S)."""
    q, seq_len = inputs.shape
    device = inputs.device
    b = inputs.repeat_interleave(num_samples, 0).clone()
    w = len(b)
    passes = torch.zeros(w, dtype=torch.int32, device=device)
    log_q = torch.zeros(w, dtype=torch.float64, device=device)
    var = torch.arange(n, device=device)
    for _ in range(n):
        free = b[:, :n] >= 3 + 2 * n  # (W, n) variables still unassigned
        active = free.any(1)
        if not active.any():
            break
        idx = active.nonzero().squeeze(1)
        probs = predict(b[idx])[:, :n].float()  # (w, n, 2): FALSE, TRUE
        passes[idx] += 1
        free = free[idx]
        total = probs.sum(2)
        p_true = torch.where(total > 1e-12, probs[..., 1] / total.clamp_min(1e-12), torch.full_like(total, 0.5))
        top = torch.maximum(p_true, 1 - p_true)
        sure = free & (top >= certain)

        # One undecided variable per pass gets a real choice
        undecided = free & ~sure
        if order == "confident":
            score = torch.where(undecided, top, torch.full_like(top, -1.0))
        else:
            score = torch.where(undecided, torch.rand(top.shape, generator=generator, device=device), torch.full_like(top, -1.0))
        chosen_var = score.argmax(1)
        has_choice = undecided.any(1)
        rows = torch.arange(len(idx), device=device)
        p = p_true[rows, chosen_var]
        if choice == "sample":
            value = torch.rand(len(idx), generator=generator, device=device) < p
        else:
            value = p >= 0.5
        chosen_prob = torch.where(value, p, 1 - p).double().clamp_min(1e-300)
        log_q[idx] += torch.where(has_choice, chosen_prob.log(), torch.zeros_like(chosen_prob))

        values = p_true >= 0.5  # forced variables take their likelier value
        commit_mask = sure.clone()
        commit_mask[rows[has_choice], chosen_var[has_choice]] = True
        values[rows[has_choice], chosen_var[has_choice]] = value[has_choice]
        slots = b[idx, :n]
        b[idx, :n] = torch.where(commit_mask, torch.where(values, 3 + var, 3 + n + var).to(b.dtype), slots)

    preds = torch.zeros_like(b)
    preds[:, :n] = torch.where(b[:, :n] < 3 + n, sat_utils.TRUE, sat_utils.FALSE)
    return preds.view(q, num_samples, seq_len), passes.view(q, num_samples), log_q.view(q, num_samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="Model checkpoint (omit with --oracle)")
    parser.add_argument("--oracle", action="store_true", help="Use exact solution shares instead of a model (tests the decoder)")
    parser.add_argument("--data", required=True, help="3-SAT dataset directory; its test split is decoded")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--choice", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--order", choices=("confident", "random"), default="confident")
    parser.add_argument("--certain", type=float, default=0.99, help="Probability treated as a forced value (above 1: never)")
    parser.add_argument("--shard_size", type=int, default=256, help="Test queries per shard")
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
    n = sat_utils.num_vars(metadata.vocab_size)

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
        predict = oracle_predictor(n, device)
    else:
        model, step = load_model(args.checkpoint, metadata, 0.0, device)
        print(f"[rank {rank}] Loaded {args.checkpoint} (step {step}) on {device}")
        predict = model_predictor(model, args.model_batch, device, token=[sat_utils.FALSE, sat_utils.TRUE])

    for done, shard in enumerate(todo, 1):
        start, end = shard * args.shard_size, min((shard + 1) * args.shard_size, num_queries)
        inputs = torch.from_numpy(np.array(data["inputs"][start:end], dtype=np.int64)).to(device)
        generator = torch.Generator(device=device).manual_seed(args.seed * 1_000_003 + start)
        preds, passes, log_q = commit(predict, inputs, n, args.num_samples, args.choice, args.order, args.certain, generator)

        tmp_path = f"{shard_path(shard)}.tmp{rank}"
        with open(tmp_path, "wb") as f:
            np.savez(f, query_start=start, preds=preds.cpu().numpy().astype(np.uint8), passes=passes.cpu().numpy(),
                     log_q=log_q.cpu().numpy())
        os.replace(tmp_path, shard_path(shard))
        print(f"[rank {rank}] shard {shard} (queries {start}-{end - 1}) saved, {done}/{len(todo)}, "
              f"{passes.float().mean():.2f} model passes per sample", flush=True)


if __name__ == "__main__":
    main()
