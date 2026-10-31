"""What an N-Queens model's per-cell queen probabilities hold, and how much of commit decoding is the readout.

commit_decode.py lifts naive TRM from 1% (argmax over every cell's token at once) to 67% at ~1.06 model passes,
so almost nothing is re-run: the gain must come mostly from reading each row's most likely column. This tells
apart the explanations from one model pass per test board:

  blend          rows the solutions disagree on get queen mass spread over the solutions' columns (the exact
                 share); cell argmax then leaves the row empty, and the row's argmax picks one solution's column
                 per row, which need not fit together.
  under-confident the model puts its queen mass on one column per row, but below 0.5, so cell argmax drops it.

Usage: python scripts/nqueens_marginals.py --data data/nqueens-10-1oml --out results/nqueens_10_commit/marginals \\
           --runs trm_seed42=checkpoints/Nqueens-10-compare/trm_seed42/final/checkpoint.pt [name=checkpoint ...]

Reports, per run, on the rows without a given queen: valid boards from cell argmax (p >= 0.5) and from the row's
argmax (one pass, no re-running); the row's queen mass (sum of p); the normalized top (largest p over the row's
sum: the commit decoder treats >= 0.99 as sure), on rows all solutions agree on and on rows they don't; and mean
|normalized p - exact share of solutions| on rows they don't. By number of solutions. Writes <out>/summary.md.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluators.nqueens_utils import verify_nqueens  # noqa: E402
from sample_predictions import load_split, load_model  # noqa: E402
from walk_decode import model_predictor  # noqa: E402

QUEEN, EMPTY = 2, 1
SOLUTION_BUCKETS = ((1, 1), (2, 4), (5, 19), (20, 10**9))


def bucket_name(lo, hi):
    return str(lo) if lo == hi else f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}"


def score(inputs, probs, shares, num_solutions, n):
    """Per-query stats. inputs/probs/shares: (Q, n, n)."""
    open_rows = ~(inputs == QUEEN).any(2)  # (Q, n) rows the model must fill
    mass = probs.sum(2)
    norm = probs / np.maximum(mass, 1e-12)[..., None]
    top = norm.max(2)
    agreed = (shares.max(2) > 1 - 1e-9)  # every solution has its queen in the same column
    disagree = open_rows & ~agreed

    cell = np.where(inputs == QUEEN, QUEEN, np.where(probs >= 0.5, QUEEN, EMPTY))
    row = inputs.copy()
    q_idx, r_idx = np.nonzero(open_rows)
    row[q_idx, r_idx, norm.argmax(2)[q_idx, r_idx]] = QUEEN
    valid = lambda boards: np.array([verify_nqueens(b.ravel(), n) for b in boards])

    out = {}
    for name, queries in [("all", np.ones(len(inputs), bool))] + [
            (bucket_name(lo, hi), (num_solutions >= lo) & (num_solutions <= hi)) for lo, hi in SOLUTION_BUCKETS]:
        if not queries.any():
            continue
        o, d, a = open_rows[queries], disagree[queries], (open_rows & agreed)[queries]
        mean = lambda x, m: float(x[queries][m].mean()) if m.any() else None
        out[name] = {
            "queries": int(queries.sum()),
            "valid_cell_argmax": float(valid(cell[queries]).mean()),
            "valid_row_argmax": float(valid(row[queries]).mean()),
            "row_mass": mean(mass, o),
            "top_agreed_rows": mean(top, a),
            "top_disagreed_rows": mean(top, d),
            "sure_disagreed_rows": mean(top >= 0.99, d),
            "mae_share_disagreed_rows": mean(np.abs(norm - shares).sum(2) / 2, d),  # total variation per row
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="name=checkpoint pairs")
    parser.add_argument("--model_batch", type=int, default=8192)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data, metadata = load_split(args.data)
    inputs = np.array(data["inputs"], dtype=np.int64)
    n = int(round(np.sqrt(inputs.shape[1])))
    label_set = np.load(os.path.join(args.data, "test", "all__label_set.npy"))
    num_solutions = np.load(os.path.join(args.data, "test", "all__num_solutions.npy"))
    stored = np.minimum(num_solutions, label_set.shape[1])
    shares = np.stack([(label_set[q, :k] == QUEEN).mean(0) for q, k in enumerate(stored)]).reshape(-1, n, n)
    boards = inputs.reshape(-1, n, n)
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for spec in args.runs:
        name, checkpoint = spec.split("=", 1)
        model, step = load_model(checkpoint, metadata, 0.0, device)
        probs = model_predictor(model, args.model_batch, device, token=QUEEN)(torch.from_numpy(inputs)).cpu().numpy()
        results[name] = {"checkpoint": checkpoint, "step": step, **score(boards, probs.reshape(-1, n, n), shares, num_solutions, n)}
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(results[name], f, indent=2)
        print(f"{name}: valid {100 * results[name]['all']['valid_cell_argmax']:.1f}% cell argmax, "
              f"{100 * results[name]['all']['valid_row_argmax']:.1f}% row argmax", flush=True)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    fmt = lambda x, pct=False: "-" if x is None else f"{100 * x:.1f}" if pct else f"{x:.3f}"
    lines = ["# N-Queens: per-cell queen probabilities and one-pass readouts", "",
             "Rows = rows without a given queen. Agreed rows: every solution has its queen in the same column. "
             "Normalized top = largest p in the row / the row's sum (commit_decode.py commits >= 0.99 without choosing). "
             "TV to share: total variation between the row's normalized p and the exact share of solutions per column.", "",
             "| run | solutions | queries | valid, cell argmax (%) | valid, row argmax (%) | row queen mass | "
             "normalized top, agreed rows | normalized top, disagreed rows | disagreed rows >= 0.99 (%) | TV to share, disagreed rows |",
             "|" + "---|" * 10]
    for nm, r in results.items():
        for b in [k for k in r if isinstance(r[k], dict)]:
            s = r[b]
            lines.append(f"| {nm} | {b} | {s['queries']} | {fmt(s['valid_cell_argmax'], True)} | {fmt(s['valid_row_argmax'], True)} | "
                         f"{fmt(s['row_mass'])} | {fmt(s['top_agreed_rows'])} | {fmt(s['top_disagreed_rows'])} | "
                         f"{fmt(s['sure_disagreed_rows'], True)} | {fmt(s['mae_share_disagreed_rows'])} |")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
