"""How well a maze model's per-cell probability of PATH matches the exact share of shortest paths through each cell.

The walk decoder (walk_decode.py) samples uniformly only if these probabilities are close to the exact shares,
so this checks its premise before any decoding: a model trained on one random path per maze should predict the
shares even though its argmax (all cells at once) blends paths.

Usage: python scripts/maze_marginals.py --data data/maze-15-1oml --out results/maze_15/marginals \\
           --runs trm_seed42=checkpoints/Maze-15-compare/trm_seed42/final/checkpoint.pt [name=checkpoint ...]

Reports, per run, over the cells a path could use (open cells): mean |p - share| on all mazes and by number of
shortest paths, the same on cells only some shortest paths use (where the choice is), and calibration (mean share
per bin of p). Writes <out>/summary.md and <out>/<name>.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluators.maze_utils import ShortestPaths, OPEN  # noqa: E402
from sample_predictions import load_split, load_model  # noqa: E402
from walk_decode import model_predictor  # noqa: E402

COUNT_BUCKETS = ((1, 1), (2, 3), (4, 15), (16, 63), (64, 255), (256, 10**9))
BINS = np.linspace(0, 1, 11)


def bucket_name(lo, hi):
    return str(lo) if lo == hi else f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}"


def score(probs, shares, open_cells, num_paths):
    err = np.abs(probs - shares)
    choice = open_cells & (shares > 0) & (shares < 1 - 1e-9)
    by_paths = {}
    for lo, hi in COUNT_BUCKETS:
        mazes = (num_paths >= lo) & (num_paths <= hi)
        if mazes.any():
            by_paths[bucket_name(lo, hi)] = float(err[mazes][open_cells[mazes]].mean())
    p, s = probs[open_cells], shares[open_cells]
    bins = np.clip(np.digitize(p, BINS) - 1, 0, len(BINS) - 2)
    calibration = [{"bin": [float(BINS[b]), float(BINS[b + 1])], "cells": int((bins == b).sum()),
                    "mean_p": float(p[bins == b].mean()) if (bins == b).any() else None,
                    "mean_share": float(s[bins == b].mean()) if (bins == b).any() else None} for b in range(len(BINS) - 1)]
    ece = sum(c["cells"] * abs(c["mean_p"] - c["mean_share"]) for c in calibration if c["cells"]) / max(len(p), 1)
    return {"mae_open": float(err[open_cells].mean()), "mae_choice_cells": float(err[choice].mean()),
            "mae_by_num_paths": by_paths, "ece": float(ece), "calibration": calibration}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="name=checkpoint pairs")
    parser.add_argument("--model_batch", type=int, default=2048)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data, metadata = load_split(args.data)
    inputs = np.array(data["inputs"], dtype=np.int64)
    n = int(round(np.sqrt(inputs.shape[1])))
    stats = [ShortestPaths(row.reshape(n, n)) for row in inputs]
    shares = np.stack([sp.share.ravel() for sp in stats])
    num_paths = np.array([sp.num_paths for sp in stats])
    open_cells = inputs == OPEN
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for spec in args.runs:
        name, checkpoint = spec.split("=", 1)
        model, step = load_model(checkpoint, metadata, 0.0, device)
        probs = model_predictor(model, args.model_batch, device)(torch.from_numpy(inputs)).cpu().numpy()
        results[name] = {"checkpoint": checkpoint, "step": step, **score(probs, shares, open_cells, num_paths)}
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(results[name], f, indent=2)
        print(f"{name}: mean |p - share| {results[name]['mae_open']:.4f} on open cells, "
              f"{results[name]['mae_choice_cells']:.4f} on choice cells, ECE {results[name]['ece']:.4f}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    names = list(results)
    buckets = list(results[names[0]]["mae_by_num_paths"])
    lines = ["# Per-cell PATH probability vs exact share of shortest paths", "",
             "Mean |p - share| over open cells (lower is better); choice cells: used by some but not all shortest paths.", "",
             "| run | all open cells | choice cells | ECE | " + " | ".join(f"{b} paths" for b in buckets) + " |",
             "|" + "---|" * (4 + len(buckets))]
    for nm in names:
        r = results[nm]
        lines.append(f"| {nm} | {r['mae_open']:.4f} | {r['mae_choice_cells']:.4f} | {r['ece']:.4f} | "
                     + " | ".join(f"{r['mae_by_num_paths'][b]:.4f}" for b in buckets) + " |")
    lines += ["", "## Calibration: mean exact share per bin of predicted p", "",
              "| run | " + " | ".join(f"{BINS[b]:.1f}-{BINS[b + 1]:.1f}" for b in range(len(BINS) - 1)) + " |",
              "|" + "---|" * len(BINS)]
    for nm in names:
        cal = results[nm]["calibration"]
        lines.append(f"| {nm} | " + " | ".join("-" if c["mean_share"] is None else f"{c['mean_share']:.2f}" for c in cal) + " |")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
