"""Why walk-decoded mazes fail: classify every invalid sample of the walk runs (walk_decode.py).

Usage: python scripts/analyze_walk_failures.py --data data/maze-15-1oml --samples samples/maze_15 --out results/maze_15

For each run whose meta.json says decoder=walk, each invalid sample is one of:
  detour     reached the goal, but by a longer route
  stuck      never reached the goal (walked into a dead end)
and is located by its first wrong step: the first walked cell on no shortest path, counted in steps from the start
and as a share of the shortest length, and whether the cell before it was a fork (more than one shortest
continuation) or a forced move. Writes <out>/walk_failures.md and prints it.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluators.maze_utils import ShortestPaths, START, GOAL, PATH, path_order  # noqa: E402
from scripts.analyze_maze_samples import load_run, verify_all  # noqa: E402

COUNT_BUCKETS = ((1, 1), (2, 3), (4, 15), (16, 63), (64, 255), (256, 10**9))


def bucket_name(lo, hi):
    return str(lo) if lo == hi else f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}"


def classify(sp: ShortestPaths, pred: np.ndarray):
    """(kind, first wrong step, first wrong step / shortest length, wrong step taken at a fork)."""
    n = sp.grid.shape[0]
    path = (pred.reshape(n, n) == PATH)
    order = path_order(sp.grid[None], path[None])[0]
    goal_r, goal_c = sp.goal
    near_goal = [(goal_r + dr, goal_c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    reached = any(0 <= r < n and 0 <= c < n and (order[r, c] >= 0) for r, c in near_goal)
    kind = "detour" if reached else "stuck"
    off = path & ~sp.on_path
    if not off.any():
        return kind, None, None, None
    cells = np.argwhere(off)
    steps = order[off]
    first = cells[np.argmin(np.where(steps >= 0, steps, 10**9))]
    step = int(order[tuple(first)])
    # The cell walked just before: the neighbour one step earlier along the walk
    before = None
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        r, c = first[0] + dr, first[1] + dc
        if 0 <= r < n and 0 <= c < n and order[r, c] == step - 1:
            before = (r, c)
    fork = before is not None and sp.on_path[before] and len(sp._next_cells(before)) > 1
    return kind, step, step / max(sp.length, 1), fork


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_samples", type=int, default=20, help="Samples per maze to classify")
    args = parser.parse_args()

    inputs = np.load(os.path.join(args.data, "test", "all__inputs.npy"))
    n = int(round(np.sqrt(inputs.shape[1])))
    stats = [ShortestPaths(row.reshape(n, n)) for row in inputs]
    num_paths = np.array([sp.num_paths for sp in stats])
    lines = ["# Walk decoding failures", "",
             f"Invalid samples among the first {args.max_samples} per maze. First wrong step: the first walked cell on no "
             "shortest path; at a fork = the cell before it had several shortest continuations.", "",
             "| run | invalid (%) | detour | stuck | wrong turn at a fork | first wrong step (share of length) | "
             + " | ".join(f"invalid, {bucket_name(lo, hi)} paths" for lo, hi in COUNT_BUCKETS) + " |",
             "|" + "---|" * (6 + len(COUNT_BUCKETS))]
    os.makedirs(args.out, exist_ok=True)
    for meta_path in sorted(glob.glob(os.path.join(args.samples, "*", "meta.json"))):
        with open(meta_path) as f:
            if json.load(f).get("decoder") != "walk":
                continue
        run_dir = os.path.dirname(meta_path)
        preds, _, _ = load_run(run_dir)
        if preds is None:
            continue
        preds = preds[:, :args.max_samples]
        valid = verify_all(inputs[:len(preds)], preds)
        kinds, forks, shares = [], [], []
        for m, s in zip(*np.where(~valid)):
            kind, _, share, fork = classify(stats[m], preds[m, s])
            kinds.append(kind)
            if share is not None:
                shares.append(share)
                forks.append(fork)
        kinds = np.array(kinds)
        by_bucket = []
        for lo, hi in COUNT_BUCKETS:
            mazes = (num_paths[:len(preds)] >= lo) & (num_paths[:len(preds)] <= hi)
            by_bucket.append(f"{100 * (~valid[mazes]).mean():.1f}" if mazes.any() else "-")
        pct = lambda x: f"{100 * x:.1f}" if len(kinds) else "-"
        lines.append(f"| {os.path.basename(run_dir)} | {100 * (~valid).mean():.1f} | {pct((kinds == 'detour').mean())} | "
                     f"{pct((kinds == 'stuck').mean())} | {pct(np.mean(forks)) if forks else '-'} | "
                     f"{np.mean(shares):.2f} | " + " | ".join(by_bucket) + " |" if shares else
                     f"| {os.path.basename(run_dir)} | {100 * (~valid).mean():.1f} | - | - | - | - | " + " | ".join(by_bucket) + " |")
    lines.append("")
    lines.append("detour / stuck / at a fork: shares of the invalid samples.")
    with open(os.path.join(args.out, "walk_failures.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
