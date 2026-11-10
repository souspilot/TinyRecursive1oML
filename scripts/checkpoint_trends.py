"""Print how validity and diversity change over a run's checkpoints, with the summary used to compare methods.

Usage: python scripts/checkpoint_trends.py <run> [maze nqueens sudoku graph]    (after scripts/score_checkpoints.sh)

ratio and distinct are on multi-solution inputs, among 100 samples; unique is the validity on unique-solution inputs.
Single checkpoints are noisy (ratio moves by ~0.04 between neighbouring ones), so:
  summary       the mean of the last LAST checkpoints, used to compare methods and to choose beta
  trend         a straight line fitted to ratio over the second half of training (gain per 100k steps, +- 1 SE)
  train longer  the trend gains at least MIN_GAIN per 100k steps and is more than 2 SE above zero
"""
import glob
import json
import os
import re
import sys

import numpy as np

LAST = 4
MIN_GAIN = 0.06  # per 100k steps, i.e. 0.03 over the last third of a 150k-step run


def load_checkpoints(results_dir, run):
    """(step, result) in order of step, for every scored checkpoint of the run."""
    found = []
    for path in glob.glob(os.path.join(results_dir, f"{run}_step*.json")):
        match = re.fullmatch(re.escape(run) + r"_step(\d+)\.json", os.path.basename(path))
        if match:
            with open(path) as f:
                found.append((int(match.group(1)), json.load(f)))
    return sorted(found, key=lambda item: item[0])


def row(result):
    groups = result["groups"]
    diversity = groups["all"]["diversity"][str(result["num_samples"])]
    return dict(validity=100 * groups["all"]["validity"], ratio=diversity["ratio_to_uniform"],
                distinct=diversity["distinct_valid"], unique=100 * groups["1"]["validity"])


def trend(steps, ratios):
    """Fitted gain per 100k steps and its standard error, over the second half of training."""
    steps, ratios = np.array(steps), np.array(ratios)
    half = steps >= steps[-1] / 2
    if half.sum() < 4:
        return None, None
    x, y = steps[half] / 1e5, ratios[half]
    slope, intercept = np.polyfit(x, y, 1)
    residual_sd = (y - (slope * x + intercept)).std(ddof=2)
    return slope, residual_sd / np.sqrt(((x - x.mean()) ** 2).sum())


def report(task, run):
    checkpoints = load_checkpoints(f"results/{task}-val", run)
    if not checkpoints:
        print(f"{task}: no scored checkpoints in results/{task}-val\n")
        return
    steps = [step for step, _ in checkpoints]
    rows = [row(result) for _, result in checkpoints]
    first = checkpoints[0][1]
    print(f"{task}: {run}, {first['num_queries']} validation inputs x {first['num_samples']} samples")
    print(f"{'step':>8} {'validity':>9} {'ratio':>6} {'distinct':>9} {'unique':>7}")
    for step, r in zip(steps, rows):
        print(f"{step:>8} {r['validity']:>9.1f} {r['ratio']:>6.3f} {r['distinct']:>9.1f} {r['unique']:>7.1f}")
    last = {k: np.mean([r[k] for r in rows[-LAST:]]) for k in rows[0]}
    print(f"{f'last {min(LAST, len(rows))}':>8} {last['validity']:>9.1f} {last['ratio']:>6.3f} {last['distinct']:>9.1f} "
          f"{last['unique']:>7.1f}   <- summary")
    slope, se = trend(steps, [r["ratio"] for r in rows])
    if slope is None:
        print("Too few checkpoints in the second half for a trend.\n")
        return
    longer = slope >= MIN_GAIN and slope > 2 * se
    print(f"Ratio trend over the second half: {slope:+.3f} +- {se:.3f} per 100k steps; "
          f"train longer: {'yes' if longer else 'no'}\n")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    run = sys.argv[1]
    for task in sys.argv[2:] or ["maze", "nqueens", "sudoku", "graph"]:
        report(task, run)


if __name__ == "__main__":
    main()
