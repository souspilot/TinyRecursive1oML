"""Print how validity and diversity change over a run's checkpoints, and apply the rules for reading them.

Usage: python scripts/checkpoint_trends.py <run> [maze nqueens sudoku graph]    (after scripts/score_checkpoints.sh)

ratio and distinct are on multi-solution inputs, among 100 samples; unique is the validity on unique-solution inputs.
Rules, compared between the last checkpoint and the one nearest two thirds of the way through:
  train longer  ratio gained more than RATIO_GAIN, or validity gained more than VALIDITY_GAIN points
  unstable      the last checkpoint is below the run's best by more than PEAK_RATIO or PEAK_VALIDITY points
"""
import glob
import json
import os
import re
import sys

RATIO_GAIN = 0.03  # provisional, until a second seed gives the real noise level
VALIDITY_GAIN = 2.0
PEAK_RATIO = 0.05
PEAK_VALIDITY = 2.0


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
    samples = str(result["num_samples"])
    diversity = groups["all"]["diversity"][samples]
    return dict(validity=100 * groups["all"]["validity"], ratio=diversity["ratio_to_uniform"],
                distinct=diversity["distinct_valid"], unique=100 * groups["1"]["validity"])


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

    last = rows[-1]
    index = min(range(len(steps)), key=lambda i: abs(steps[i] - steps[-1] * 2 / 3))
    if index == len(steps) - 1:
        print("Too few checkpoints to apply the rules.\n")
        return
    ratio_gain = last["ratio"] - rows[index]["ratio"]
    validity_gain = last["validity"] - rows[index]["validity"]
    longer = ratio_gain > RATIO_GAIN or validity_gain > VALIDITY_GAIN
    ratio_drop = max(r["ratio"] for r in rows) - last["ratio"]
    validity_drop = max(r["validity"] for r in rows) - last["validity"]
    unstable = ratio_drop > PEAK_RATIO or validity_drop > PEAK_VALIDITY
    print(f"From step {steps[index]} to {steps[-1]}: ratio {ratio_gain:+.3f}, validity {validity_gain:+.1f} points; "
          f"train longer: {'yes' if longer else 'no'}")
    print(f"Below the run's best by: ratio {ratio_drop:.3f}, validity {validity_drop:.1f} points; "
          f"unstable: {'yes' if unstable else 'no'}\n")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    run = sys.argv[1]
    for task in sys.argv[2:] or ["maze", "nqueens", "sudoku", "graph"]:
        report(task, run)


if __name__ == "__main__":
    main()
