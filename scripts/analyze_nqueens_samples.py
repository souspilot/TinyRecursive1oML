"""Analyze N-Queens predictions sampled by sample_predictions.py.

Usage: python scripts/analyze_nqueens_samples.py --data data/nqueens-10-1oml --samples samples/nqueens_10 \\
           --out results/hidden_constraints_10

Every run in --samples (a directory per run, holding meta.json and shards) is compared against the full
solution set of each test query (label_set of a --one-of-many dataset). Reports, per run:

1. Hidden-constraint test. For each multi-solution query, every non-given cell that holds a queen in some
   but not all of its k solutions defines a hidden requirement "a queen goes here", met by a fraction f of
   the solutions. The model never sees the requirement; an external checker accepts a sample only if it is a
   valid solution that meets it. Success with K samples = some sample among the first K is accepted.
   Requirements are grouped by f; each query counts once per group (its requirements in the group averaged).
   References: f itself (expected success of an answer unrelated to the requirement) and the best any single
   fixed answer could do (the query's solution meeting the most requirements in the group).
2. Accuracy vs number of samples K: majority vote without a verifier, any valid sample with a verifier, and
   coverage (share of the query's solutions found), for all, one-solution and multi-solution queries.
3. First-sample accuracy by number of solutions and by number of given queens.
4. Variety on one-solution queries: distinct answers among the samples, and share of valid samples.

Writes <out>/<run>.json, <out>/summary.md and two figures.
"""
import argparse
import glob
import json
import os

import numpy as np

SAMPLE_COUNTS = (1, 2, 3, 5, 10, 20, 50, 100, 250, 500, 1000)
F_EDGES = np.linspace(0, 1, 11)
SOLUTION_BUCKETS = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 9), (10, 19), (20, 49), (50, 10**9))


def load_test(data_path):
    split = os.path.join(data_path, "test")
    inputs = np.load(os.path.join(split, "all__inputs.npy"))
    label_set = np.load(os.path.join(split, "all__label_set.npy"))
    num_solutions = np.load(os.path.join(split, "all__num_solutions.npy"))
    return inputs, label_set, num_solutions


def load_run(run_dir):
    """Samples of one run as (queries, samples, seq_len), or None if shards are missing."""
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    paths = [os.path.join(run_dir, f"shard_{i:05d}.npz") for i in range(meta["num_shards"])]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"Skipping {run_dir}: {len(missing)} of {len(paths)} shards missing (sampling unfinished)")
        return None, meta
    shards = [np.load(p) for p in paths]
    preds = np.concatenate([s["preds"] for s in sorted(shards, key=lambda s: int(s["query_start"]))])
    return preds, meta


def nqueens_valid(boards, inputs):
    """Vectorized N-Queens check: one queen per row and column, at most one per diagonal, given queens kept.

    boards: (..., n*n) with 2 = queen; inputs: broadcastable to boards."""
    n = int(round(np.sqrt(boards.shape[-1])))
    queens = (boards == 2).reshape(*boards.shape[:-1], n, n)
    rows_ok = (queens.sum(-1) == 1).all(-1)
    cols_ok = (queens.sum(-2) == 1).all(-1)
    r, c = np.indices((n, n))
    diag = np.zeros((n * n, 2 * (2 * n - 1)), dtype=np.int32)
    diag[np.arange(n * n), (r - c + n - 1).ravel()] = 1
    diag[np.arange(n * n), (2 * n - 1 + r + c).ravel()] = 1
    diags_ok = ((queens.reshape(*boards.shape[:-1], n * n).astype(np.int32) @ diag) <= 1).all(-1)
    givens_ok = ((inputs != 2) | (boards == 2)).all(-1)
    return rows_ok & cols_ok & diags_ok & givens_ok


def match_solutions(preds, label_set, num_solutions):
    """Index of the solution each sample equals (-1 if none), and an id per distinct board for voting."""
    num_queries, num_samples, _ = preds.shape
    solution_index = np.full((num_queries, num_samples), -1, dtype=np.int64)
    board_id = np.empty((num_queries, num_samples), dtype=np.int64)
    for q in range(num_queries):
        k = int(num_solutions[q])
        ids = {label_set[q, j].tobytes(): j for j in range(k)}
        for s in range(num_samples):
            key = preds[q, s].tobytes()
            if key not in ids:
                ids[key] = len(ids)  # an invalid board, numbered after the k solutions
            board_id[q, s] = ids[key]
            if ids[key] < k:
                solution_index[q, s] = ids[key]
    return solution_index, board_id


def majority_correct(board_id, num_solutions, k_samples):
    """Whether the most frequent board among the first k_samples is a solution (ties: earliest seen)."""
    correct = np.zeros(len(board_id), dtype=bool)
    for q, ids in enumerate(board_id[:, :k_samples]):
        values, first, counts = np.unique(ids, return_index=True, return_counts=True)
        winner = values[np.lexsort((first, -counts))[0]]
        correct[q] = winner < num_solutions[q]
    return correct


def hidden_constraints(inputs, label_set, num_solutions, solution_index, sample_counts):
    """Success per requirement-difficulty group (see module docstring)."""
    num_bins = len(F_EDGES) - 1
    success = {k: np.zeros(num_bins) for k in sample_counts}
    sum_f, best_single, num_queries = np.zeros(num_bins), np.zeros(num_bins), np.zeros(num_bins)
    num_samples = solution_index.shape[1]
    for q in np.where(num_solutions > 1)[0]:
        k = int(num_solutions[q])
        queens = label_set[q, :k] == 2
        counts = queens.sum(0)
        cells = np.where((inputs[q] != 2) & (counts > 0) & (counts < k))[0]
        if len(cells) == 0:
            continue
        f = counts[cells] / k
        # First sample that found each solution, then the first sample that meets each requirement
        first_found = np.full(k, num_samples)
        for s, j in enumerate(solution_index[q]):
            if j >= 0 and first_found[j] == num_samples:
                first_found[j] = s
        first_met = np.where(queens[:, cells], first_found[:, None], num_samples).min(0)

        bins = np.clip(np.digitize(f, F_EDGES) - 1, 0, num_bins - 1)
        for b in np.unique(bins):
            in_bin = bins == b
            num_queries[b] += 1
            sum_f[b] += f[in_bin].mean()
            best_single[b] += queens[:, cells[in_bin]].mean(1).max()
            for n in sample_counts:
                success[n][b] += (first_met[in_bin] < n).mean()

    per_query = lambda x: np.where(num_queries > 0, x / np.maximum(num_queries, 1), np.nan)
    return {
        "f_bins": [[float(F_EDGES[b]), float(F_EDGES[b + 1])] for b in range(num_bins)],
        "queries": num_queries.astype(int).tolist(),
        "mean_f": per_query(sum_f).tolist(),
        "best_single_answer": per_query(best_single).tolist(),
        "success": {str(n): per_query(success[n]).tolist() for n in sample_counts},
    }


def analyze_run(preds, inputs, label_set, num_solutions):
    num_samples = preds.shape[1]
    sample_counts = [k for k in SAMPLE_COUNTS if k <= num_samples]
    solution_index, board_id = match_solutions(preds, label_set, num_solutions)

    # The solution sets are complete, so a board is valid exactly when it is one of them
    valid = nqueens_valid(preds, inputs[:, None, :])
    disagreements = int((valid != (solution_index >= 0)).sum())

    subsets = {"all": np.ones(len(inputs), bool), "one_solution": num_solutions == 1, "multi_solution": num_solutions > 1}
    by_samples = {}
    for n in sample_counts:
        found = solution_index[:, :n] >= 0
        distinct_found = np.array([len(set(row[row >= 0])) for row in solution_index[:, :n]])
        majority = majority_correct(board_id, num_solutions, n)
        by_samples[str(n)] = {
            name: {
                "majority_vote": float(majority[mask].mean()),
                "any_valid": float(found[mask].any(1).mean()),
                "coverage": float((distinct_found[mask] / num_solutions[mask]).mean()),
            }
            for name, mask in subsets.items()
        }

    givens = (inputs == 2).sum(1)
    first_correct = solution_index[:, 0] >= 0
    by_solutions = {}
    for lo, hi in SOLUTION_BUCKETS:
        in_bucket = (num_solutions >= lo) & (num_solutions <= hi)
        if not in_bucket.any():
            continue
        name = str(lo) if lo == hi else f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
        by_solutions[name] = {
            "queries": int(in_bucket.sum()),
            "first_sample_accuracy": float(first_correct[in_bucket].mean()),
            "by_givens": {str(g): {"queries": int((in_bucket & (givens == g)).sum()),
                                   "first_sample_accuracy": float(first_correct[in_bucket & (givens == g)].mean())}
                          for g in np.unique(givens[in_bucket])},
        }

    one_solution = num_solutions == 1
    variety = {
        "distinct_answers": float(np.mean([len(np.unique(row)) for row in board_id[one_solution]])),
        "valid_sample_share": float((solution_index[one_solution] >= 0).mean()),
    } if one_solution.any() else None

    return {
        "num_samples": num_samples,
        "num_queries": len(inputs),
        "validity_check_disagreements": disagreements,
        "by_samples": by_samples,
        "hidden_constraints": hidden_constraints(inputs, label_set, num_solutions, solution_index, sample_counts),
        "by_solutions": by_solutions,
        "one_solution_variety": variety,
    }


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def write_summary(results, out_dir):
    names = sorted(results)
    lines = ["# Hidden-constraint experiment", ""]

    lines += ["## Hidden-constraint test", "",
              "Success (%) when an unseen requirement met by a fraction f of a query's solutions must also hold. "
              "`@K` = accepted sample among the first K, picked by an external checker.", ""]
    hc = {n: results[n]["hidden_constraints"] for n in names}
    first = hc[names[0]]
    columns = [(n, k) for n in names for k in hc[n]["success"] if k in ("1", "5", "20", str(results[n]["num_samples"]))]
    columns = list(dict.fromkeys(columns))
    lines.append("| f | queries | f (unrelated answer) | best single answer | " + " | ".join(f"{n} @{k}" for n, k in columns) + " |")
    lines.append("|" + "---|" * (4 + len(columns)))
    for b, (lo, hi) in enumerate(first["f_bins"]):
        if first["queries"][b] == 0:
            continue
        lines.append(f"| {lo:.1f}-{hi:.1f} | {first['queries'][b]} | {pct(first['mean_f'][b])} | {pct(first['best_single_answer'][b])} | "
                     + " | ".join(pct(hc[n]["success"][k][b]) for n, k in columns) + " |")

    lines += ["", "## Accuracy vs number of samples", "",
              "Majority vote needs no verifier; any-valid assumes one picks a valid sample. Coverage = share of a query's solutions found.", ""]
    lines.append("| run | samples | majority (all) | majority (one-sol) | majority (multi-sol) | any-valid (all) | coverage (multi-sol) |")
    lines.append("|---|---|---|---|---|---|---|")
    for n in names:
        for k, m in results[n]["by_samples"].items():
            lines.append(f"| {n} | {k} | {pct(m['all']['majority_vote'])} | {pct(m['one_solution']['majority_vote'])} | "
                         f"{pct(m['multi_solution']['majority_vote'])} | {pct(m['all']['any_valid'])} | {pct(m['multi_solution']['coverage'])} |")

    lines += ["", "## First-sample accuracy by number of solutions (and given queens)", ""]
    buckets = list(results[names[0]]["by_solutions"])
    lines.append("| run | " + " | ".join(f"{b} sol (n={results[names[0]]['by_solutions'][b]['queries']})" for b in buckets) + " |")
    lines.append("|" + "---|" * (1 + len(buckets)))
    for n in names:
        lines.append(f"| {n} | " + " | ".join(pct(results[n]["by_solutions"][b]["first_sample_accuracy"]) for b in buckets) + " |")
    lines += ["", "Split by given queens (accuracy %, queries in brackets):", ""]
    for b in buckets:
        givens = results[names[0]]["by_solutions"][b]["by_givens"]
        lines.append(f"- **{b} solutions**: " + "; ".join(
            f"{g} given (n={givens[g]['queries']}): " + ", ".join(f"{n} {pct(results[n]['by_solutions'][b]['by_givens'][g]['first_sample_accuracy'])}" for n in names)
            for g in givens))

    lines += ["", "## Variety on one-solution queries", "", "| run | samples | distinct answers | valid samples (%) |", "|---|---|---|---|"]
    for n in names:
        v = results[n]["one_solution_variety"]
        if v:
            lines.append(f"| {n} | {results[n]['num_samples']} | {v['distinct_answers']:.2f} | {pct(v['valid_sample_share'])} |")

    disagreements = {n: results[n]["validity_check_disagreements"] for n in names if results[n]["validity_check_disagreements"]}
    lines += ["", f"Validity cross-check: {'all samples agree' if not disagreements else f'DISAGREEMENTS {disagreements}'} "
                  "(rule-based N-Queens check vs membership in the solution set)."]

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(results)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    hc0 = results[names[0]]["hidden_constraints"]
    keep = [b for b, q in enumerate(hc0["queries"]) if q > 0]
    x = [np.mean(hc0["f_bins"][b]) for b in keep]
    ax.plot(x, [hc0["mean_f"][b] for b in keep], "k--", label="unrelated single answer (f)")
    ax.plot(x, [hc0["best_single_answer"][b] for b in keep], "k:", label="best single answer")
    for n in names:
        success = results[n]["hidden_constraints"]["success"]
        for k in sorted({"1", str(results[n]["num_samples"])}, key=int):
            ax.plot(x, [success[k][b] for b in keep], marker="o", label=f"{n} @{k}")
    ax.set_xlabel("f: share of a query's solutions meeting the hidden requirement")
    ax.set_ylabel("success")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hidden_constraints.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in names:
        by_samples = results[n]["by_samples"]
        ks = sorted(by_samples, key=int)
        if len(ks) == 1:
            ax.axhline(by_samples[ks[0]]["all"]["majority_vote"], linestyle=":", label=f"{n} (1 sample)")
            continue
        line, = ax.plot([int(k) for k in ks], [by_samples[k]["all"]["majority_vote"] for k in ks], marker="o", label=f"{n} majority vote")
        ax.plot([int(k) for k in ks], [by_samples[k]["all"]["any_valid"] for k in ks], marker="x", linestyle="--",
                color=line.get_color(), label=f"{n} any valid (verifier)")
    ax.set_xscale("log")
    ax.set_xlabel("samples per query")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_vs_samples.png"), dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--samples", required=True, help="Directory with one subdirectory per run")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    inputs, label_set, num_solutions = load_test(args.data)
    os.makedirs(args.out, exist_ok=True)
    results = {}
    for run_dir in sorted(glob.glob(os.path.join(args.samples, "*", "meta.json"))):
        run_dir = os.path.dirname(run_dir)
        preds, meta = load_run(run_dir)
        if preds is None:
            continue
        name = os.path.basename(run_dir)
        print(f"Analyzing {name} ({preds.shape[1]} samples per query)")
        results[name] = {"meta": meta, **analyze_run(preds, inputs, label_set, num_solutions)}
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(results[name], f, indent=2)
    if not results:
        print("No complete runs to analyze")
        return
    write_summary(results, args.out)
    plot(results, args.out)
    print(f"\nSaved {args.out}/summary.md, hidden_constraints.png, accuracy_vs_samples.png")


if __name__ == "__main__":
    main()
