"""Summarize N-Queens pass@k results written by scripts/nqueens_eval_all.sh into one comparison table.

Usage: python scripts/summarize_nqueens.py results/nqueens_8

Reads every <run>.json in the directory, prints a Markdown table, and writes summary.md and summary.csv
next to them. "One-sol" and "multi-sol" split test queries by how many valid completions they have, as in
the 1oML paper (https://arxiv.org/abs/2008.11990). Solution counts above 20 are grouped with 20.
"""
import csv
import glob
import json
import os
import sys

METHOD_ORDER = ["gram", "trm", "trm_random", "trm_unique", "trm_minloss", "trm_iexplr", "trm_selectr"]

COLUMNS = [
    ("run", "Run"),
    ("num_samples", "Samples"),
    ("pass1", "pass@1"),
    ("pass1_os", "pass@1 one-sol"),
    ("pass1_ms", "pass@1 multi-sol"),
    ("majority", "Majority vote"),
    ("passk", "pass@K"),
    ("coverage_ms", "Multi-sol coverage"),
    ("unique", "Unique answers"),
    ("puzzles", "Test queries"),
]


def bucket_average(metrics, name, buckets):
    """Average a per-solution-count metric over the given buckets, weighted by their query counts."""
    total = weighted = 0.0
    for num_sol in buckets:
        count = metrics.get(f"NQueens/count_{num_sol}sol", 0)
        if count and f"NQueens/{name}_{num_sol}sol" in metrics:
            total += count
            weighted += count * metrics[f"NQueens/{name}_{num_sol}sol"]
    return weighted / total if total else None


def summarize(result):
    metrics, k = result["metrics"], result["num_samples"]
    return {
        "run": result["run"],
        "num_samples": k,
        "pass1": metrics["NQueens/exact_pass@1"],
        "pass1_os": bucket_average(metrics, "exact_pass@1", [1]),
        "pass1_ms": bucket_average(metrics, "exact_pass@1", range(2, 21)),
        "majority": metrics["NQueens/majority_exact"],
        "passk": metrics.get(f"NQueens/exact_pass@{k}"),
        "coverage_ms": bucket_average(metrics, "answer_coverage", range(2, 21)),
        "unique": metrics["NQueens/avg_unique_samples"],
        "puzzles": int(metrics["NQueens/total_puzzles"]),
    }


def sort_key(row):
    method = row["run"].rsplit("_seed", 1)[0]
    return (METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER), row["run"])


PERCENT_KEYS = {"pass1", "pass1_os", "pass1_ms", "majority", "passk", "coverage_ms"}


def fmt(key, value):
    if value is None:
        return "n/a"
    if key in PERCENT_KEYS:
        return f"{100 * value:.2f}"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main(results_dir):
    rows = []
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path) as f:
            rows.append(summarize(json.load(f)))
    if not rows:
        print(f"No results found in {results_dir}")
        return
    rows.sort(key=sort_key)

    header = "| " + " | ".join(title for _, title in COLUMNS) + " |"
    lines = [header, "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(fmt(key, row[key]) for key, _ in COLUMNS) + " |" for row in rows]
    lines += [
        "",
        "Accuracies are % of test queries. pass@1 and its splits use the first sample; Majority vote and",
        "pass@K use all K samples. Multi-sol coverage is the share of each query's solutions found among the",
        "K samples. TRM runs are deterministic, so they use 1 sample and their pass@K equals pass@1.",
    ]
    table = "\n".join(lines)
    print(table)

    with open(os.path.join(results_dir, "summary.md"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(results_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[key for key, _ in COLUMNS])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {os.path.join(results_dir, 'summary.md')} and summary.csv")


if __name__ == "__main__":
    main(sys.argv[1])
