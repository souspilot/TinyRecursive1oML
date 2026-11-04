"""Whether 3-SAT models solve the formulas they trained on: first-sample validity on the train and test splits.

A wide gap means the model memorized its training formulas; low validity on both means it never learned to solve them.
Validity comes from the clause checker, so unlike training metrics it counts every solution, not only stored ones,
and every formula at full recursion depth. Both splits are drawn to the same distribution of solution counts.

Usage:
    python scripts/sat_train_test.py --data data/sat-16-1oml --out results/sat_16 \\
        --runs naive=checkpoints/Sat-16-compare/trm_seed42/final/checkpoint.pt [name=checkpoint ...]

Writes <out>/train_vs_test.md and <out>/train_vs_test.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluators import sat_utils  # noqa: E402
from sample_predictions import load_split, load_model, make_batch, sample  # noqa: E402

COUNT_BUCKETS = ((1, 1), (2, 3), (4, 15), (16, 63), (64, 255), (256, 2**62))


def bucket_name(lo, hi):
    return str(lo) if lo == hi else f"{lo}+" if hi > 2**32 else f"{lo}-{hi}"


def first_sample_validity(model, data, metadata, n, max_queries, batch_size, device):
    num_queries = min(max_queries, len(data["inputs"]))
    valid = np.zeros(num_queries, dtype=bool)
    for start in range(0, num_queries, batch_size):
        end = min(start + batch_size, num_queries)
        torch.manual_seed(start)
        preds, _ = sample(model, make_batch(data, metadata, start, end), 1, device)
        valid[start:end] = sat_utils.batch_verify(np.array(data["inputs"][start:end]), preds[:, 0].cpu().numpy(), n)
    num_solutions = np.array(data["num_solutions"][:num_queries])
    by_solutions = {bucket_name(lo, hi): float(valid[(num_solutions >= lo) & (num_solutions <= hi)].mean())
                    for lo, hi in COUNT_BUCKETS if ((num_solutions >= lo) & (num_solutions <= hi)).any()}
    return {"queries": num_queries, "all": float(valid.mean()), "by_solutions": by_solutions}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="name=checkpoint pairs")
    parser.add_argument("--max_queries", type=int, default=2000, help="Formulas per split")
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for run in args.runs:
        name, checkpoint = run.split("=", 1)
        results[name] = {"checkpoint": checkpoint}
        model = None
        for split in ("train", "test"):
            data, metadata = load_split(args.data, split)
            if model is None:
                model, results[name]["step"] = load_model(checkpoint, metadata, 0.0, device)
            n = sat_utils.num_vars(metadata.vocab_size)
            results[name][split] = first_sample_validity(model, data, metadata, n, args.max_queries, args.batch_size, device)
            print(f"{name} {split}: {100 * results[name][split]['all']:.1f}% valid", flush=True)

    buckets = [bucket_name(lo, hi) for lo, hi in COUNT_BUCKETS]
    lines = ["# 3-SAT: first-sample validity on train and test formulas (%)", "",
             "| run | split | formulas | all | " + " | ".join(buckets) + " |", "|" + "---|" * (4 + len(buckets))]
    for name, r in results.items():
        for split in ("train", "test"):
            s = r[split]
            lines.append(f"| {name} | {split} | {s['queries']} | {100 * s['all']:.1f} | " + " | ".join(
                f"{100 * s['by_solutions'][b]:.1f}" if b in s["by_solutions"] else "n/a" for b in buckets) + " |")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train_vs_test.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.out, "train_vs_test.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
