"""Score samples from sample_predictions.py on the five multi-solution datasets with one set of metrics.

Usage: python scripts/analyze_multi_samples.py --task sudoku --data data/sudoku-multi-1oml --samples samples/sudoku --out results/sudoku

--data is the -1oml dataset the samples came from (its label_set holds each input's stored solutions); each run in
--samples is a directory with meta.json and shards, sampled from the split named in its meta.json (test by default).
Tasks: maze, nqueens, sudoku, graph, sat. Validity uses the task's own checker (evaluators/*_utils.py). Two
samples are the same solution when they agree on the cells that make up the answer (for graph colouring, after
renaming colours to canonical form, so a colouring and its relabelling count once).

Reported per run, for all inputs and per group of solution counts (1, 2-3, 4-15, 16-63, 64-255, 256+):

1. Validity: mean over all samples (the expected single-sample validity), and any-valid among the first K samples.
2. Diversity, on multi-solution inputs: distinct valid solutions among the first K samples, against the number a
   uniform sampler over the input's solutions would find with the same number of valid draws (n(1 - (1 - 1/n)^v)).
   The ratio separates diversity from validity: 1 = as diverse as uniform sampling, 1/v = one solution repeated.
   Coverage = distinct valid / number of solutions.
3. Unique-solution inputs: distinct answers among the samples (1 = the noise is ignored where it should be).
4. Error anatomy (invalid samples): cells differing from the nearest stored solution, and on multi-solution inputs
   the share of invalid samples whose every cell agrees with some stored solution while matching none: a blend of
   solutions rather than an error of their own. Stored solutions are at most 16 per input, so both are bounds.
5. Held-out solutions (train split only): among valid samples on inputs with more solutions than stored, the share
   not among the stored ones, against a uniform sampler's 1 - stored / solutions. Near 0 = only reproduces what it
   was trained on; near the uniform share = produces unseen solutions as readily as seen ones.

Writes <out>/<run>.json and <out>/summary.md.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluators import sat_utils  # noqa: E402
from evaluators.graph_coloring_utils import verify_coloring  # noqa: E402
from evaluators.maze_utils import batch_verify_maze  # noqa: E402
from evaluators.nqueens_utils import verify_nqueens  # noqa: E402
from evaluators.sudoku_utils import verify_sudoku  # noqa: E402

TASKS = ("maze", "nqueens", "sudoku", "graph", "sat")
SAMPLE_COUNTS = (1, 2, 5, 10, 20, 50, 100, 250, 500, 1000)
COUNT_BUCKETS = ((1, 1), (2, 3), (4, 15), (16, 63), (64, 255), (256, 2**62))


class Task:
    """Validity and the cells that make up an answer, for one task's token encoding."""

    def __init__(self, name: str, seq_len: int, vocab_size: int):
        self.name = name
        if name == "graph":
            self.answer_len = int((-1 + np.sqrt(1 + 8 * seq_len)) / 2)  # nodes; the colours are the first n tokens
        elif name == "sat":
            self.answer_len = sat_utils.num_vars(vocab_size)  # the assignment is the first n tokens
        else:
            self.answer_len = seq_len

    def answer(self, seqs: np.ndarray) -> np.ndarray:
        """(..., answer_len) answer cells; graph colourings in canonical form (colours by first appearance)."""
        cells = seqs[..., :self.answer_len]
        if self.name != "graph":
            return cells
        flat = cells.reshape(-1, self.answer_len)
        canon = np.empty_like(flat)
        for i, row in enumerate(flat):
            _, first = np.unique(row, return_index=True)
            order = row[np.sort(first)]
            canon[i] = 3 + np.argmax(row[:, None] == order[None, :], axis=1)
        return canon.reshape(cells.shape)

    def valid(self, inputs: np.ndarray, preds: np.ndarray) -> np.ndarray:
        """(queries, samples) bool, for inputs (queries, seq_len) and preds (queries, samples, seq_len)."""
        q, s, l = preds.shape
        flat_inputs = np.repeat(inputs, s, axis=0)
        flat_preds = preds.reshape(q * s, l)
        if self.name == "maze":
            out = np.concatenate([batch_verify_maze(torch.from_numpy(flat_inputs[i:i + 20000].astype(np.int64)),
                                                    torch.from_numpy(flat_preds[i:i + 20000].astype(np.int64))).numpy()
                                  for i in range(0, len(flat_preds), 20000)])
        elif self.name == "nqueens":
            out = np.array([((x != 2) | (p == 2)).all() and verify_nqueens(p) for x, p in zip(flat_inputs, flat_preds)])
        elif self.name == "sudoku":
            out = np.array([verify_sudoku(x, p) for x, p in zip(flat_inputs, flat_preds)])
        elif self.name == "graph":
            # verify_coloring skips nodes without a colour token (< 3), so require one on every node
            out = np.array([(p[:self.answer_len] >= 3).all() and verify_coloring(x, p) for x, p in zip(flat_inputs, flat_preds)])
        else:
            out = sat_utils.batch_verify(flat_inputs, flat_preds, self.answer_len)
        return out.reshape(q, s)

    def valid_up_to_relabel(self, inputs, preds):
        """Graph colouring only: proper colourings whose colours aren't in canonical order (rejected by valid())."""
        canon = self.answer(preds)
        return self.valid(inputs, np.concatenate([canon, preds[..., self.answer_len:]], -1))


def bucket_name(lo, hi):
    return str(lo) if lo == hi else f"{lo}+" if hi > 2**32 else f"{lo}-{hi}"


def load_run(run_dir):
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    paths = [os.path.join(run_dir, f"shard_{i:05d}.npz") for i in range(meta["num_shards"])]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"Skipping {run_dir}: {len(missing)} of {len(paths)} shards missing (sampling unfinished)")
        return None, meta
    shards = sorted((dict(np.load(p)) for p in paths), key=lambda s: int(s["query_start"]))
    return np.concatenate([s["preds"] for s in shards]), meta


def load_split(data_path, split):
    d = os.path.join(data_path, split)
    with open(os.path.join(d, "dataset.json")) as f:
        metadata = json.load(f)
    load = lambda k: np.load(os.path.join(d, f"all__{k}.npy"), mmap_mode="r")
    return metadata, load("inputs"), load("label_set"), np.asarray(load("num_solutions"))


def solution_ids(answers: np.ndarray, valid: np.ndarray):
    """Per query, an id per sample: equal ids = same answer. Returns (queries, samples) int array."""
    ids = np.empty(answers.shape[:2], dtype=np.int64)
    for q in range(len(answers)):
        _, ids[q] = np.unique(answers[q], axis=0, return_inverse=True)
    return ids


def distinct_valid_upto(ids, valid, k):
    """Distinct valid answers and number of valid samples among the first k, per query."""
    distinct = np.array([len(np.unique(i[:k][v[:k]])) for i, v in zip(ids, valid)])
    return distinct, valid[:, :k].sum(1)


def uniform_distinct(n, v):
    """Expected distinct solutions among v uniform draws from n."""
    return n * (1 - (1 - 1 / n) ** v)


def stored_match(answers, stored, stored_ok, chunk=16):
    """For each sample: whether it equals some stored solution, the fewest cells differing from one, and whether
    every cell agrees with at least one stored solution. In chunks of queries, as (q, s, stored, cells) is large."""
    nearest = np.empty(answers.shape[:2], dtype=np.int64)
    union = np.empty(answers.shape[:2], dtype=bool)
    for i in range(0, len(answers), chunk):
        a, st, ok = answers[i:i + chunk], stored[i:i + chunk], stored_ok[i:i + chunk]
        eq = a[:, :, None, :] == st[:, None, :, :]  # (chunk, s, stored, cells)
        nearest[i:i + chunk] = np.where(ok[:, None, :], (~eq).sum(-1), np.iinfo(np.int64).max).min(-1)
        union[i:i + chunk] = (eq & ok[:, None, :, None]).any(2).all(-1)
    return nearest == 0, nearest, union


def analyze_run(task, preds, inputs, label_set, num_solutions, split):
    q, s, _ = preds.shape
    counts = [k for k in SAMPLE_COUNTS if k <= s]
    valid = task.valid(inputs, preds)
    answers = task.answer(preds)
    ids = solution_ids(answers, valid)

    stored = task.answer(np.asarray(label_set))
    stored_ok = (np.asarray(label_set) != 0).any(-1)
    num_stored = stored_ok.sum(1)
    in_stored, nearest, union = stored_match(answers, stored, stored_ok)

    groups = [("all", np.ones(q, dtype=bool))] + [
        (bucket_name(lo, hi), (num_solutions >= lo) & (num_solutions <= hi)) for lo, hi in COUNT_BUCKETS]
    multi = num_solutions > 1
    result = {"num_queries": int(q), "num_samples": int(s), "split": split, "groups": {}}
    for name, mask in groups:
        if not mask.any():
            continue
        g = {"queries": int(mask.sum()), "validity": float(valid[mask].mean()),
             "any_valid": {str(k): float(valid[mask, :k].any(1).mean()) for k in counts}}
        gm = mask & multi
        if gm.any() and s > 1:
            g["diversity"] = {}
            for k in counts[1:]:
                distinct, v = distinct_valid_upto(ids[gm], valid[gm], k)
                n = num_solutions[gm]
                has = v >= 2  # with fewer than 2 valid samples diversity is undefined
                g["diversity"][str(k)] = {
                    "distinct_valid": float(distinct.mean()),
                    "uniform_distinct": float(uniform_distinct(n, v).mean()),
                    "ratio_to_uniform": float((distinct[has] / uniform_distinct(n[has], v[has])).mean()) if has.any() else None,
                    "coverage": float((distinct / n).mean()),
                }
        invalid = ~valid & mask[:, None]
        if invalid.any():
            g["errors"] = {"invalid_samples": int(invalid.sum()),
                           "median_cells_from_nearest_stored": float(np.median(nearest[invalid])),
                           "mean_cells_from_nearest_stored": float(nearest[invalid].mean()),
                           "matches_a_stored_solution": float(in_stored[invalid].mean())}
            invalid_multi = invalid & multi[:, None]
            if invalid_multi.any():
                g["errors"]["blend_of_stored_multi"] = float((union & ~in_stored)[invalid_multi].mean())
        result["groups"][name] = g

    unique = num_solutions == 1
    if unique.any():
        result["unique_inputs"] = {"distinct_answers": float(np.mean([len(np.unique(i)) for i in ids[unique]])),
                                   "validity": float(valid[unique].mean())}
    if task.name == "graph":
        result["graph_valid_up_to_relabel"] = float(task.valid_up_to_relabel(inputs, preds).mean())
    if split == "train":
        held = num_solutions > num_stored
        v = valid & held[:, None]
        if v.any():
            uniform_share = 1 - num_stored[held] / num_solutions[held]
            novel = ~in_stored & v
            result["held_out"] = {
                "queries": int(held.sum()),
                "valid_samples": int(v.sum()),
                "unseen_share_of_valid": float(novel.sum() / v.sum()),
                "uniform_unseen_share": float(np.average(uniform_share, weights=v[held].sum(1)) if v[held].sum() else uniform_share.mean()),
                "queries_with_unseen_valid": float(novel[held].any(1).mean()),
                "distinct_unseen_per_query": float(np.mean([len(np.unique(i[n])) for i, n in zip(ids[held], novel[held])])),
            }
    return result


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def write_summary(results, out_dir):
    lines = ["# Multi-solution sample analysis", ""]
    for run, r in results.items():
        k = str(r["num_samples"])
        lines += [f"## {run} ({r['split']} split, {r['num_queries']} inputs x {r['num_samples']} samples)", "",
                  f"| solutions | inputs | validity | any-valid@{k} | distinct valid@{k} | uniform | ratio | coverage | "
                  f"cells off (median) | blends |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name, g in r["groups"].items():
            d = g.get("diversity", {}).get(k, {})
            e = g.get("errors", {})
            fmt = lambda x, p=1: "-" if x is None else f"{x:.{p}f}"
            lines.append(f"| {name} | {g['queries']} | {pct(g['validity'])} | {pct(g['any_valid'][k])} | "
                         f"{fmt(d.get('distinct_valid'))} | {fmt(d.get('uniform_distinct'))} | {fmt(d.get('ratio_to_uniform'), 2)} | "
                         f"{pct(d.get('coverage'))} | {fmt(e.get('median_cells_from_nearest_stored'), 0)} | "
                         f"{pct(e.get('blend_of_stored_multi'))} |")
        u = r.get("unique_inputs")
        if u:
            lines += ["", f"Unique-solution inputs: {u['distinct_answers']:.2f} distinct answers per input, validity {pct(u['validity'])}%."]
        if "graph_valid_up_to_relabel" in r:
            lines.append(f"Graph colouring: {pct(r['graph_valid_up_to_relabel'])}% of samples are proper colourings up to relabelling.")
        h = r.get("held_out")
        if h:
            lines.append(f"Held-out solutions ({h['queries']} inputs): {pct(h['unseen_share_of_valid'])}% of valid samples are unseen "
                         f"(uniform: {pct(h['uniform_unseen_share'])}%); {pct(h['queries_with_unseen_valid'])}% of inputs get one; "
                         f"{h['distinct_unseen_per_query']:.2f} distinct unseen per input.")
        lines.append("")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--data", required=True, help="The -1oml dataset the samples came from")
    parser.add_argument("--samples", required=True, help="Directory of runs (each with meta.json and shards), or one run")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    run_dirs = [args.samples] if os.path.exists(os.path.join(args.samples, "meta.json")) else \
        sorted(os.path.dirname(p) for p in glob.glob(os.path.join(args.samples, "*", "meta.json")))
    results = {}
    for run_dir in run_dirs:
        preds, meta = load_run(run_dir)
        if preds is None:
            continue
        split = meta.get("split", "test")
        metadata, inputs, label_set, num_solutions = load_split(args.data, split)
        q = len(preds)
        task = Task(args.task, metadata["seq_len"], metadata["vocab_size"])
        run = os.path.basename(os.path.normpath(run_dir))
        print(f"Analyzing {run}: {q} {split} inputs x {preds.shape[1]} samples")
        results[run] = analyze_run(task, preds, np.asarray(inputs[:q]), np.asarray(label_set[:q]), num_solutions[:q], split)
        with open(os.path.join(args.out, f"{run}.json"), "w") as f:
            json.dump(results[run], f, indent=2)
    write_summary(results, args.out)


if __name__ == "__main__":
    main()
