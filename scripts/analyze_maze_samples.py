"""Analyze maze predictions sampled by sample_predictions.py, using exact shortest-path statistics.

Usage: python scripts/analyze_maze_samples.py --data data/maze-15-1oml --samples samples/maze_15 --out results/maze_15

Every run in --samples (a directory per run, holding meta.json and shards) is scored against each test maze's
exact shortest paths (evaluators/maze_utils.py): a sample is valid when it marks any shortest path. Reports:

1. Hidden-constraint test. For each maze with several shortest paths, every cell used by some but not all of
   them defines a hidden requirement "the path goes through here", met by a fraction f of the paths (exact).
   Success with K samples = some valid sample among the first K uses the cell. Requirements are grouped by f;
   each maze counts once per group. References: f, and the best any single path could do (exact, by dynamic
   programming over the shortest paths).
2. Accuracy vs number of samples K: majority vote (no verifier), any valid sample (verifier), and coverage
   (distinct valid paths found / exact number of shortest paths), for all, unique-path and multi-path mazes.
3. Even sampling, for runs with several samples: how far the cells used by a maze's valid samples are from the
   exact share of shortest paths through each cell, against the same measure for a uniform sampler drawing the
   same number of paths. A ratio near 1 means paths are sampled about as evenly as uniformly at random.
4. First-sample accuracy by number of shortest paths and by path length.
5. Variety on unique-path mazes: distinct answers among the samples, and share of valid samples.
6. Compute and self-verification: model passes per sample (1 for one-shot sampling; walk_decode.py records its
   own), and best-of-K picked by the model's halting logit, for runs whose shards saved it.

Writes <out>/<run>.json, <out>/summary.md and figures (if matplotlib is installed).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluators.maze_utils import ShortestPaths, batch_verify_maze, PATH  # noqa: E402

SAMPLE_COUNTS = (1, 2, 3, 5, 10, 20, 50, 100, 250, 500, 1000)
F_EDGES = np.linspace(0, 1, 11)
COUNT_BUCKETS = ((1, 1), (2, 3), (4, 15), (16, 63), (64, 255), (256, 10**9))
LENGTH_BUCKETS = ((0, 11), (12, 15), (16, 23), (24, 10**9))
EVEN_SAMPLING_MIN_VALID = 20


def load_run(run_dir):
    """Samples of one run as (mazes, samples, seq_len), or None if shards are missing; also its meta and the
    per-sample extras every shard saved (`q`: halting logits, `passes`: model passes)."""
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    paths = [os.path.join(run_dir, f"shard_{i:05d}.npz") for i in range(meta["num_shards"])]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"Skipping {run_dir}: {len(missing)} of {len(paths)} shards missing (sampling unfinished)")
        return None, meta, {}
    shards = sorted((np.load(p) for p in paths), key=lambda s: int(s["query_start"]))
    extras = {k: np.concatenate([s[k] for s in shards]) for k in ("q", "passes") if all(k in s for s in shards)}
    return np.concatenate([s["preds"] for s in shards]), meta, extras


def verify_all(inputs, preds, chunk=20000):
    """(mazes, samples) bool: whether each sample marks a shortest path."""
    num_mazes, num_samples, seq_len = preds.shape
    flat_inputs = np.repeat(inputs, num_samples, axis=0)
    flat_preds = preds.reshape(-1, seq_len)
    valid = np.concatenate([
        batch_verify_maze(torch.from_numpy(flat_inputs[i:i + chunk].astype(np.int64)),
                          torch.from_numpy(flat_preds[i:i + chunk].astype(np.int64))).numpy()
        for i in range(0, len(flat_preds), chunk)])
    return valid.reshape(num_mazes, num_samples)


def board_ids(preds):
    """An id per distinct board within each maze, numbered in order of first appearance."""
    ids = np.empty(preds.shape[:2], dtype=np.int64)
    for m in range(preds.shape[0]):
        seen = {}
        for s in range(preds.shape[1]):
            ids[m, s] = seen.setdefault(preds[m, s].tobytes(), len(seen))
    return ids


def majority_valid(ids, valid, k):
    """Whether the most frequent board among the first k samples is valid (ties: earliest seen)."""
    out = np.zeros(len(ids), dtype=bool)
    for m in range(len(ids)):
        counts = np.bincount(ids[m, :k])
        winner = int(np.argmax(counts))  # argmax takes the lowest id, i.e. the earliest seen, on ties
        out[m] = valid[m, :k][ids[m, :k] == winner][0]
    return out


def hidden_constraints(stats, preds, valid, sample_counts):
    num_bins = len(F_EDGES) - 1
    success = {k: np.zeros(num_bins) for k in sample_counts}
    sum_f, best_single, num_mazes = np.zeros(num_bins), np.zeros(num_bins), np.zeros(num_bins)
    num_samples = preds.shape[1]
    for m, sp in enumerate(stats):
        if sp.num_paths < 2:
            continue
        share = sp.share.ravel()
        cells = np.where((share > 0) & (share < 1 - 1e-12))[0]
        if len(cells) == 0:
            continue
        uses = (preds[m][:, cells] == PATH) & valid[m][:, None]  # (samples, cells)
        first_met = np.where(uses.any(0), uses.argmax(0), num_samples)
        bins = np.clip(np.digitize(share[cells], F_EDGES) - 1, 0, num_bins - 1)
        for b in np.unique(bins):
            in_bin = bins == b
            weights = np.zeros(sp.grid.size)
            weights[cells[in_bin]] = 1
            num_mazes[b] += 1
            sum_f[b] += share[cells[in_bin]].mean()
            best_single[b] += sp.max_weight_path(weights.reshape(sp.grid.shape)) / in_bin.sum()
            for k in sample_counts:
                success[k][b] += (first_met[in_bin] < k).mean()
    per_maze = lambda x: np.where(num_mazes > 0, x / np.maximum(num_mazes, 1), np.nan)
    return {
        "f_bins": [[float(F_EDGES[b]), float(F_EDGES[b + 1])] for b in range(num_bins)],
        "mazes": num_mazes.astype(int).tolist(),
        "mean_f": per_maze(sum_f).tolist(),
        "best_single_path": per_maze(best_single).tolist(),
        "success": {str(k): per_maze(success[k]).tolist() for k in sample_counts},
    }


def even_sampling(stats, preds, valid, ids, rng):
    """Cell-usage discrepancy of valid samples vs exact path shares, model vs a uniform sampler (same counts)."""
    model, uniform, distinct_model, distinct_uniform = [], [], [], []
    for m, sp in enumerate(stats):
        n_valid = int(valid[m].sum())
        if sp.num_paths < 2 or n_valid < EVEN_SAMPLING_MIN_VALID:
            continue
        share = sp.share.ravel()
        cells = np.where((share > 0) & (share < 1 - 1e-12))[0]
        usage = (preds[m][valid[m]][:, cells] == PATH).mean(0)
        model.append(np.abs(usage - share[cells]).mean())
        distinct_model.append(len(np.unique(ids[m][valid[m]])))
        drawn = [sp.sample_path(rng) for _ in range(n_valid)]
        reference = np.zeros(sp.grid.size)
        for path in drawn:
            for cell in path:
                reference[np.ravel_multi_index(cell, sp.grid.shape)] += 1 / n_valid
        uniform.append(np.abs(reference[cells] - share[cells]).mean())
        distinct_uniform.append(len({tuple(p) for p in drawn}))
    if not model:
        return None
    return {
        "mazes": len(model),
        "cell_discrepancy_model": float(np.mean(model)),
        "cell_discrepancy_uniform": float(np.mean(uniform)),
        "ratio": float(np.mean(model) / max(np.mean(uniform), 1e-12)),
        "distinct_paths_model": float(np.mean(distinct_model)),
        "distinct_paths_uniform": float(np.mean(distinct_uniform)),
    }


def bucket_name(lo, hi):
    return str(lo) if lo == hi else f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}"


def analyze_run(preds, inputs, stats, rng, extras=None):
    extras = extras or {}
    num_samples = preds.shape[1]
    sample_counts = [k for k in SAMPLE_COUNTS if k <= num_samples]
    valid = verify_all(inputs, preds)
    ids = board_ids(preds)
    num_paths = np.array([sp.num_paths for sp in stats])
    lengths = np.array([sp.length for sp in stats])

    subsets = {"all": np.ones(len(stats), bool), "unique_path": num_paths == 1, "multi_path": num_paths > 1}
    by_samples = {}
    for k in sample_counts:
        distinct = np.array([len(np.unique(ids[m, :k][valid[m, :k]])) for m in range(len(stats))])
        majority = majority_valid(ids, valid, k)
        by_samples[str(k)] = {name: {"majority_vote": float(majority[mask].mean()),
                                     "any_valid": float(valid[mask, :k].any(1).mean()),
                                     "coverage": float((distinct[mask] / num_paths[mask]).mean())}
                              for name, mask in subsets.items() if mask.any()}

    first = valid[:, 0]
    by_paths = {bucket_name(lo, hi): {"mazes": int(((num_paths >= lo) & (num_paths <= hi)).sum()),
                                      "first_sample_accuracy": float(first[(num_paths >= lo) & (num_paths <= hi)].mean())}
                for lo, hi in COUNT_BUCKETS if ((num_paths >= lo) & (num_paths <= hi)).any()}
    by_length = {bucket_name(lo, hi): {"mazes": int(((lengths >= lo) & (lengths <= hi)).sum()),
                                       "first_sample_accuracy": float(first[(lengths >= lo) & (lengths <= hi)].mean())}
                 for lo, hi in LENGTH_BUCKETS if ((lengths >= lo) & (lengths <= hi)).any()}

    best_q = None
    if "q" in extras:
        best_q = {}
        for k in sample_counts:
            pick = extras["q"][:, :k].argmax(1)
            chosen = valid[np.arange(len(stats)), pick]
            best_q[str(k)] = {name: float(chosen[mask].mean()) for name, mask in subsets.items() if mask.any()}

    unique = num_paths == 1
    return {
        "passes_per_sample": float(extras["passes"].mean()) if "passes" in extras else 1.0,
        "best_q": best_q,
        "num_samples": num_samples,
        "num_mazes": len(stats),
        "by_samples": by_samples,
        "hidden_constraints": hidden_constraints(stats, preds, valid, sample_counts),
        "even_sampling": even_sampling(stats, preds, valid, ids, rng) if num_samples > 1 else None,
        "by_num_paths": by_paths,
        "by_path_length": by_length,
        "unique_path_variety": {"distinct_answers": float(np.mean([len(np.unique(r)) for r in ids[unique]])),
                                "valid_sample_share": float(valid[unique].mean())} if unique.any() else None,
    }


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def write_summary(results, out_dir):
    names = sorted(results)
    lines = ["# Maze experiment", ""]

    lines += ["## Hidden-constraint test", "",
              "Success (%) when the path must also pass through an unseen cell used by a fraction f of the shortest "
              "paths. `@K` = accepted sample among the first K, picked by an external checker.", ""]
    hc = {n: results[n]["hidden_constraints"] for n in names}
    first = hc[names[0]]
    columns = list(dict.fromkeys((n, k) for n in names for k in hc[n]["success"]
                                 if k in ("1", "5", "20", str(results[n]["num_samples"]))))
    lines.append("| f | mazes | f (unrelated path) | best single path | " + " | ".join(f"{n} @{k}" for n, k in columns) + " |")
    lines.append("|" + "---|" * (4 + len(columns)))
    for b, (lo, hi) in enumerate(first["f_bins"]):
        if first["mazes"][b]:
            lines.append(f"| {lo:.1f}-{hi:.1f} | {first['mazes'][b]} | {pct(first['mean_f'][b])} | {pct(first['best_single_path'][b])} | "
                         + " | ".join(pct(hc[n]["success"][k][b]) for n, k in columns) + " |")

    lines += ["", "## Accuracy vs number of samples", "",
              "Majority vote needs no verifier; any-valid assumes one picks a valid sample. Coverage = distinct valid "
              "paths found / exact number of shortest paths.", "",
              "| run | samples | majority (all) | majority (unique-path) | majority (multi-path) | any-valid (all) | coverage (multi-path) |",
              "|---|---|---|---|---|---|---|"]
    for n in names:
        for k, m in results[n]["by_samples"].items():
            lines.append(f"| {n} | {k} | {pct(m['all']['majority_vote'])} | {pct(m.get('unique_path', {}).get('majority_vote'))} | "
                         f"{pct(m.get('multi_path', {}).get('majority_vote'))} | {pct(m['all']['any_valid'])} | "
                         f"{pct(m.get('multi_path', {}).get('coverage'))} |")

    lines += ["", "## Even sampling", "",
              f"Mean |cell usage − exact share of shortest paths| over multi-path mazes with ≥{EVEN_SAMPLING_MIN_VALID} valid "
              "samples, against a uniform sampler drawing the same number of paths. Ratio ≈ 1: as even as uniform.", "",
              "| run | mazes | discrepancy | uniform reference | ratio | distinct paths | uniform distinct |", "|---|---|---|---|---|---|---|"]
    for n in names:
        e = results[n]["even_sampling"]
        if e:
            lines.append(f"| {n} | {e['mazes']} | {e['cell_discrepancy_model']:.4f} | {e['cell_discrepancy_uniform']:.4f} | "
                         f"{e['ratio']:.2f} | {e['distinct_paths_model']:.1f} | {e['distinct_paths_uniform']:.1f} |")

    for title, key in (("number of shortest paths", "by_num_paths"), ("path length (steps)", "by_path_length")):
        buckets = list(results[names[0]][key])
        lines += ["", f"## First-sample accuracy by {title}", "",
                  "| run | " + " | ".join(f"{b} (n={results[names[0]][key][b]['mazes']})" for b in buckets) + " |",
                  "|" + "---|" * (1 + len(buckets))]
        for n in names:
            lines.append(f"| {n} | " + " | ".join(pct(results[n][key][b]["first_sample_accuracy"]) for b in buckets) + " |")

    lines += ["", "## Variety on unique-path mazes", "", "| run | samples | distinct answers | valid samples (%) |", "|---|---|---|---|"]
    for n in names:
        v = results[n]["unique_path_variety"]
        if v:
            lines.append(f"| {n} | {results[n]['num_samples']} | {v['distinct_answers']:.2f} | {pct(v['valid_sample_share'])} |")

    counts = [k for k in ("1", "5", "20", "100") if any(k in (results[n]["best_q"] or {}) for n in names)]
    lines += ["", "## Compute and self-verification", "",
              "Model passes per sample (each pass runs the full recursion), and accuracy when the sample with the highest "
              "halting logit among the first K is picked (no external checker).", "",
              "| run | passes per sample | " + " | ".join(f"best-Q @{k}" for k in counts) + " |",
              "|" + "---|" * (2 + len(counts))]
    for n in names:
        bq = results[n].get("best_q") or {}
        lines.append(f"| {n} | {results[n].get('passes_per_sample', 1.0):.2f} | "
                     + " | ".join(pct(bq[k]["all"]) if k in bq else "-" for k in counts) + " |")

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping figures (pip install matplotlib)")
        return
    names = sorted(results)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    hc0 = results[names[0]]["hidden_constraints"]
    keep = [b for b, m in enumerate(hc0["mazes"]) if m > 0]
    x = [np.mean(hc0["f_bins"][b]) for b in keep]
    ax.plot(x, [hc0["mean_f"][b] for b in keep], "k--", label="unrelated single path (f)")
    ax.plot(x, [hc0["best_single_path"][b] for b in keep], "k:", label="best single path")
    for n in names:
        success = results[n]["hidden_constraints"]["success"]
        for k in sorted({"1", max(success, key=int)}, key=int):  # first and largest sample count recorded
            ax.plot(x, [success[k][b] for b in keep], marker="o", label=f"{n} @{k}")
    ax.set_xlabel("f: share of shortest paths through the required cell")
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
    ax.set_xlabel("samples per maze")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_vs_samples.png"), dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Dataset directory; its test split was sampled")
    parser.add_argument("--samples", required=True, help="Directory with one subdirectory per run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0, help="Seed of the uniform reference sampler")
    args = parser.parse_args()

    inputs = np.load(os.path.join(args.data, "test", "all__inputs.npy"))
    size = int(round(np.sqrt(inputs.shape[1])))
    stats = [ShortestPaths(row.reshape(size, size)) for row in inputs]
    os.makedirs(args.out, exist_ok=True)
    results = {}
    for meta_path in sorted(glob.glob(os.path.join(args.samples, "*", "meta.json"))):
        run_dir = os.path.dirname(meta_path)
        preds, meta, extras = load_run(run_dir)
        if preds is None:
            continue
        name = os.path.basename(run_dir)
        print(f"Analyzing {name} ({preds.shape[1]} samples per maze)")
        results[name] = {"meta": meta, **analyze_run(preds, inputs[:len(preds)], stats[:len(preds)],
                                                     np.random.default_rng(args.seed), extras)}
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(results[name], f, indent=2)
    if not results:
        print("No complete runs to analyze")
        return
    write_summary(results, args.out)
    plot(results, args.out)
    print(f"\nSaved {args.out}/summary.md")


if __name__ == "__main__":
    main()
