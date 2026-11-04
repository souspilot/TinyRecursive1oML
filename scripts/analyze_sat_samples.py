"""Analyze 3-SAT samples from sample_predictions.py or sat_commit_decode.py against every solution of each formula.

Usage: python scripts/analyze_sat_samples.py --data data/sat-16-1oml --samples samples/sat_16 --out results/sat_16

The stored label_set holds at most 16 solutions, so every formula's solutions are enumerated again here (exact, 2^n
assignments). Reports, per run (a directory in --samples holding meta.json and shards):

1. Accuracy vs number of samples K: majority vote (no verifier), any valid sample (with a verifier) and coverage
   (share of the formula's solutions found), also against what K uniform draws would cover.
2. First-sample accuracy by number of solutions, by clause density alpha = clauses / variables, and by backbone
   (non-given variables with the same value in every solution: large backbones leave few real choices).
3. Evenness on multi-solution formulas, from the valid samples: the error of each variable's sampled share of true
   values against its exact share (var_shares), and the total-variation distance of the sampled solutions from
   uniform, each next to what an exact uniform sampler with the same number of valid samples gets (sampling noise).
4. Hidden constraints: an unseen requirement "variable i = v", met by a share f of the solutions, binned by f.
5. If the shards have them: halting-head selection (`q`: the sample with the highest final halting logit), model
   passes (`passes`), and for sat_commit_decode.py runs that sample without forcing values, Knuth's solution-count
   estimate (`log_q`: mean of valid / q over samples, against the exact count).

Accuracy tables also hold a `random guess` row, computed exactly: uniform values on the free variables, givens copied.

Writes <out>/<run>.json and <out>/summary.md.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluators import sat_utils  # noqa: E402

SAMPLE_COUNTS = (1, 2, 5, 10, 20, 50, 100, 250, 500, 1000)
COUNT_BUCKETS = ((1, 1), (2, 3), (4, 15), (16, 63), (64, 255), (256, 2**62))
ALPHA_EDGES = (2.0, 2.5, 3.0, 3.5, 4.0, 4.6)
BACKBONE_BUCKETS = ((0, 0), (1, 4), (5, 8), (9, 12), (13, 16))
F_EDGES = np.linspace(0, 1, 11)
MIN_VALID_FOR_EVENNESS = 10


def bucket_name(lo, hi):
    return str(lo) if lo == hi else f"{lo}+" if hi > 2**32 else f"{lo}-{hi}"


def load_test(data_path):
    split = os.path.join(data_path, "test")
    with open(os.path.join(split, "dataset.json")) as f:
        n = sat_utils.num_vars(json.load(f)["vocab_size"])
    load = lambda k: np.load(os.path.join(split, f"all__{k}.npy"))
    test = {k: load(k) for k in ("inputs", "num_solutions", "var_shares", "num_clauses", "num_givens")}
    assignments = sat_utils.Assignments(n)
    solutions = []
    for x in test["inputs"]:
        variables, negated, givens = sat_utils.decode(x, n)
        solutions.append(assignments.solutions(variables, negated, {i: bool(g) for i, g in enumerate(givens) if g >= 0}))
    assert (np.array([len(s) for s in solutions]) == test["num_solutions"]).all(), "enumeration disagrees with the dataset"
    test["given"] = test["inputs"][:, :n] < 3 + 2 * n
    shares = test["var_shares"]
    test["backbone"] = (((shares == 0) | (shares == 1)) & ~test["given"]).sum(1)
    return n, test, solutions, assignments


def load_run(run_dir):
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    paths = [os.path.join(run_dir, f"shard_{i:05d}.npz") for i in range(meta["num_shards"])]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"Skipping {run_dir}: {len(missing)} of {len(paths)} shards missing (sampling unfinished)")
        return None, meta
    shards = sorted((dict(np.load(p)) for p in paths), key=lambda s: int(s["query_start"]))
    fields = {k: np.concatenate([s[k] for s in shards]) for k in shards[0] if k != "query_start"}
    return fields, meta


def match(preds, n, solutions):
    """Assignment index per sample (-1 if some variable isn't TRUE / FALSE) and the solution it is (-1 if none)."""
    values = preds[..., :n]
    assigned = np.isin(values, (sat_utils.FALSE, sat_utils.TRUE)).all(-1)
    index = np.where(assigned, ((values == sat_utils.TRUE).astype(np.int64) << np.arange(n)).sum(-1), -1)
    solution = np.full(index.shape, -1, dtype=np.int64)
    for q, sols in enumerate(solutions):
        pos = np.searchsorted(sols, index[q])
        hit = (pos < len(sols)) & (sols[np.minimum(pos, len(sols) - 1)] == index[q])
        solution[q] = np.where(hit, pos, -1)
    return index, solution


def majority_correct(index, solution, k_samples):
    """Whether the most frequent assignment among the first k_samples is a solution (ties: earliest seen)."""
    correct = np.zeros(len(index), dtype=bool)
    for q in range(len(index)):
        values, first, counts = np.unique(index[q, :k_samples], return_index=True, return_counts=True)
        correct[q] = solution[q, first[np.lexsort((first, -counts))[0]]] >= 0
    return correct


def evenness(solution, test, solutions, assignments, rng):
    """Marginal error and TV from uniform on multi-solution formulas, for the samples and for exact uniform draws."""
    out = {"queries": 0, "marginal_error": 0.0, "marginal_error_uniform": 0.0, "tv": 0.0, "tv_uniform": 0.0}
    for q in np.where(test["num_solutions"] > 1)[0]:
        found = solution[q][solution[q] >= 0]
        if len(found) < MIN_VALID_FOR_EVENNESS:
            continue
        k = len(solutions[q])
        free = ~test["given"][q]
        values = assignments.values(solutions[q])
        uniform = rng.integers(k, size=len(found))
        marginal = lambda picks: np.abs(values[picks].mean(0) - test["var_shares"][q])[free].mean()
        tv = lambda picks: 0.5 * (np.abs(np.bincount(picks, minlength=k) / len(picks) - 1 / k).sum())
        out["queries"] += 1
        out["marginal_error"] += marginal(found)
        out["marginal_error_uniform"] += marginal(uniform)
        out["tv"] += tv(found)
        out["tv_uniform"] += tv(uniform)
    if out["queries"]:
        for key in ("marginal_error", "marginal_error_uniform", "tv", "tv_uniform"):
            out[key] /= out["queries"]
    return out


def hidden_constraints(solution, test, solutions, assignments, sample_counts):
    """Success per requirement-difficulty group: some sample among the first K is a solution with variable i = v."""
    num_bins = len(F_EDGES) - 1
    success = {k: np.zeros(num_bins) for k in sample_counts}
    sum_f, best_single, num_queries = np.zeros(num_bins), np.zeros(num_bins), np.zeros(num_bins)
    num_samples = solution.shape[1]
    for q in np.where(test["num_solutions"] > 1)[0]:
        values = assignments.values(solutions[q])  # (k, n)
        share = test["var_shares"][q]
        open_vars = np.where(~test["given"][q] & (share > 0) & (share < 1))[0]
        if len(open_vars) == 0:
            continue
        # Requirements: (variable, value) for both values of every variable that isn't fixed by the formula
        req_var = np.repeat(open_vars, 2)
        req_val = np.tile([True, False], len(open_vars))
        f = np.where(req_val, share[req_var], 1 - share[req_var])
        meets = values[:, req_var] == req_val  # (k, requirements)
        first_found = np.full(len(solutions[q]), num_samples)
        for s, j in enumerate(solution[q]):
            if j >= 0 and first_found[j] == num_samples:
                first_found[j] = s
        first_met = np.where(meets, first_found[:, None], num_samples).min(0)
        bins = np.clip(np.digitize(f, F_EDGES) - 1, 0, num_bins - 1)
        for b in np.unique(bins):
            in_bin = bins == b
            num_queries[b] += 1
            sum_f[b] += f[in_bin].mean()
            best_single[b] += meets[:, in_bin].mean(1).max()
            for k in sample_counts:
                success[k][b] += (first_met[in_bin] < k).mean()
    per_query = lambda x: np.where(num_queries > 0, x / np.maximum(num_queries, 1), np.nan)
    return {
        "f_bins": [[float(F_EDGES[b]), float(F_EDGES[b + 1])] for b in range(num_bins)],
        "queries": num_queries.astype(int).tolist(),
        "mean_f": per_query(sum_f).tolist(),
        "best_single_answer": per_query(best_single).tolist(),
        "success": {str(k): per_query(success[k]).tolist() for k in sample_counts},
    }


def formula_groups(test, n):
    num_solutions = test["num_solutions"]
    alpha = test["num_clauses"] / n
    return {
        "by_solutions": [(bucket_name(lo, hi), (num_solutions >= lo) & (num_solutions <= hi)) for lo, hi in COUNT_BUCKETS],
        "by_alpha": [(f"{lo:.1f}-{hi:.1f}", (alpha >= lo) & (alpha < hi)) for lo, hi in zip(ALPHA_EDGES[:-1], ALPHA_EDGES[1:])],
        "by_backbone": [(bucket_name(lo, hi), (test["backbone"] >= lo) & (test["backbone"] <= hi)) for lo, hi in BACKBONE_BUCKETS],
    }


def random_guess(n, test):
    """Exact results of guessing the free variables uniformly and copying the givens."""
    num_solutions = test["num_solutions"]
    assignments_left = 2.0 ** (n - test["num_givens"])
    hit = num_solutions / assignments_left
    multi = num_solutions > 1
    by_samples = {str(k): {"any_valid": float((1 - (1 - hit) ** k).mean()),
                           "coverage_multi_solution": float((1 - (1 - 1 / assignments_left[multi]) ** k).mean())}
                  for k in SAMPLE_COUNTS}
    first_sample = {g: {name: float(hit[mask].mean()) if mask.any() else None for name, mask in buckets}
                    for g, buckets in formula_groups(test, n).items()}
    return {"by_samples": by_samples, "first_sample": first_sample}


def analyze_run(fields, meta, n, test, solutions, assignments, rng):
    preds = fields["preds"]
    num_queries, num_samples = preds.shape[:2]
    sample_counts = [k for k in SAMPLE_COUNTS if k <= num_samples]
    index, solution = match(preds, n, solutions)
    valid = solution >= 0
    num_solutions = test["num_solutions"]

    # Cross-check membership in the enumerated solutions against the clause checker on the first samples
    check = min(num_queries, 500)
    rule_valid = np.array([[sat_utils.verify(test["inputs"][q], preds[q, s], n) for s in range(min(num_samples, 4))]
                           for q in range(check)])
    disagreements = int((rule_valid != valid[:check, :rule_valid.shape[1]]).sum())

    subsets = {"all": np.ones(num_queries, bool), "one_solution": num_solutions == 1, "multi_solution": num_solutions > 1}
    by_samples = {}
    for k in sample_counts:
        distinct = np.array([len(np.unique(row[row >= 0])) for row in solution[:, :k]])
        uniform_distinct = num_solutions * (1 - (1 - 1 / num_solutions) ** k)
        majority = majority_correct(index, solution, k)
        by_samples[str(k)] = {
            name: {
                "majority_vote": float(majority[mask].mean()),
                "any_valid": float(valid[mask, :k].any(1).mean()),
                "coverage": float((distinct[mask] / num_solutions[mask]).mean()),
                "coverage_uniform_sampler": float((uniform_distinct[mask] / num_solutions[mask]).mean()),
            }
            for name, mask in subsets.items()
        }

    first = valid[:, 0]
    groups = formula_groups(test, n)
    first_sample = {g: {name: {"queries": int(mask.sum()), "accuracy": float(first[mask].mean()) if mask.any() else None,
                               "valid_samples": float(valid[mask].mean()) if mask.any() else None}
                        for name, mask in buckets}
                    for g, buckets in groups.items()}

    result = {
        "num_samples": num_samples,
        "num_queries": num_queries,
        "validity_check_disagreements": disagreements,
        "by_samples": by_samples,
        "first_sample": first_sample,
        "evenness": evenness(solution, test, solutions, assignments, rng),
        "hidden_constraints": hidden_constraints(solution, test, solutions, assignments, sample_counts),
    }
    if "q" in fields and num_samples > 1:
        best = fields["q"].argmax(1)
        result["halting_head_selection"] = {name: float(valid[np.arange(num_queries), best][mask].mean())
                                            for name, mask in subsets.items()}
    if "passes" in fields:
        result["model_passes"] = float(fields["passes"].mean())
    # Greedy or forced commits don't record the probability of the sample, so only sampled commits estimate counts
    if "log_q" in fields and meta.get("choice") == "sample" and meta.get("certain", 0) > 1:
        estimate = (valid * np.exp(-fields["log_q"])).mean(1)
        found = estimate > 0  # With no valid sample the estimate is 0, which says nothing about its accuracy
        log2_ratio = np.log2(np.where(found, estimate, 1.0) / num_solutions)
        result["count_estimate"] = {
            name: {"queries": int(mask.sum()),
                   "no_valid_sample": float((~found[mask]).mean()),
                   "within_2x": float((found[mask] & (np.abs(log2_ratio[mask]) <= 1)).mean()),
                   "median_log2_ratio": float(np.median(log2_ratio[mask & found])) if (mask & found).any() else None}
            for name, mask in groups["by_solutions"] if mask.any()
        }
    return result


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def write_summary(results, guess, out_dir):
    names = sorted(results)
    lines = ["# 3-SAT samples", ""]

    lines += ["## Accuracy vs number of samples", "",
              "Majority vote needs no verifier; any-valid assumes one picks a valid sample. Coverage = share of a formula's "
              "solutions found; `uniform` = what as many exact uniform draws would cover.", ""]
    lines.append("| run | samples | majority (all) | majority (one-sol) | majority (multi-sol) | any-valid (all) | "
                 "coverage (multi-sol) | uniform |")
    lines.append("|---|---|---|---|---|---|---|---|")
    max_samples = max(results[n]["num_samples"] for n in names)
    for k, m in guess["by_samples"].items():
        if int(k) <= max_samples:
            lines.append(f"| random guess | {k} | n/a | n/a | n/a | {pct(m['any_valid'])} | {pct(m['coverage_multi_solution'])} | n/a |")
    for n in names:
        for k, m in results[n]["by_samples"].items():
            lines.append(f"| {n} | {k} | {pct(m['all']['majority_vote'])} | {pct(m['one_solution']['majority_vote'])} | "
                         f"{pct(m['multi_solution']['majority_vote'])} | {pct(m['all']['any_valid'])} | "
                         f"{pct(m['multi_solution']['coverage'])} | {pct(m['multi_solution']['coverage_uniform_sampler'])} |")

    titles = {"by_solutions": "number of solutions", "by_alpha": "clause density (clauses / variables)",
              "by_backbone": "backbone (non-given variables fixed in every solution)"}
    for g, title in titles.items():
        buckets = results[names[0]]["first_sample"][g]
        keep = [b for b in buckets if buckets[b]["queries"]]
        lines += ["", f"## First-sample accuracy by {title}", ""]
        lines.append("| run | " + " | ".join(f"{b} (n={buckets[b]['queries']})" for b in keep) + " |")
        lines.append("|" + "---|" * (1 + len(keep)))
        lines.append("| random guess | " + " | ".join(f"{100 * guess['first_sample'][g][b]:.2f}" for b in keep) + " |")
        for n in names:
            lines.append(f"| {n} | " + " | ".join(pct(results[n]["first_sample"][g][b]["accuracy"]) for b in keep) + " |")

    lines += ["", "## Evenness on multi-solution formulas", "",
              f"From the valid samples of formulas with at least {MIN_VALID_FOR_EVENNESS}. Each value is next to an exact "
              "uniform sampler with the same number of valid samples, whose error is pure sampling noise.", "",
              "| run | formulas | marginal error | uniform | TV from uniform | uniform |", "|---|---|---|---|---|---|"]
    for n in names:
        e = results[n]["evenness"]
        if e["queries"]:
            lines.append(f"| {n} | {e['queries']} | {e['marginal_error']:.3f} | {e['marginal_error_uniform']:.3f} | "
                         f"{e['tv']:.3f} | {e['tv_uniform']:.3f} |")

    lines += ["", "## Hidden constraints", "",
              "Success (%) when an unseen requirement `variable = value`, met by a share f of the solutions, must also hold. "
              "`@K` = some valid sample among the first K meets it.", ""]
    hc = {n: results[n]["hidden_constraints"] for n in names}
    first = hc[names[0]]
    columns = [(n, k) for n in names for k in hc[n]["success"] if k in ("1", "10", str(results[n]["num_samples"]))]
    lines.append("| f | formulas | f (unrelated answer) | best single answer | " + " | ".join(f"{n} @{k}" for n, k in columns) + " |")
    lines.append("|" + "---|" * (4 + len(columns)))
    for b, (lo, hi) in enumerate(first["f_bins"]):
        if first["queries"][b]:
            lines.append(f"| {lo:.1f}-{hi:.1f} | {first['queries'][b]} | {pct(first['mean_f'][b])} | "
                         f"{pct(first['best_single_answer'][b])} | " + " | ".join(pct(hc[n]["success"][k][b]) for n, k in columns) + " |")

    selection = [n for n in names if "halting_head_selection" in results[n]]
    if selection:
        lines += ["", "## Halting-head selection", "", "Accuracy of the sample with the highest halting logit.", "",
                  "| run | samples | all | one-sol | multi-sol |", "|---|---|---|---|---|"]
        for n in selection:
            s = results[n]["halting_head_selection"]
            lines.append(f"| {n} | {results[n]['num_samples']} | {pct(s['all'])} | {pct(s['one_solution'])} | {pct(s['multi_solution'])} |")

    decoded = [n for n in names if "model_passes" in results[n]]
    if decoded:
        lines += ["", "## Model passes per sample", "", "| run | passes |", "|---|---|"]
        lines += [f"| {n} | {results[n]['model_passes']:.1f} |" for n in decoded]

    counting = [n for n in names if "count_estimate" in results[n]]
    if counting:
        lines += ["", "## Solution counts from commit sampling (Knuth's estimator)", "",
                  "By exact count: % of formulas with no valid sample (estimate 0), % estimated within a factor of 2, and the "
                  "median log2(estimate / exact count) among formulas with a valid sample.", ""]
        buckets = list(results[counting[0]]["count_estimate"])
        lines.append("| run | " + " | ".join(f"{b}: none / within 2x / log2 ratio" for b in buckets) + " |")
        lines.append("|" + "---|" * (1 + len(buckets)))
        ratio = lambda x: "n/a" if x is None else f"{x:+.2f}"
        for n in counting:
            c = results[n]["count_estimate"]
            lines.append(f"| {n} | " + " | ".join(
                f"{pct(c[b]['no_valid_sample'])} / {pct(c[b]['within_2x'])} / {ratio(c[b]['median_log2_ratio'])}" for b in buckets) + " |")

    disagreements = {n: results[n]["validity_check_disagreements"] for n in names if results[n]["validity_check_disagreements"]}
    lines += ["", f"Validity cross-check: {'all samples agree' if not disagreements else f'DISAGREEMENTS {disagreements}'} "
                  "(clause checker vs membership in the enumerated solutions)."]

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="3-SAT dataset directory (its test split was sampled)")
    parser.add_argument("--samples", required=True, help="Directory with one subdirectory per run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0, help="For the uniform-sampler references")
    args = parser.parse_args()

    n, test, solutions, assignments = load_test(args.data)
    os.makedirs(args.out, exist_ok=True)
    results = {}
    for meta_path in sorted(glob.glob(os.path.join(args.samples, "*", "meta.json"))):
        run_dir = os.path.dirname(meta_path)
        fields, meta = load_run(run_dir)
        if fields is None:
            continue
        name = os.path.basename(run_dir)
        print(f"Analyzing {name} ({fields['preds'].shape[1]} samples per formula)")
        sub = {k: v[:len(fields["preds"])] if isinstance(v, np.ndarray) else v for k, v in test.items()}
        results[name] = {"meta": meta, **analyze_run(fields, meta, n, sub, solutions[:len(fields["preds"])], assignments,
                                                     np.random.default_rng(args.seed))}
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(results[name], f, indent=2)
    if not results:
        print("No complete runs to analyze")
        return
    write_summary(results, random_guess(n, test), args.out)
    print(f"\nSaved {args.out}/summary.md")


if __name__ == "__main__":
    main()
