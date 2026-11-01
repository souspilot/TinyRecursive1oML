"""
Random 3-SAT with Many Solutions

Each example is a random 3-SAT formula over --num-vars variables with alpha * n clauses (alpha drawn uniformly from
[--alpha-min, --alpha-max]), optionally with some variables given, encoded as one token sequence (see
evaluators/sat_utils.py). The label is one satisfying assignment. Every satisfying assignment is enumerated
exactly, so formulas are drawn to a target distribution of solution counts: --unique-share with a unique solution,
the rest spread evenly over the groups 2-3, 4-15, 16-63, 64-255 and 256+. Unsatisfiable formulas are dropped.

The clause density alpha is the ambiguity dial: few clauses leave thousands of solutions, and near the
satisfiability threshold (alpha ~ 4.27 for large n) a handful. Giving variables (from a random solution) also
removes solutions, as given queens do in N-Queens.

Writes four datasets sharing the same test split (one row per formula, with label_set holding up to
--max-stored-solutions solutions drawn uniformly, num_solutions the exact count, and per formula var_shares (the
share of solutions setting each variable true), num_clauses and num_givens):
  <out>          train: every stored solution, grouped by formula, so each epoch sees one random stored solution
  <out>-1oml     train: one row per formula with label_set of its stored solutions (for 1oML objectives)
  <out>-single   train: one fixed random solution per formula
  <out>-unique   train: only formulas with a unique solution

Usage:
    python dataset/build_sat_dataset.py --num-vars 16 --num-train 100000 --num-test 2000 --output-dir data/sat-16
"""
import json
import math
import os
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.common import PuzzleDatasetMetadata  # noqa: E402
from evaluators import sat_utils  # noqa: E402

cli = ArgParser()

COUNT_BOUNDS = (1, 3, 15, 63, 255, 2 ** 62)  # inclusive upper bounds: 1, 2-3, 4-15, 16-63, 64-255, 256+


class DataProcessConfig(BaseModel):
    num_vars: int = 16
    output_dir: str = "data/sat-16"
    num_train: int = 100000
    num_test: int = 2000
    alpha_min: float = 2.0
    alpha_max: float = 4.5
    max_given_share: float = 0.25  # each formula gives a uniform number of variables in [0, share * n]
    unique_share: float = 1 / 3
    max_stored_solutions: int = 16
    seed: int = 42


def count_group(num_solutions: int) -> int:
    return next(i for i, bound in enumerate(COUNT_BOUNDS) if num_solutions <= bound)


def random_formula(rng: np.random.Generator, n: int, m: int):
    """m clauses of 3 distinct variables with random signs."""
    variables = np.argsort(rng.random((m, n)), axis=1)[:, :3]
    return variables, rng.random((m, 3)) < 0.5


def candidate(rng: np.random.Generator, assignments: sat_utils.Assignments, config: DataProcessConfig):
    """A formula with its givens and solutions, or None if it is unsatisfiable."""
    n = config.num_vars
    m = int(round(rng.uniform(config.alpha_min, config.alpha_max) * n))
    variables, negated = random_formula(rng, n, m)
    solutions = assignments.solutions(variables, negated)
    if len(solutions) == 0:
        return None
    num_givens = int(rng.integers(int(config.max_given_share * n) + 1))
    givens = {}
    if num_givens:
        source = assignments.values(solutions[rng.integers(len(solutions))][None])[0]
        givens = {int(i): bool(source[i]) for i in rng.choice(n, num_givens, replace=False)}
        solutions = assignments.solutions(variables, negated, givens)
    return variables, negated, givens, solutions


def generate_chunk(args):
    seed, config, num_candidates, needed = args
    rng = np.random.default_rng(seed)
    assignments = sat_utils.Assignments(config.num_vars)
    n, max_clauses = config.num_vars, max_clauses_for(config)
    out = []
    for _ in range(num_candidates):
        drawn = candidate(rng, assignments, config)
        if drawn is None or not needed[count_group(len(drawn[3]))]:
            continue
        variables, negated, givens, solutions = drawn
        values = assignments.values(solutions)
        stored = values[rng.permutation(len(values))[:config.max_stored_solutions]]
        out.append(dict(
            inputs=sat_utils.encode(variables, negated, givens, n, max_clauses),
            labels=[sat_utils.label(a, n, max_clauses) for a in stored],
            num_solutions=len(solutions),
            var_shares=values.mean(0).astype(np.float32),
            num_clauses=len(variables),
            num_givens=len(givens),
        ))
    return out


def max_clauses_for(config: DataProcessConfig) -> int:
    return math.ceil(config.alpha_max * config.num_vars)


def quotas_for(num: int, share_unique: float, num_groups: int):
    unique = round(num * share_unique)
    rest = num - unique
    return [unique] + [rest // (num_groups - 1) + (1 if i < rest % (num_groups - 1) else 0) for i in range(num_groups - 1)]


def draw_formulas(config: DataProcessConfig, quotas, rng, seen_inputs):
    """Formulas until every solution-count group has its quota, generated in parallel but deterministically."""
    filled = [[] for _ in quotas]
    with Pool(cpu_count()) as pool, tqdm(total=sum(quotas), desc="Drawing formulas") as bar:
        # Rounds of chunks with seeds drawn up front, so the result doesn't depend on the number of processes
        while not all(len(f) >= q for f, q in zip(filled, quotas)):
            needed = tuple(len(f) < q for f, q in zip(filled, quotas))
            tasks = [(int(rng.integers(2**63)), config, 1000, needed) for _ in range(64)]
            for results in pool.imap(generate_chunk, tasks):
                for example in results:
                    g = count_group(example["num_solutions"])
                    key = example["inputs"].tobytes()
                    if len(filled[g]) < quotas[g] and key not in seen_inputs:
                        seen_inputs.add(key)
                        filled[g].append(example)
                        bar.update(1)
    formulas = [f for group in filled for f in group]
    rng.shuffle(formulas)
    return formulas


def save_split(save_dir: str, rows, config: DataProcessConfig, groups=None, with_label_set=True):
    """rows: list of (example, label sequence). groups: rows per group."""
    os.makedirs(save_dir, exist_ok=True)
    n = len(rows)
    seq_len = config.num_vars + 3 * max_clauses_for(config)
    group_sizes = groups if groups is not None else [1] * n
    data = {
        "inputs": np.stack([e["inputs"] for e, _ in rows]),
        "labels": np.stack([lbl for _, lbl in rows]),
        "num_solutions": np.array([e["num_solutions"] for e, _ in rows], dtype=np.int32),
        "var_shares": np.stack([e["var_shares"] for e, _ in rows]),
        "num_clauses": np.array([e["num_clauses"] for e, _ in rows], dtype=np.int32),
        "num_givens": np.array([e["num_givens"] for e, _ in rows], dtype=np.int32),
        "puzzle_identifiers": np.zeros(n, dtype=np.int32),
        "puzzle_indices": np.arange(n + 1, dtype=np.int32),
        "group_indices": np.concatenate([[0], np.cumsum(group_sizes)]).astype(np.int32),
    }
    if with_label_set:
        label_set = np.zeros((n, config.max_stored_solutions, seq_len), dtype=np.uint8)  # 0 = ignore id, so padding is skipped
        for i, (e, _) in enumerate(rows):
            label_set[i, :len(e["labels"])] = np.stack(e["labels"])
        data["label_set"] = label_set
    for k, v in data.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    metadata = PuzzleDatasetMetadata(
        seq_len=seq_len, vocab_size=3 + 3 * config.num_vars, pad_id=0, ignore_label_id=0, blank_identifier_id=0,
        num_puzzle_identifiers=1, total_groups=len(group_sizes), mean_puzzle_examples=1, total_puzzles=n, sets=["all"])
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)


def describe(name, formulas):
    counts = np.array([f["num_solutions"] for f in formulas])
    lows = [1] + [b + 1 for b in COUNT_BOUNDS[:-1]]
    groups = ", ".join(f"{lo}: {np.sum(counts == lo)}" if lo == hi else
                       f"{lo}+: {np.sum(counts >= lo)}" if hi > 2 ** 32 else f"{lo}-{hi}: {np.sum((counts >= lo) & (counts <= hi))}"
                       for lo, hi in zip(lows, COUNT_BOUNDS))
    clauses = np.array([f["num_clauses"] for f in formulas])
    givens = np.array([f["num_givens"] for f in formulas])
    print(f"{name}: {len(formulas)} formulas; solutions per formula: {groups} (max {counts.max()}); clauses "
          f"{clauses.min()}-{clauses.max()} (mean {clauses.mean():.1f}); givens mean {givens.mean():.1f}; "
          f"stored solutions per formula: mean {np.mean([len(f['labels']) for f in formulas]):.1f}")


@cli.command(singleton=True)
def build_sat_dataset(config: DataProcessConfig):
    rng = np.random.default_rng(config.seed)
    seen = set()

    test = draw_formulas(config, quotas_for(config.num_test, config.unique_share, len(COUNT_BOUNDS)), rng, seen)
    train = draw_formulas(config, quotas_for(config.num_train, config.unique_share, len(COUNT_BOUNDS)), rng, seen)
    describe("test", test)
    describe("train", train)

    # Test: one row per formula, labelled with its first stored solution, identical in every dataset
    test_rows = [(f, f["labels"][0]) for f in test]
    pick = lambda f: f["labels"][rng.integers(len(f["labels"]))]
    variants = {
        "": dict(rows=[(f, lbl) for f in train for lbl in f["labels"]],
                 groups=[len(f["labels"]) for f in train], with_label_set=False),
        "-1oml": dict(rows=[(f, pick(f)) for f in train]),
        "-single": dict(rows=[(f, pick(f)) for f in train], with_label_set=False),
        "-unique": dict(rows=[(f, f["labels"][0]) for f in train if f["num_solutions"] == 1]),
    }
    for suffix, spec in variants.items():
        out = config.output_dir + suffix
        save_split(os.path.join(out, "train"), spec["rows"], config,
                   groups=spec.get("groups"), with_label_set=spec.get("with_label_set", True))
        save_split(os.path.join(out, "test"), test_rows, config)
        with open(os.path.join(out, "identifiers.json"), "w") as f:
            json.dump(["<blank>"], f)
        print(f"Saved {out}: {len(spec['rows'])} train examples in {len(spec.get('groups') or spec['rows'])} groups")


if __name__ == "__main__":
    cli()
