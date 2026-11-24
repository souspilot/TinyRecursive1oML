"""Shared pieces for multi-solution dataset builders in the layout of build_maze_dataset.py.

A builder supplies generate_chunk(args) -> list of examples, where args = (seed, task_args, num_candidates, needed)
and each example is a dict with:
  inputs         (seq_len,) uint8 input sequence
  labels         list of (seq_len,) uint8 label sequences: up to max_stored solutions, in random order
  num_solutions  exact number of solutions
  any other scalar fields, saved per row as all__<field>.npy (int32)
`needed[g]` says whether solution-count group g still needs examples, so slow work can be skipped for full groups.

Inputs are drawn until every group of solution counts has its quota (--unique-share with a unique solution, the
rest spread evenly over the other groups), deterministically for a given seed whatever the number of processes.
write_variants then writes four datasets sharing the same test split (one row per input, label_set holding the
stored solutions, num_solutions the exact count):
  <out>          train: every stored solution, grouped by input, so each epoch sees one random stored solution
  <out>-1oml     train: one row per input with label_set of its stored solutions (for 1oML objectives)
  <out>-single   train: one fixed random solution per input
  <out>-unique   train: only inputs with a unique solution
Training only ever sees the stored solutions, so on inputs with more solutions than max_stored the others are held
out: sampling on training inputs tests whether a model produces solutions it never saw (label_set in <out>-1oml/train
lists the stored ones).
"""
import json
import os
from multiprocessing import Pool, cpu_count
from typing import Callable, List, Sequence

import numpy as np
from tqdm import tqdm

from dataset.common import PuzzleDatasetMetadata


def count_group(num_solutions: int, bounds: Sequence[int]) -> int:
    """Index of the first group whose inclusive upper bound holds num_solutions."""
    return next(i for i, bound in enumerate(bounds) if num_solutions <= bound)


def quotas_for(num: int, unique_share: float, num_groups: int) -> List[int]:
    unique = round(num * unique_share)
    rest = num - unique
    return [unique] + [rest // (num_groups - 1) + (1 if i < rest % (num_groups - 1) else 0) for i in range(num_groups - 1)]


def draw(generate_chunk: Callable, task_args, quotas, bounds, rng: np.random.Generator, seen_inputs: set,
         chunk: int = 200, desc: str = "Drawing inputs"):
    """Examples until every solution-count group has its quota, generated in parallel but deterministically."""
    filled = [[] for _ in quotas]
    stalled_rounds = 0
    with Pool(cpu_count()) as pool, tqdm(total=sum(quotas), desc=desc) as bar:
        # Rounds of chunks with seeds drawn up front, so the result doesn't depend on the number of processes
        while not all(len(f) >= q for f, q in zip(filled, quotas)):
            needed = tuple(len(f) < q for f, q in zip(filled, quotas))
            tasks = [(int(rng.integers(2**63)), task_args, chunk, needed) for _ in range(64)]
            added = 0
            for results in pool.imap(generate_chunk, tasks):
                for example in results:
                    g = count_group(example["num_solutions"], bounds)
                    key = example["inputs"].tobytes()
                    if len(filled[g]) < quotas[g] and key not in seen_inputs:
                        seen_inputs.add(key)
                        filled[g].append(example)
                        bar.update(1)
                        added += 1
            # Fail instead of hanging when a group has run out of distinct inputs
            stalled_rounds = 0 if added else stalled_rounds + 1
            if stalled_rounds >= 20:
                lows = [1] + [b + 1 for b in bounds[:-1]]
                short = ", ".join(f"{lo}-{hi} solutions: {len(f)}/{q}" for lo, hi, f, q in zip(lows, bounds, filled, quotas)
                                  if len(f) < q)
                raise RuntimeError(f"{desc}: no new inputs in 20 rounds of {64 * chunk} candidates; groups short of "
                                   f"their quota ({short}). Lower the quota or merge these groups.")
    examples = [e for group in filled for e in group]
    rng.shuffle(examples)
    return examples


def save_split(save_dir: str, rows, seq_len: int, vocab_size: int, max_stored: int, extra_fields: Sequence[str] = (),
               groups=None, with_label_set: bool = True):
    """rows: list of (example, label sequence). groups: rows per group (default one each)."""
    os.makedirs(save_dir, exist_ok=True)
    n = len(rows)
    group_sizes = groups if groups is not None else [1] * n
    data = {
        "inputs": np.stack([e["inputs"] for e, _ in rows]).astype(np.uint8),
        "labels": np.stack([lbl for _, lbl in rows]).astype(np.uint8),
        "num_solutions": np.array([e["num_solutions"] for e, _ in rows], dtype=np.int32),
        "puzzle_identifiers": np.zeros(n, dtype=np.int32),
        "puzzle_indices": np.arange(n + 1, dtype=np.int32),
        "group_indices": np.concatenate([[0], np.cumsum(group_sizes)]).astype(np.int32),
    }
    for field in extra_fields:
        data[field] = np.array([e[field] for e, _ in rows], dtype=np.int32)
    if with_label_set:
        label_set = np.zeros((n, max_stored, seq_len), dtype=np.uint8)  # 0 = ignore id, so padding is skipped
        for i, (e, _) in enumerate(rows):
            label_set[i, :len(e["labels"])] = np.stack(e["labels"])
        data["label_set"] = label_set
    for k, v in data.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    metadata = PuzzleDatasetMetadata(
        seq_len=seq_len, vocab_size=vocab_size, pad_id=0, ignore_label_id=0, blank_identifier_id=0,
        num_puzzle_identifiers=1, total_groups=len(group_sizes), mean_puzzle_examples=1, total_puzzles=n, sets=["all"])
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)


def write_variants(output_dir: str, train, test, rng: np.random.Generator, seq_len: int, vocab_size: int,
                   max_stored: int, extra_fields: Sequence[str] = ()):
    # Test: one row per input, labelled with its first stored solution, identical in every dataset
    test_rows = [(e, e["labels"][0]) for e in test]
    pick = lambda e: e["labels"][rng.integers(len(e["labels"]))]
    variants = {
        "": dict(rows=[(e, lbl) for e in train for lbl in e["labels"]],
                 groups=[len(e["labels"]) for e in train], with_label_set=False),
        "-1oml": dict(rows=[(e, pick(e)) for e in train]),
        "-single": dict(rows=[(e, pick(e)) for e in train], with_label_set=False),
        "-unique": dict(rows=[(e, e["labels"][0]) for e in train if e["num_solutions"] == 1]),
    }
    for suffix, spec in variants.items():
        out = output_dir + suffix
        save_split(os.path.join(out, "train"), spec["rows"], seq_len, vocab_size, max_stored, extra_fields,
                   groups=spec.get("groups"), with_label_set=spec.get("with_label_set", True))
        save_split(os.path.join(out, "test"), test_rows, seq_len, vocab_size, max_stored, extra_fields)
        with open(os.path.join(out, "identifiers.json"), "w") as f:
            json.dump(["<blank>"], f)
        print(f"Saved {out}: {len(spec['rows'])} train examples in {len(spec.get('groups') or spec['rows'])} groups")


def describe(name: str, examples, bounds, extra_fields: Sequence[str] = ()):
    counts = np.array([e["num_solutions"] for e in examples])
    lows = [1] + [b + 1 for b in bounds[:-1]]
    groups = ", ".join(f"{lo}: {np.sum(counts == lo)}" if lo == hi else f"{lo}-{hi}: {np.sum((counts >= lo) & (counts <= hi))}"
                       for lo, hi in zip(lows, bounds))
    extras = "; ".join(f"{f} {np.min([e[f] for e in examples])}-{np.max([e[f] for e in examples])} "
                       f"(mean {np.mean([e[f] for e in examples]):.1f})" for f in extra_fields)
    held = np.mean([e["num_solutions"] > len(e["labels"]) for e in examples])
    print(f"{name}: {len(examples)} inputs; solutions per input: {groups} (max {counts.max()}); {extras}; stored "
          f"solutions per input: mean {np.mean([len(e['labels']) for e in examples]):.1f}; inputs with held-out "
          f"solutions: {100 * held:.1f}%")
