"""
Maze Shortest-Path Dataset with Multiple Shortest Paths

Each maze is a size x size token grid (see evaluators/maze_utils.py): a standard maze on a lattice of rooms at
odd coordinates, with extra walls removed to create loops, and a start and a goal room. The label marks the
cells of one shortest start-goal path. Loops make several paths equally short; their number is counted exactly,
so mazes are drawn to a target distribution of path counts: --unique-share of mazes with a unique shortest
path, the rest spread evenly over log2-spaced groups (2-3, 4-7, ...) up to the largest count the lattice allows.

Writes four datasets sharing the same test split (one row per maze, with label_set holding up to
--max-stored-paths paths and num_solutions the exact number of shortest paths):
  <out>          train: every stored path, grouped by maze, so each epoch sees one random stored path per maze
  <out>-1oml     train: one row per maze with label_set of its stored paths (for 1oML objectives)
  <out>-single   train: one fixed random path per maze
  <out>-unique   train: only mazes with a unique shortest path

Usage:
    python dataset/build_maze_dataset.py --size 15 --num-train 100000 --num-test 2000 --output-dir data/maze-15
"""
import json
import math
import os
import sys
from multiprocessing import Pool, cpu_count
from typing import List

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.common import PuzzleDatasetMetadata  # noqa: E402
from evaluators.maze_utils import ShortestPaths, WALL, OPEN, START, GOAL, PATH  # noqa: E402

cli = ArgParser()


class DataProcessConfig(BaseModel):
    size: int = 15
    output_dir: str = "data/maze-15"
    num_train: int = 100000
    num_test: int = 2000
    unique_share: float = 1 / 3
    max_stored_paths: int = 16
    min_length: int = 8  # shortest path length in steps
    seed: int = 42


def make_maze(rng: np.random.Generator, size: int, loop_prob: float) -> np.ndarray:
    """A random spanning-tree maze on the room lattice, then each remaining inner wall removed with loop_prob."""
    rooms = (size - 1) // 2
    grid = np.full((size, size), WALL, dtype=np.uint8)
    grid[1::2, 1::2] = OPEN
    seen = np.zeros((rooms, rooms), dtype=bool)
    stack = [(int(rng.integers(rooms)), int(rng.integers(rooms)))]
    seen[stack[0]] = True
    while stack:
        r, c = stack[-1]
        unvisited = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                     if 0 <= r + dr < rooms and 0 <= c + dc < rooms and not seen[r + dr, c + dc]]
        if not unvisited:
            stack.pop()
            continue
        nr, nc = unvisited[rng.integers(len(unvisited))]
        grid[r + nr + 1, c + nc + 1] = OPEN  # the wall between the two rooms
        seen[nr, nc] = True
        stack.append((nr, nc))
    for r in range(1, size - 1):
        for c in range(1, size - 1):
            if grid[r, c] == WALL and r % 2 != c % 2 and rng.random() < loop_prob:
                grid[r, c] = OPEN
    return grid


def candidate(rng: np.random.Generator, size: int, min_length: int):
    """A maze with start and goal. Half are drawn as typical mazes, half with many loops and distant endpoints,
    which is where large path counts come from."""
    many_paths = rng.random() < 0.5
    grid = make_maze(rng, size, 0.7 + 0.3 * rng.random() if many_paths else rng.random() ** 0.5)
    rooms = np.argwhere(grid == OPEN)
    rooms = rooms[(rooms[:, 0] % 2 == 1) & (rooms[:, 1] % 2 == 1)]
    for _ in range(50):
        start, goal = rooms[rng.choice(len(rooms), 2, replace=False)]
        if not many_paths or np.abs(start - goal).sum() // 2 >= (size - 1) // 2 + 1:
            break
    grid[tuple(start)], grid[tuple(goal)] = START, GOAL
    paths = ShortestPaths(grid)
    return (grid, paths) if paths.length >= min_length else (None, None)


def stored_paths(paths: ShortestPaths, rng: np.random.Generator, limit: int) -> List[np.ndarray]:
    """Label grids for up to limit distinct shortest paths: all of them if few, else drawn uniformly."""
    if paths.num_paths <= limit:
        chosen = paths.all_paths()
    else:
        chosen, seen = [], set()
        while len(chosen) < limit:
            path = paths.sample_path(rng)
            if tuple(path) not in seen:
                seen.add(tuple(path))
                chosen.append(path)
    order = rng.permutation(len(chosen))
    return [paths.label(chosen[i]) for i in order]


def count_bins(max_count: int):
    """Upper bounds (inclusive) of the path-count groups: 1, then 2-3, 4-7, ..., up to max_count."""
    bounds = [1]
    while bounds[-1] < max_count:
        bounds.append(min(2 * bounds[-1] + 1, max_count))
    return bounds


def count_group(num_paths: int, bounds) -> int:
    return next(i for i, bound in enumerate(bounds) if num_paths <= bound)


def generate_chunk(args):
    seed, size, min_length, max_stored, num_candidates, bounds, needed = args
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num_candidates):
        grid, paths = candidate(rng, size, min_length)
        # Sampling stored paths is the slow part, so only do it for groups that still need mazes
        if grid is not None and needed[count_group(paths.num_paths, bounds)]:
            out.append((grid, paths.num_paths, stored_paths(paths, rng, max_stored)))
    return out


def draw_mazes(config: DataProcessConfig, quotas, bounds, rng, seen_inputs):
    """Mazes until every path-count group has its quota, generated in parallel but deterministically."""
    filled = [[] for _ in quotas]
    chunk = 1000
    with Pool(cpu_count()) as pool, tqdm(total=sum(quotas), desc="Drawing mazes") as bar:
        # Rounds of chunks with seeds drawn up front, so the result doesn't depend on the number of processes
        while not all(len(f) >= q for f, q in zip(filled, quotas)):
            needed = tuple(len(f) < q for f, q in zip(filled, quotas))
            tasks = [(int(rng.integers(2**63)), config.size, config.min_length, config.max_stored_paths, chunk, bounds, needed)
                     for _ in range(64)]
            for results in pool.imap(generate_chunk, tasks):
                for grid, num_paths, labels in results:
                    b = count_group(num_paths, bounds)
                    key = grid.tobytes()
                    if len(filled[b]) < quotas[b] and key not in seen_inputs:
                        seen_inputs.add(key)
                        filled[b].append((grid, num_paths, labels))
                        bar.update(1)
    mazes = [m for group in filled for m in group]
    rng.shuffle(mazes)
    return mazes


def quotas_for(num: int, share_unique: float, num_bins: int):
    unique = round(num * share_unique)
    rest = num - unique
    return [unique] + [rest // (num_bins - 1) + (1 if i < rest % (num_bins - 1) else 0) for i in range(num_bins - 1)]


def save_split(save_dir: str, examples, size: int, max_stored: int, groups=None, with_label_set=True):
    """examples: list of (input grid, label grid, num_paths, stored label grids). groups: examples per group."""
    os.makedirs(save_dir, exist_ok=True)
    n = len(examples)
    group_sizes = groups if groups is not None else [1] * n
    data = {
        "inputs": np.stack([e[0].ravel() for e in examples]).astype(np.uint8),
        "labels": np.stack([e[1].ravel() for e in examples]).astype(np.uint8),
        "num_solutions": np.array([e[2] for e in examples], dtype=np.int32),
        "puzzle_identifiers": np.zeros(n, dtype=np.int32),
        "puzzle_indices": np.arange(n + 1, dtype=np.int32),
        "group_indices": np.concatenate([[0], np.cumsum(group_sizes)]).astype(np.int32),
    }
    if with_label_set:
        label_set = np.zeros((n, max_stored, size * size), dtype=np.uint8)  # 0 = ignore id, so padding is skipped
        for i, e in enumerate(examples):
            label_set[i, :len(e[3])] = np.stack([g.ravel() for g in e[3]])
        data["label_set"] = label_set
    for k, v in data.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    metadata = PuzzleDatasetMetadata(
        seq_len=size * size, vocab_size=PATH + 1, pad_id=0, ignore_label_id=0, blank_identifier_id=0,
        num_puzzle_identifiers=1, total_groups=len(group_sizes), mean_puzzle_examples=1, total_puzzles=n, sets=["all"])
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)


def describe(name, mazes, bounds):
    counts = np.array([m[1] for m in mazes])
    lengths = np.array([ShortestPaths(m[0]).length for m in mazes[:2000]])
    lows = [1] + [b + 1 for b in bounds[:-1]]
    groups = ", ".join(f"{lo}-{hi}: {np.sum((counts >= lo) & (counts <= hi))}" if lo != hi else f"{lo}: {np.sum(counts == lo)}"
                       for lo, hi in zip(lows, bounds))
    print(f"{name}: {len(mazes)} mazes; shortest paths per maze: {groups}; path length {lengths.min()}-{lengths.max()} "
          f"(mean {lengths.mean():.1f}); stored paths per maze: mean {np.mean([len(m[2]) for m in mazes]):.1f}")


@cli.command(singleton=True)
def build_maze_dataset(config: DataProcessConfig):
    rng = np.random.default_rng(config.seed)
    rooms = (config.size - 1) // 2
    bounds = count_bins(math.comb(2 * (rooms - 1), rooms - 1))  # most paths: open lattice, opposite corners
    seen = set()

    test = draw_mazes(config, quotas_for(config.num_test, config.unique_share, len(bounds)), bounds, rng, seen)
    train = draw_mazes(config, quotas_for(config.num_train, config.unique_share, len(bounds)), bounds, rng, seen)
    describe("test", test, bounds)
    describe("train", train, bounds)

    # Test: one row per maze, labelled with its first stored path, identical in every dataset
    test_rows = [(grid, labels[0], num_paths, labels) for grid, num_paths, labels in test]
    pick = lambda labels: labels[rng.integers(len(labels))]
    variants = {
        "": dict(rows=[(grid, label, num_paths, labels) for grid, num_paths, labels in train for label in labels],
                 groups=[len(labels) for _, _, labels in train], with_label_set=False),
        "-1oml": dict(rows=[(grid, pick(labels), num_paths, labels) for grid, num_paths, labels in train]),
        "-single": dict(rows=[(grid, pick(labels), num_paths, labels[:1]) for grid, num_paths, labels in train], with_label_set=False),
        "-unique": dict(rows=[(grid, labels[0], num_paths, labels) for grid, num_paths, labels in train if num_paths == 1]),
    }
    for suffix, spec in variants.items():
        out = config.output_dir + suffix
        save_split(os.path.join(out, "train"), spec["rows"], config.size, config.max_stored_paths,
                   groups=spec.get("groups"), with_label_set=spec.get("with_label_set", True))
        save_split(os.path.join(out, "test"), test_rows, config.size, config.max_stored_paths)
        with open(os.path.join(out, "identifiers.json"), "w") as f:
            json.dump(["<blank>"], f)
        print(f"Saved {out}: {len(spec['rows'])} train examples in {len(spec.get('groups') or spec['rows'])} groups")


if __name__ == "__main__":
    cli()
