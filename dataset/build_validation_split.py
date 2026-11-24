"""Add a validation split, used to choose beta and to read training curves, so the test split is only used to report.

Writes <task>-multi-1oml/val with 1000 inputs (one row per input, in random order, the format of the test split) and
changes nothing that already exists:

  maze, sudoku, graph   1000 new inputs from the task's own generator, none of them in train or test.
  nqueens               Every possible input is already in train or test, so new ones don't exist. val is the first
                        1000 test inputs (already in random order) and test_no_val is the rest. Report N-Queens on
                        test_no_val whenever val was used to decide anything.

Usage: python dataset/build_validation_split.py [maze nqueens sudoku graph]     (default: all four; CPU only)
Then check it with: python scripts/analyze_multi_samples.py --task <task> --data <dataset> --check_data
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import multi_solution  # noqa: E402

NUM_VAL = 1000
SEED = 4242  # not the seed of the train and test draws (42)
DATA = {"maze": "data/maze-multi-1oml", "nqueens": "data/nqueens-multi-1oml",
        "sudoku": "data/sudoku-multi-1oml", "graph": "data/graph-coloring-multi-1oml"}


def inputs_of(data_dir: str, split: str) -> np.ndarray:
    return np.load(os.path.join(data_dir, split, "all__inputs.npy"))


def used_inputs(data_dir: str) -> set:
    return {row.tobytes() for split in ("train", "test") for row in inputs_of(data_dir, split)}


def build_maze(data_dir: str, rng: np.random.Generator):
    from dataset import build_maze_dataset as maze
    config = maze.DataProcessConfig()
    rooms = (config.size - 1) // 2
    bounds = maze.count_bins(math.comb(2 * (rooms - 1), rooms - 1))
    quotas = maze.quotas_for(NUM_VAL, config.unique_share, len(bounds))
    val = maze.draw_mazes(config, quotas, bounds, rng, used_inputs(data_dir))
    maze.describe("val", val, bounds)
    rows = [(grid, labels[0], num_paths, labels) for grid, num_paths, labels in val]
    maze.save_split(os.path.join(data_dir, "val"), rows, config.size, config.max_stored_paths)


def build_multi_solution(data_dir: str, rng: np.random.Generator, module, bounds, seq_len: int, vocab_size: int,
                         extra_field: str, chunk: int):
    config = module.DataProcessConfig()
    fields = {k: getattr(config, k) for k in module.DataProcessConfig.model_fields}
    quotas = multi_solution.quotas_for(NUM_VAL, config.unique_share, len(bounds))
    val = multi_solution.draw(module.generate_chunk, fields, quotas, bounds, rng, used_inputs(data_dir), chunk=chunk,
                              desc="Drawing validation inputs")
    multi_solution.describe("val", val, bounds, [extra_field])
    multi_solution.save_split(os.path.join(data_dir, "val"), [(e, e["labels"][0]) for e in val], seq_len, vocab_size,
                              config.max_stored_solutions, [extra_field])


def build_sudoku(data_dir: str, rng: np.random.Generator):
    from dataset import build_sudoku_multi_dataset as sudoku
    bounds = sudoku.bounds_for(sudoku.DataProcessConfig().max_solutions)
    build_multi_solution(data_dir, rng, sudoku, bounds, 81, 11, "num_clues", chunk=200)


def build_graph(data_dir: str, rng: np.random.Generator):
    from dataset import build_graph_coloring_multi_dataset as graph
    config = graph.DataProcessConfig()
    n, k = config.num_nodes, config.num_colors
    build_multi_solution(data_dir, rng, graph, graph.bounds_for(n, k), n * (n + 1) // 2, 3 + k, "num_edges", chunk=1000)


def save_rows(data_dir: str, split: str, rows: range):
    """Copy test rows into a new split with the same format."""
    source = os.path.join(data_dir, "test")
    os.makedirs(os.path.join(data_dir, split), exist_ok=True)
    for field in ("inputs", "labels", "num_solutions", "label_set"):
        data = np.load(os.path.join(source, f"all__{field}.npy"), mmap_mode="r")
        np.save(os.path.join(data_dir, split, f"all__{field}.npy"), np.array(data[rows.start:rows.stop]))
    n = len(rows)
    np.save(os.path.join(data_dir, split, "all__puzzle_identifiers.npy"), np.zeros(n, dtype=np.int32))
    for field in ("puzzle_indices", "group_indices"):
        np.save(os.path.join(data_dir, split, f"all__{field}.npy"), np.arange(n + 1, dtype=np.int32))
    with open(os.path.join(source, "dataset.json")) as f:
        metadata = json.load(f)
    metadata.update(total_groups=n, total_puzzles=n)
    with open(os.path.join(data_dir, split, "dataset.json"), "w") as f:
        json.dump(metadata, f)
    print(f"Saved {split}: {n} inputs copied from test")


def build_nqueens(data_dir: str, rng: np.random.Generator):
    total = len(inputs_of(data_dir, "test"))
    save_rows(data_dir, "val", range(0, NUM_VAL))
    save_rows(data_dir, "test_no_val", range(NUM_VAL, total))


def main():
    builders = {"maze": build_maze, "nqueens": build_nqueens, "sudoku": build_sudoku, "graph": build_graph}
    tasks = sys.argv[1:] or list(builders)
    unknown = [t for t in tasks if t not in builders]
    if unknown:
        sys.exit(f"Unknown task(s): {unknown} (maze, nqueens, sudoku, graph)")
    for task in tasks:
        data_dir = DATA[task]
        if os.path.exists(os.path.join(data_dir, "val", "dataset.json")):
            print(f"{data_dir}/val exists, skipping")
            continue
        builders[task](data_dir, np.random.default_rng(SEED))
        val = inputs_of(data_dir, "val")
        test = inputs_of(data_dir, "test")
        if val.shape[1:] != test.shape[1:]:
            sys.exit(f"{task}: val inputs have shape {val.shape[1:]}, test inputs {test.shape[1:]}")
        print(f"{data_dir}/val: {len(val)} inputs")


if __name__ == "__main__":
    main()
