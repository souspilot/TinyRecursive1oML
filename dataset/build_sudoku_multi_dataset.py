"""
Sudoku with Many Solutions

Each input is a 9x9 Sudoku with clues kept from a random complete grid: --clue-min to --clue-max clues at random
cells. Fewer clues leave more solutions, so the clue count is the ambiguity dial. Every solution is enumerated
exactly (a bitmask solver); puzzles with more than --max-solutions are dropped. Inputs are drawn to a target
distribution of solution counts (--unique-share unique, the rest spread over 2-3, 4-15, 16-63, 64-255 and
256-max), and each stores up to --max-stored-solutions solutions; see dataset/multi_solution.py for the four
datasets written and the held-out solutions on training inputs.

Tokens follow evaluators/sudoku_utils.py: PAD=0, blank=1, digits 1-9 are tokens 2-10. The label is the whole
solved grid (clues included). Per input, num_clues is saved too.

Usage:
    python dataset/build_sudoku_multi_dataset.py --num-train 100000 --num-test 2000 --output-dir data/sudoku-multi
"""
import os
import sys

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import multi_solution  # noqa: E402

cli = ArgParser()

BOXES = np.array([(r // 3) * 3 + c // 3 for r in range(9) for c in range(9)])
ALL = 0x3FE  # bits 1..9


class DataProcessConfig(BaseModel):
    output_dir: str = "data/sudoku-multi"
    num_train: int = 100000
    num_test: int = 2000
    clue_min: int = 26
    clue_max: int = 50
    max_solutions: int = 1024
    unique_share: float = 1 / 3
    max_stored_solutions: int = 16
    seed: int = 42


def bounds_for(max_solutions: int):
    return tuple(b for b in (1, 3, 15, 63, 255) if b < max_solutions) + (max_solutions,)


def enumerate_solutions(grid: np.ndarray, cap: int, rng: np.random.Generator = None):
    """Solutions of a grid (81 ints, 0 = blank), up to cap + 1 of them (so more than cap means "too many").
    With rng, candidates are tried in random order (to draw a random complete grid with cap=1)."""
    grid = [int(v) for v in grid]
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for i, v in enumerate(grid):
        if v:
            bit = 1 << v
            r, c, b = i // 9, i % 9, BOXES[i]
            if (rows[r] | cols[c] | boxes[b]) & bit:
                return []
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
    blanks = [i for i, v in enumerate(grid) if not v]
    found = []

    def search():
        if len(found) > cap:
            return
        # Most constrained blank first
        best, best_mask, best_count = -1, 0, 10
        for i in blanks:
            if grid[i]:
                continue
            mask = ALL & ~(rows[i // 9] | cols[i % 9] | boxes[BOXES[i]])
            count = bin(mask).count("1")
            if count < best_count:
                best, best_mask, best_count = i, mask, count
                if count <= 1:
                    break
        if best < 0:
            found.append(np.array(grid, dtype=np.uint8))
            return
        if best_count == 0:
            return
        digits = [d for d in range(1, 10) if best_mask >> d & 1]
        if rng is not None:
            rng.shuffle(digits)
        r, c, b = best // 9, best % 9, BOXES[best]
        for d in digits:
            bit = 1 << d
            grid[best] = d
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            search()
            rows[r] &= ~bit
            cols[c] &= ~bit
            boxes[b] &= ~bit
            grid[best] = 0
            if len(found) > cap:
                return

    search()
    return found


def random_grid(rng: np.random.Generator) -> np.ndarray:
    return enumerate_solutions(np.zeros(81, dtype=np.uint8), 0, rng)[0]


def generate_chunk(args):
    seed, config, num_candidates, needed = args
    config = DataProcessConfig(**config)  # passed as a dict: argdantic's wrapped config class can't be pickled
    rng = np.random.default_rng(seed)
    bounds = bounds_for(config.max_solutions)
    out = []
    for _ in range(num_candidates):
        solution = random_grid(rng)
        num_clues = int(rng.integers(config.clue_min, config.clue_max + 1))
        puzzle = np.zeros(81, dtype=np.uint8)
        keep = rng.choice(81, num_clues, replace=False)
        puzzle[keep] = solution[keep]
        solutions = enumerate_solutions(puzzle, config.max_solutions)
        if len(solutions) > config.max_solutions or not needed[multi_solution.count_group(len(solutions), bounds)]:
            continue
        order = rng.permutation(len(solutions))[:config.max_stored_solutions]
        out.append(dict(
            inputs=np.where(puzzle > 0, puzzle + 1, 1).astype(np.uint8),
            labels=[(solutions[i] + 1).astype(np.uint8) for i in order],
            num_solutions=len(solutions),
            num_clues=num_clues,
        ))
    return out


@cli.command(singleton=True)
def build_sudoku_multi_dataset(config: DataProcessConfig):
    rng = np.random.default_rng(config.seed)
    bounds = bounds_for(config.max_solutions)
    fields = {k: getattr(config, k) for k in DataProcessConfig.model_fields}
    seen = set()
    test = multi_solution.draw(generate_chunk, fields, multi_solution.quotas_for(config.num_test, config.unique_share, len(bounds)),
                               bounds, rng, seen, desc="Drawing test puzzles")
    train = multi_solution.draw(generate_chunk, fields, multi_solution.quotas_for(config.num_train, config.unique_share, len(bounds)),
                                bounds, rng, seen, desc="Drawing train puzzles")
    multi_solution.describe("test", test, bounds, ["num_clues"])
    multi_solution.describe("train", train, bounds, ["num_clues"])
    multi_solution.write_variants(config.output_dir, train, test, rng, seq_len=81, vocab_size=11,
                                  max_stored=config.max_stored_solutions, extra_fields=["num_clues"])


if __name__ == "__main__":
    cli()
