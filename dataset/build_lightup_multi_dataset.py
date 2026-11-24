"""
Light Up (Akari) with Many Solutions

Base puzzles are the 10x10 light up (akari) puzzles of Pencil Puzzle Bench (https://arxiv.org/abs/2603.02119, MIT licence;
https://huggingface.co/datasets/bluecoconut/pencil-puzzle-bench, file full_dataset.jsonl), made by human setters
and each with a unique solution. Base puzzles are split into train, test and validation pools first (80/10/10), so
no puzzle appears in more than one split.

An input is a base puzzle under a random rotation or reflection with the numbers of k of its numbered walls removed
(the wall stays), k uniform from 0 to --max-hidden-share of them. Removing a number only removes a constraint, so
every solution of the base puzzle stays a solution and the count can only grow. It is counted exactly with clingo (answer set solver, pip install clingo) up to the top
of the last group. Inputs are drawn to a target distribution of solution counts (--unique-share unique, the rest
spread evenly over 2-3, 4-15, 16-63, 64-255, 256-1023) and each stores up to --max-stored-solutions; see
dataset/multi_solution.py for the four datasets written and the held-out solutions on training inputs. Also writes
<out>-1oml/val from the validation pool, in the format of the test split.

Encoding: evaluators/lightup_utils.py.

Usage (CPU only, needs clingo):
    python dataset/build_lightup_multi_dataset.py --source full_dataset.jsonl --output-dir data/lightup-multi
"""
import json
import os
import sys
from functools import lru_cache

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import multi_solution  # noqa: E402
from evaluators.lightup_utils import BULB, NO_BULB, VOCAB_SIZE, encode, segments  # noqa: E402

cli = ArgParser()
BOUNDS = (1, 3, 15, 63, 255, 1023)
SIZE = 10


class DataProcessConfig(BaseModel):
    source: str = "full_dataset.jsonl"
    output_dir: str = "data/lightup-multi"
    num_train: int = 100000
    num_test: int = 2000
    num_val: int = 1000
    unique_share: float = 1 / 3
    max_hidden_share: float = 0.4
    max_stored_solutions: int = 16
    seed: int = 42


def decode_url(url: str) -> np.ndarray:
    """puzz.link akari URL -> (h, w) grid: -2 = white, -1 = wall without a number, 0..4 = numbered wall."""
    body = url.split("?", 1)[1].split("/")
    w, h, s = int(body[1]), int(body[2]), body[3]
    cells = np.full(w * h, -2, dtype=np.int64)
    c = 0
    for ch in s:
        if c >= w * h:
            break
        if ch == ".":
            cells[c] = -1
        elif ch in "01234":
            cells[c] = int(ch)
        elif ch in "56789":
            cells[c] = int(ch) - 5; c += 1      # then one white cell
        elif ch in "abcde":
            cells[c] = int(ch, 16) - 10; c += 2  # then two white cells
        elif "g" <= ch <= "z":
            c += int(ch, 36) - 16
        else:
            raise ValueError(f"Unexpected character {ch!r} in {url}")
        c += 1
    return cells.reshape(h, w)


def solve(grid: np.ndarray, cap: int, timeout: float = 2.0):
    """Up to cap solutions, as a list of (n * n,) bool arrays (True = bulb); None if not done within timeout seconds."""
    import clingo
    h, w = grid.shape
    white = grid == -2
    row, col = segments(white)
    lines = [f"w({r},{c}). rs({r},{c},{row[r, c]}). cs({r},{c},{col[r, c]})." for r, c in zip(*np.nonzero(white))]
    lines += [f"num({r},{c},{grid[r, c]})." for r, c in zip(*np.nonzero(grid >= 0))]
    lines += ["{ b(R,C) } :- w(R,C).",
              "hr(S) :- b(R,C), rs(R,C,S).", "hc(S) :- b(R,C), cs(R,C,S).",
              ":- rs(_,_,S), #count{ R,C : b(R,C), rs(R,C,S) } > 1.",
              ":- cs(_,_,S), #count{ R,C : b(R,C), cs(R,C,S) } > 1.",
              ":- w(R,C), rs(R,C,S), cs(R,C,T), not hr(S), not hc(T).",
              ":- num(R,C,N), #count{ R2,C2 : b(R2,C2), |R-R2|+|C-C2|=1 } != N.",
              "#show b/2."]
    ctl = clingo.Control([str(cap), "--warn=none", "--project"])
    ctl.add("base", [], "\n".join(lines))
    ctl.ground([("base", [])])
    out = []

    def on_model(model):
        sol = np.zeros(h * w, dtype=bool)
        for a in model.symbols(shown=True):
            sol[a.arguments[0].number * w + a.arguments[1].number] = True
        out.append(sol)

    with ctl.solve(on_model=on_model, async_=True) as handle:
        if not handle.wait(timeout):
            handle.cancel()
            return None
    return out


@lru_cache(maxsize=None)
def load_pool(path: str, pool: str):
    with open(path) as f:
        return [np.array(p) for p in json.load(f)[pool]]


def transform(grid: np.ndarray, tid: int) -> np.ndarray:
    """One of the 8 rotations and reflections of the grid."""
    out = np.rot90(grid, tid % 4)
    return np.ascontiguousarray(out.T if tid >= 4 else out)


def hide_numbers(grid: np.ndarray, rng: np.random.Generator, max_share: float) -> np.ndarray:
    known = np.argwhere(grid >= 0)
    k = int(rng.integers(round(max_share * len(known)) + 1))
    out = grid.copy()
    for r, c in known[rng.choice(len(known), size=k, replace=False)]:
        out[r, c] = -1
    return out


def generate_chunk(args):
    seed, task, num_candidates, needed = args
    rng = np.random.default_rng(seed)
    pool = load_pool(task["pool_file"], task["pool"])
    out = []
    for _ in range(num_candidates):
        numbers = hide_numbers(transform(pool[rng.integers(len(pool))], int(rng.integers(8))), rng, task["max_hidden"])
        solutions = solve(numbers, BOUNDS[-1] + 1)
        if not solutions or len(solutions) > BOUNDS[-1] or not needed[multi_solution.count_group(len(solutions), BOUNDS)]:
            continue
        order = rng.permutation(len(solutions))[:task["max_stored"]]
        labels = [np.where(solutions[i], BULB, NO_BULB).astype(np.uint8) for i in order]
        out.append(dict(inputs=encode(numbers), labels=labels, num_solutions=len(solutions),
                        num_walls=int((numbers != -2).sum()), num_numbers=int((numbers >= 0).sum())))
    return out


@cli.command(singleton=True)
def build_lightup_multi_dataset(config: DataProcessConfig):
    rng = np.random.default_rng(config.seed)
    base = []
    with open(config.source) as f:
        for line in f:
            r = json.loads(line)
            if r["pid"] == "lightup" and int(r["width"]) == SIZE and int(r["height"]) == SIZE:
                base.append(decode_url(r["puzzlink_url"]).tolist())
    order = rng.permutation(len(base))
    n_test, n_val = len(base) // 10, len(base) // 10
    pools = {"test": [base[i] for i in order[:n_test]], "val": [base[i] for i in order[n_test:n_test + n_val]],
             "train": [base[i] for i in order[n_test + n_val:]]}
    print(f"{len(base)} base puzzles: " + ", ".join(f"{k} {len(v)}" for k, v in pools.items()))
    os.makedirs(config.output_dir, exist_ok=True)
    pool_file = os.path.abspath(os.path.join(config.output_dir, "base_puzzles.json"))
    with open(pool_file, "w") as f:
        json.dump(pools, f)

    task = lambda pool: dict(pool_file=pool_file, pool=pool, max_stored=config.max_stored_solutions,
                             max_hidden=config.max_hidden_share)
    quotas = lambda num: multi_solution.quotas_for(num, config.unique_share, len(BOUNDS))
    seen = set()
    test = multi_solution.draw(generate_chunk, task("test"), quotas(config.num_test), BOUNDS, rng, seen,
                               chunk=20, desc="Drawing test inputs")
    val = multi_solution.draw(generate_chunk, task("val"), quotas(config.num_val), BOUNDS, rng, seen,
                              chunk=20, desc="Drawing validation inputs")
    train = multi_solution.draw(generate_chunk, task("train"), quotas(config.num_train), BOUNDS, rng, seen,
                                chunk=20, desc="Drawing train inputs")
    extra = ["num_walls", "num_numbers"]
    for name, split in (("test", test), ("val", val), ("train", train)):
        multi_solution.describe(name, split, BOUNDS, extra)
    seq_len = SIZE * SIZE
    multi_solution.write_variants(config.output_dir, train, test, rng, seq_len=seq_len, vocab_size=VOCAB_SIZE,
                                  max_stored=config.max_stored_solutions, extra_fields=extra)
    multi_solution.save_split(os.path.join(config.output_dir + "-1oml", "val"), [(e, e["labels"][0]) for e in val],
                              seq_len, VOCAB_SIZE, config.max_stored_solutions, extra)


if __name__ == "__main__":
    cli()
