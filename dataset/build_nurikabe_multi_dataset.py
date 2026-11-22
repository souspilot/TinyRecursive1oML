"""
Nurikabe with Many Solutions

Base puzzles are the 10x10 nurikabe puzzles of Pencil Puzzle Bench (https://arxiv.org/abs/2603.02119, MIT licence;
https://huggingface.co/datasets/bluecoconut/pencil-puzzle-bench, file full_dataset.jsonl), made by human setters
and each with a unique solution. Base puzzles are split into train, test and validation pools first (80/10/10), so
no puzzle appears in more than one split.

An input is a base puzzle under a random rotation or reflection with k of its numbers replaced by "?" (an island of
any size), k uniform from 0 to --max-hidden-share of them. Deleting a number outright would forbid its island, so
hiding the value is the change that only removes a constraint: every solution of the base puzzle stays a solution
and the count can only grow. It is counted exactly with clingo (answer set solver, pip install clingo) up to the top
of the last group. Inputs are drawn to a target distribution of solution counts (--unique-share unique, the rest
spread evenly over 2-3, 4-15, 16-63, 64-255, 256-1023) and each stores up to --max-stored-solutions; see
dataset/multi_solution.py for the four datasets written and the held-out solutions on training inputs. Also writes
<out>-1oml/val from the validation pool, in the format of the test split.

Encoding: evaluators/nurikabe_utils.py.

Usage (CPU only, needs clingo):
    python dataset/build_nurikabe_multi_dataset.py --source full_dataset.jsonl --output-dir data/nurikabe-multi
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
from evaluators.nurikabe_utils import MAX_NUMBER, SHADED, UNSHADED, VOCAB_SIZE, encode  # noqa: E402

cli = ArgParser()
BOUNDS = (1, 3, 15, 63, 255, 1023)
SIZE = 10


class DataProcessConfig(BaseModel):
    source: str = "full_dataset.jsonl"
    output_dir: str = "data/nurikabe-multi"
    num_train: int = 100000
    num_test: int = 2000
    num_val: int = 1000
    unique_share: float = 1 / 3
    max_hidden_share: float = 0.4
    max_stored_solutions: int = 16
    seed: int = 42


def decode_url(url: str) -> np.ndarray:
    """puzz.link nurikabe URL -> (h, w) numbers: 0 = empty, -1 = "?", n >= 1 = number."""
    body = url.split("?", 1)[1].split("/")
    w, h, s = int(body[1]), int(body[2]), body[3]
    cells = np.zeros(w * h, dtype=np.int64)
    i, c = 0, 0
    while i < len(s) and c < w * h:
        ch = s[i]
        if ch in "0123456789abcdef":
            cells[c] = int(ch, 16); i += 1
        elif ch == "-":
            cells[c] = int(s[i + 1:i + 3], 16); i += 3
        elif ch == "+":
            cells[c] = int(s[i + 1:i + 4], 16); i += 4
        elif ch == ".":
            cells[c] = -1; i += 1
        elif "g" <= ch <= "z":
            c += int(ch, 36) - 16; i += 1
        else:
            raise ValueError(f"Unexpected character {ch!r} in {url}")
        c += 1
    return cells.reshape(h, w)


def solve(numbers: np.ndarray, cap: int, timeout: float = 2.0):
    """Up to cap solutions, as a list of (n * n,) bool arrays (True = shaded); None if not done within timeout seconds."""
    import clingo
    h, w = numbers.shape
    clues = [(k, r, c, int(numbers[r, c])) for k, (r, c) in enumerate(zip(*np.nonzero(numbers)))]
    lines = [f"cell({r},{c})." for r in range(h) for c in range(w)]
    lines += [f"clue({k},{r},{c},{v})." for k, r, c, v in clues]
    # A cell can join island k only if it is close enough to k's number (any cell for "?")
    lines += [f"may({r},{c},{k})." for k, r0, c0, v in clues for r in range(h) for c in range(w)
              if v < 0 or abs(r - r0) + abs(c - c0) < v]
    lines += ["{ s(R,C) } :- cell(R,C).", "u(R,C) :- cell(R,C), not s(R,C).", ":- clue(_,R,C,_), s(R,C).",
              ":- s(R,C), s(R+1,C), s(R,C+1), s(R+1,C+1).",
              # Shaded cells connected: grow from the first shaded cell in reading order
              f"later(R,C) :- s(R2,C2), cell(R,C), R2*{w}+C2 < R*{w}+C.", "root(R,C) :- s(R,C), not later(R,C).",
              "reach(R,C) :- root(R,C).", "reach(R2,C2) :- reach(R,C), s(R2,C2), |R-R2|+|C-C2|=1.",
              ":- s(R,C), not reach(R,C).",
              # Each unshaded cell is in exactly one island, grown from its number; touching unshaded cells share it
              "1 { isl(R,C,K) : may(R,C,K) } 1 :- u(R,C).", "isl(R,C,K) :- clue(K,R,C,_).",
              ":- isl(R,C,K), u(R2,C2), |R-R2|+|C-C2|=1, not isl(R2,C2,K).",
              "ir(R,C,K) :- clue(K,R,C,_).", "ir(R2,C2,K) :- ir(R,C,K), isl(R2,C2,K), |R-R2|+|C-C2|=1.",
              ":- isl(R,C,K), not ir(R,C,K).",
              ":- clue(K,_,_,N), N > 0, #count{ R,C : isl(R,C,K) } != N.",
              "#show s/2."]
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


def hide_numbers(numbers: np.ndarray, rng: np.random.Generator, max_share: float) -> np.ndarray:
    known = np.argwhere(numbers > 0)
    k = int(rng.integers(round(max_share * len(known)) + 1))
    out = numbers.copy()
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
        labels = [np.where(solutions[i], SHADED, UNSHADED).astype(np.uint8) for i in order]
        out.append(dict(inputs=encode(numbers), labels=labels, num_solutions=len(solutions),
                        num_numbers=int((numbers != 0).sum()), num_hidden=int((numbers < 0).sum())))
    return out


@cli.command(singleton=True)
def build_nurikabe_multi_dataset(config: DataProcessConfig):
    rng = np.random.default_rng(config.seed)
    base = []
    with open(config.source) as f:
        for line in f:
            r = json.loads(line)
            if r["pid"] == "nurikabe" and int(r["width"]) == SIZE and int(r["height"]) == SIZE:
                numbers = decode_url(r["puzzlink_url"])
                if numbers.max() <= MAX_NUMBER:
                    base.append(numbers.tolist())
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
    extra = ["num_numbers", "num_hidden"]
    for name, split in (("test", test), ("val", val), ("train", train)):
        multi_solution.describe(name, split, BOUNDS, extra)
    seq_len = SIZE * SIZE
    multi_solution.write_variants(config.output_dir, train, test, rng, seq_len=seq_len, vocab_size=VOCAB_SIZE,
                                  max_stored=config.max_stored_solutions, extra_fields=extra)
    multi_solution.save_split(os.path.join(config.output_dir + "-1oml", "val"), [(e, e["labels"][0]) for e in val],
                              seq_len, VOCAB_SIZE, config.max_stored_solutions, extra)


if __name__ == "__main__":
    cli()
