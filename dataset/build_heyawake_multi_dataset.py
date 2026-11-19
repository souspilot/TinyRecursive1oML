"""
Heyawake with Many Solutions

Base puzzles are the 10x10 heyawake puzzles of Pencil Puzzle Bench (https://arxiv.org/abs/2603.02119, MIT licence;
https://huggingface.co/datasets/bluecoconut/pencil-puzzle-bench, file full_dataset.jsonl), made by human setters
and each with a unique solution. Base puzzles are split into train, test and validation pools first (80/10/10), so
no puzzle's room layout appears in more than one split.

An input is a base puzzle under a random rotation or reflection with k of its room numbers removed, k uniform from
0 to 40% of them (more removed mostly gives over 1023 solutions). Removing a number only removes a constraint, so the solution count can only grow;
it is counted exactly with clingo (answer set solver, pip install clingo) up to the top of the last group. Inputs
are drawn to a target distribution of solution counts (--unique-share unique, the rest spread evenly over 2-3,
4-15, 16-63, 64-255, 256-1023) and each stores up to --max-stored-solutions; see dataset/multi_solution.py for the
four datasets written and the held-out solutions on training inputs. Also writes <out>-1oml/val from the
validation pool, in the format of the test split.

Encoding: evaluators/heyawake_utils.py.

Usage (CPU only, needs clingo):
    python dataset/build_heyawake_multi_dataset.py --source full_dataset.jsonl --output-dir data/heyawake-multi
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
from evaluators.heyawake_utils import MAX_CLUE, SHADED, UNSHADED, VOCAB_SIZE, encode  # noqa: E402

cli = ArgParser()
BOUNDS = (1, 3, 15, 63, 255, 1023)
SIZE = 10


class DataProcessConfig(BaseModel):
    source: str = "full_dataset.jsonl"
    output_dir: str = "data/heyawake-multi"
    num_train: int = 50000
    num_test: int = 2000
    num_val: int = 1000
    unique_share: float = 1 / 3
    max_stored_solutions: int = 16
    seed: int = 42


def decode_url(url: str):
    """puzz.link heyawake URL -> (room ids (h, w), {room id: number}); rooms numbered by first cell in reading order."""
    body = url.split("?", 1)[1].split("/")
    w, h, s = int(body[1]), int(body[2]), body[3]
    nv, nh = (w - 1) * h, w * (h - 1)
    p1, p2 = (nv + 4) // 5, (nv + 4) // 5 + (nh + 4) // 5
    bits = [(int(ch, 32) >> (4 - j)) & 1 for ch in s[:p2] for j in range(5)]
    vb, hb = bits[:nv], bits[p1 * 5:p1 * 5 + nh]
    room = -np.ones((h, w), dtype=np.int64)
    n = 0
    for r in range(h):
        for c in range(w):
            if room[r, c] >= 0:
                continue
            room[r, c] = n
            stack = [(r, c)]
            while stack:
                y, x = stack.pop()
                for ny, nx, open_ in ((y, x + 1, x + 1 < w and not vb[y * (w - 1) + x]),
                                      (y, x - 1, x > 0 and not vb[y * (w - 1) + x - 1]),
                                      (y + 1, x, y + 1 < h and not hb[y * w + x]),
                                      (y - 1, x, y > 0 and not hb[(y - 1) * w + x])):
                    if open_ and room[ny, nx] < 0:
                        room[ny, nx] = n
                        stack.append((ny, nx))
            n += 1
    clues, i, rid, t = {}, 0, 0, s[p2:]
    while i < len(t) and rid < n:
        ch = t[i]
        if ch in "0123456789abcdef":
            clues[rid] = int(ch, 16); i += 1
        elif ch == "-":
            clues[rid] = int(t[i + 1:i + 3], 16); i += 3
        elif ch == "+":
            clues[rid] = int(t[i + 1:i + 4], 16); i += 4
        elif ch == ".":
            i += 1  # a room marked with an unknown number: treated as no number
        elif "g" <= ch <= "z":
            rid += int(ch, 36) - 16; i += 1
        else:
            raise ValueError(f"Unexpected character {ch!r} in {url}")
        rid += 1
    return room, clues


def solve(room: np.ndarray, clues: dict, cap: int, timeout: float = 2.0):
    """Up to cap solutions, as a list of (n * n,) bool arrays (True = shaded); None if not done within timeout seconds."""
    import clingo
    h, w = room.shape
    lines = [f"cell({r},{c}). room({r},{c},{room[r, c]})." for r in range(h) for c in range(w)]
    lines += [f"clue({k},{v})." for k, v in clues.items()]
    lines += ["{ s(R,C) } :- cell(R,C).", "u(R,C) :- cell(R,C), not s(R,C).",
              ":- s(R,C), s(R,C+1).", ":- s(R,C), s(R+1,C).",
              ":- clue(I,N), #count{ R,C : s(R,C), room(R,C,I) } != N.",
              # Shaded cells never touch, so (0,0) or (0,1) is unshaded: grow the connected area from there
              "root(0,0) :- u(0,0).", "root(0,1) :- s(0,0).", "reach(R,C) :- root(R,C).",
              "reach(R2,C2) :- reach(R,C), u(R2,C2), |R-R2|+|C-C2|=1.", ":- u(R,C), not reach(R,C).",
              "#show s/2."]
    # An unshaded straight line may cross at most one room border: forbid each shortest line crossing two
    for grid, cell in ((room, lambda a, b: f"u({a},{b})"), (room.T, lambda a, b: f"u({b},{a})")):
        for a, ids in enumerate(grid):
            for b1 in range(len(ids)):
                k = 0
                for b2 in range(b1 + 1, len(ids)):
                    k += ids[b2] != ids[b2 - 1]
                    if k == 2:
                        lines.append(":- " + ", ".join(cell(a, b) for b in range(b1, b2 + 1)) + ".")
                        break
    ctl = clingo.Control([str(cap), "--warn=none"])
    ctl.add("base", [], "\n".join(lines))
    ctl.ground([("base", [])])
    out = []

    def on_model(model):
        sol = np.zeros(h * w, dtype=bool)
        for a in model.symbols(shown=True):
            sol[a.arguments[0].number * w + a.arguments[1].number] = True
        out.append(sol)

    # A few puzzles (e.g. a room needing 30 shaded cells) take minutes: give up on them rather than wait
    with ctl.solve(on_model=on_model, async_=True) as handle:
        if not handle.wait(timeout):
            handle.cancel()
            return None
    return out


@lru_cache(maxsize=None)
def load_pool(path: str, pool: str):
    with open(path) as f:
        return [(np.array(p["room"]), {int(k): v for k, v in p["clues"].items()}) for p in json.load(f)[pool]]


def transform(room: np.ndarray, tid: int) -> np.ndarray:
    """One of the 8 rotations and reflections of the grid."""
    out = np.rot90(room, tid % 4)
    return np.ascontiguousarray(out.T if tid >= 4 else out)


def generate_chunk(args):
    seed, task, num_candidates, needed = args
    rng = np.random.default_rng(seed)
    pool = load_pool(task["pool_file"], task["pool"])
    out = []
    for _ in range(num_candidates):
        base_room, base_clues = pool[rng.integers(len(pool))]
        room = transform(base_room, int(rng.integers(8)))
        keys = sorted(base_clues)
        drop = set(rng.choice(keys, size=int(rng.integers(round(0.4 * len(keys)) + 1)), replace=False).tolist()) if keys else set()
        clues = {k: v for k, v in base_clues.items() if k not in drop}
        solutions = solve(room, clues, BOUNDS[-1] + 1)
        if not solutions or len(solutions) > BOUNDS[-1] or not needed[multi_solution.count_group(len(solutions), BOUNDS)]:
            continue
        order = rng.permutation(len(solutions))[:task["max_stored"]]
        labels = [np.where(solutions[i], SHADED, UNSHADED).astype(np.uint8) for i in order]
        out.append(dict(inputs=encode(room, clues), labels=labels, num_solutions=len(solutions),
                        num_clues=len(clues), num_rooms=int(room.max()) + 1))
    return out


@cli.command(singleton=True)
def build_heyawake_multi_dataset(config: DataProcessConfig):
    rng = np.random.default_rng(config.seed)
    base = []
    with open(config.source) as f:
        for line in f:
            r = json.loads(line)
            if r["pid"] == "heyawake" and int(r["width"]) == SIZE and int(r["height"]) == SIZE:
                room, clues = decode_url(r["puzzlink_url"])
                if max(clues.values(), default=0) <= MAX_CLUE:
                    base.append(dict(room=room.tolist(), clues=clues))
    order = rng.permutation(len(base))
    n_test, n_val = len(base) // 10, len(base) // 10
    pools = {"test": [base[i] for i in order[:n_test]], "val": [base[i] for i in order[n_test:n_test + n_val]],
             "train": [base[i] for i in order[n_test + n_val:]]}
    print(f"{len(base)} base puzzles: " + ", ".join(f"{k} {len(v)}" for k, v in pools.items()))
    os.makedirs(config.output_dir, exist_ok=True)
    pool_file = os.path.abspath(os.path.join(config.output_dir, "base_puzzles.json"))
    with open(pool_file, "w") as f:
        json.dump(pools, f)

    task = lambda pool: dict(pool_file=pool_file, pool=pool, max_stored=config.max_stored_solutions)
    quotas = lambda num: multi_solution.quotas_for(num, config.unique_share, len(BOUNDS))
    seen = set()
    test = multi_solution.draw(generate_chunk, task("test"), quotas(config.num_test), BOUNDS, rng, seen,
                               chunk=20, desc="Drawing test inputs")
    val = multi_solution.draw(generate_chunk, task("val"), quotas(config.num_val), BOUNDS, rng, seen,
                              chunk=20, desc="Drawing validation inputs")
    train = multi_solution.draw(generate_chunk, task("train"), quotas(config.num_train), BOUNDS, rng, seen,
                                chunk=20, desc="Drawing train inputs")
    extra = ["num_clues", "num_rooms"]
    for name, split in (("test", test), ("val", val), ("train", train)):
        multi_solution.describe(name, split, BOUNDS, extra)
    seq_len = SIZE * SIZE
    multi_solution.write_variants(config.output_dir, train, test, rng, seq_len=seq_len, vocab_size=VOCAB_SIZE,
                                  max_stored=config.max_stored_solutions, extra_fields=extra)
    multi_solution.save_split(os.path.join(config.output_dir + "-1oml", "val"), [(e, e["labels"][0]) for e in val],
                              seq_len, VOCAB_SIZE, config.max_stored_solutions, extra)


if __name__ == "__main__":
    cli()
