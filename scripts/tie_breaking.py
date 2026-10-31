"""Does a one-of-many model blend the solutions, or break the tie and lower its confidence? When, and on what?

The 1oML paper (Nandwani et al., ICLR'21) motivates target selection by blending: cross-entropy against several
solutions drives the output to their average, which decodes to none of them. On 10x10 N-Queens our naive and
Random TRMs do not blend: each row's queen mass sits on one column, but below 0.5, so reading every cell at
once (p >= 0.5) fails and reading each row's top column works (scripts/nqueens_marginals.py). This measures, for
N-Queens and mazes, from one model run per input (no re-running, no sampling):

  readout        valid answers from the one-pass structured readout: each row's top column (N-Queens), or the
                 greedy walk from the start over the model's PATH probabilities (mazes).
  choice points  where the solutions differ: rows the solutions put their queen in different columns of, and
                 forks (cells on shortest paths with several shortest continuations). There, the model's
                 normalized top (its largest probability among the options over their sum: ~1 = one option
                 chosen, ~the exact top = blend), the mass on the options, and the exact top for reference.
  confidence     probability on cells every solution agrees on, for one-solution and multi-solution inputs.
  by step        all of the above after each supervision step (ACT step), and how often each choice point's
                 option already matches the final step's: when is the tie broken?
  symmetry       the 8 board symmetries map solutions to solutions. Decoded on a transformed input and mapped
                 back, is the answer the same (the choice follows the input's content) or another solution
                 (it follows absolute position)? Different answers give free diversity: any-valid, majority vote
                 and coverage over the 8 views.
  init noise     the recursion starts from a fixed latent; with noise added to it (--init_noise, per input and
                 position), do the answers change?
  lex-first      N-Queens: how often a valid answer is the lexicographically first solution (columns row by row),
                 against the chance rate 1/#solutions: a positional preference.

Usage:
    python scripts/tie_breaking.py --task nqueens --data data/nqueens-10-1oml --out results/nqueens_10_commit/tie_breaking \\
        --runs naive=checkpoints/Nqueens-10-compare/trm_seed42/final/checkpoint.pt [name=checkpoint ...]
    python scripts/tie_breaking.py --task maze --data data/maze-15-1oml --out results/maze_15/tie_breaking --runs ...

Writes <out>/summary.md and <out>/<name>.json.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluators.maze_utils import ShortestPaths, batch_verify_maze, PAD, WALL, OPEN, START, GOAL, PATH  # noqa: E402
from evaluators.nqueens_utils import verify_nqueens  # noqa: E402
from sample_predictions import load_split, load_model  # noqa: E402

QUEEN = 2
REPORT_STEPS = (1, 2, 3, 4, 6, 8, 12, 16)
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))
MIN_FORK_MASS = 0.05


class NQueens:
    token = QUEEN

    def __init__(self, data_path, boards):
        q, n, _ = boards.shape
        label_set = np.load(os.path.join(data_path, "test", "all__label_set.npy"))
        self.num_solutions = np.load(os.path.join(data_path, "test", "all__num_solutions.npy"))[:q]
        stored = np.minimum(self.num_solutions, label_set.shape[1])
        self.solutions = [label_set[i, :k].reshape(k, n, n) for i, k in enumerate(stored)]
        self.shares = np.stack([(s == QUEEN).mean(0) for s in self.solutions])
        self.boards = boards
        open_rows = ~(boards == QUEEN).any(2)
        self.choice = open_rows & (self.shares.max(2) < 1 - 1e-9)
        self.agreed = open_rows & ~self.choice
        self.choice_query = np.nonzero(self.choice)[0]
        self.exact_top = self.shares.max(2)[self.choice]

    def readout(self, boards, probs):
        """Each queen-less row gets a queen at its most likely column."""
        preds = boards.copy()
        q_idx, r_idx = np.nonzero(~(boards == QUEEN).any(2))
        preds[q_idx, r_idx, probs.argmax(2)[q_idx, r_idx]] = QUEEN
        return preds

    def valid(self, preds):
        return np.array([verify_nqueens(p.ravel(), p.shape[0]) for p in preds])

    def choice_stats(self, probs):
        mass = probs.sum(2)
        norm = probs / np.maximum(mass, 1e-12)[..., None]
        unique = (self.num_solutions == 1)[:, None]
        return dict(top=norm.max(2)[self.choice], option=norm.argmax(2)[self.choice], mass=mass[self.choice],
                    active=np.ones(len(self.choice_query), bool), agreed_unique=mass[self.agreed & unique], agreed_multi=mass[self.agreed & ~unique])

    def lex_first(self, preds, valid):
        """(valid multi-solution answers that are the lexicographically first solution, mean 1/#solutions over them)."""
        hits, chance = [], []
        for i in np.nonzero(valid & (self.num_solutions > 1))[0]:
            first = min(tuple((s == QUEEN).argmax(1)) for s in self.solutions[i])
            hits.append(tuple((preds[i] == QUEEN).argmax(1)) == first)
            chance.append(1 / self.num_solutions[i])
        return (float(np.mean(hits)), float(np.mean(chance))) if hits else (None, None)


class Maze:
    token = PATH

    def __init__(self, data_path, grids):
        self.grids = grids
        self.stats = [ShortestPaths(g) for g in grids]
        self.num_solutions = np.array([sp.num_paths for sp in self.stats])
        n = grids.shape[1]
        query, options, exact_top = [], [], []
        forced = np.zeros(grids.shape, bool)
        for i, sp in enumerate(self.stats):
            forced[i] = sp.on_path & (sp.share >= 1 - 1e-9) & (grids[i] == OPEN)
            for cell in zip(*np.nonzero(sp.on_path)):
                nxt = sp._next_cells(cell)
                if len(nxt) > 1:
                    counts = np.array([sp.count_g[c] for c in nxt], dtype=np.float64)
                    query.append(i)
                    options.append([r * n + c for r, c in nxt] + [-1] * (4 - len(nxt)))
                    exact_top.append(counts.max() / counts.sum())
        self.choice_query = np.array(query, dtype=np.int64)
        self.options = np.array(options, dtype=np.int64).reshape(-1, 4)
        self.exact_top = np.array(exact_top)
        self.forced = forced

    def readout(self, grids, probs):
        """Greedy walk from the start: step to the most likely open, unvisited neighbour until next to the goal."""
        preds = grids.copy()
        n = grids.shape[1]
        for grid, p, out in zip(grids, probs, preds):
            passable = (grid != WALL) & (grid != PAD)
            pos = tuple(np.argwhere(grid == START)[0])
            visited = {pos}
            for _ in range(n * n):
                nbrs = [(pos[0] + dr, pos[1] + dc) for dr, dc in MOVES if 0 <= pos[0] + dr < n and 0 <= pos[1] + dc < n]
                if any(grid[c] == GOAL for c in nbrs):
                    break
                opts = [c for c in nbrs if passable[c] and c not in visited]
                if not opts:
                    break
                pos = max(opts, key=lambda c: p[c])
                visited.add(pos)
                out[pos] = PATH
        return preds

    def valid(self, preds):
        q = len(preds)
        return batch_verify_maze(torch.from_numpy(self.grids.reshape(q, -1).astype(np.int64)),
                                 torch.from_numpy(preds.reshape(q, -1).astype(np.int64))).numpy().astype(bool)

    def choice_stats(self, probs):
        flat = probs.reshape(len(probs), -1)
        p = np.where(self.options >= 0, flat[self.choice_query[:, None], self.options.clip(0)], 0.0)
        mass = p.sum(1)
        unique = (self.num_solutions == 1)[:, None, None]
        # A fork off the model's own route gets little mass from a model that picked a route: judge only forks it uses
        return dict(top=p.max(1) / np.maximum(mass, 1e-12), option=p.argmax(1), mass=mass, active=mass >= MIN_FORK_MASS,
                    agreed_unique=probs[self.forced & unique], agreed_multi=probs[self.forced & ~unique])

    def lex_first(self, preds, valid):
        return None, None


def dihedral(x, k, flip, inverse=False):
    """One of the 8 symmetries of the square on the last two axes (or its inverse)."""
    if not inverse:
        x = np.rot90(x, k, axes=(-2, -1))
        return np.flip(x, -1) if flip else x
    x = np.flip(x, -1) if flip else x
    return np.rot90(x, -k, axes=(-2, -1))


SYMMETRIES = [(k, flip) for flip in (False, True) for k in range(4)]


def step_probs(model, inputs, token, model_batch, device, init_noise=0.0, seed=0):
    """Probability of `token` per cell after each supervision step: (steps, N, seq_len), float16."""
    one_of_many = getattr(model, "one_of_many", "naive") != "naive"
    inner = model.model.inner
    original_reset = inner.reset_carry
    generator = torch.Generator(device=device).manual_seed(seed)

    def noisy_reset(flag, carry):
        carry = original_reset(flag, carry)
        m = flag.view(-1, 1, 1)
        for field in ("z_H", "z_L"):
            z = getattr(carry, field)
            noise = torch.randn(z.shape, generator=generator, device=z.device, dtype=torch.float32).to(z.dtype)
            setattr(carry, field, torch.where(m, z + init_noise * noise, z))
        return carry

    if init_noise > 0:
        inner.reset_carry = noisy_reset
    out = []
    try:
        with torch.inference_mode():
            for i in range(0, len(inputs), model_batch):
                x = torch.as_tensor(inputs[i:i + model_batch], device=device, dtype=torch.int32)
                batch = {"inputs": x, "labels": x.clone(), "puzzle_identifiers": torch.zeros(len(x), dtype=torch.int32, device=device)}
                if one_of_many:
                    batch["label_set"] = x.clone().unsqueeze(1)
                with torch.device(device):
                    carry = model.initial_carry(batch)
                steps = []
                while True:
                    carry, _, _, outputs, all_finish = model(carry=carry, batch=batch, return_keys={"logits"})
                    steps.append(torch.softmax(outputs["logits"].float(), -1)[..., token].half().cpu())
                    if all_finish:
                        break
                out.append(torch.stack(steps))
    finally:
        inner.reset_carry = original_reset
    return torch.cat(out, 1).numpy()


def summarize_choices(stats, final=None):
    active = stats["active"]
    top, mass = stats["top"][active].astype(np.float64), stats["mass"].astype(np.float64)
    mean = lambda x: float(x.mean()) if len(x) else None
    both = active & final["active"] if final is not None else None
    return dict(top=mean(top), sure=mean(top >= 0.99), mass=mean(mass), active=mean(active),
                agreed_unique=mean(stats["agreed_unique"].astype(np.float64)),
                agreed_multi=mean(stats["agreed_multi"].astype(np.float64)),
                same_as_final=mean(stats["option"][both] == final["option"][both]) if final is not None else None)


def analyze(task, model, inputs, n, args, device):
    grids = inputs.reshape(-1, n, n)
    multi = task.num_solutions > 1
    pct = lambda x, mask=None: float(x[mask].mean()) if mask is None or mask.any() else None
    result = {}

    # One run, every supervision step
    probs = step_probs(model, inputs, task.token, args.model_batch, device).astype(np.float32)
    num_steps = len(probs)
    final_stats = task.choice_stats(probs[-1].reshape(-1, n, n))
    by_step = {}
    for s in sorted({s for s in REPORT_STEPS if s <= num_steps} | {num_steps}):
        p = probs[s - 1].reshape(-1, n, n)
        valid = task.valid(task.readout(grids, p))
        by_step[s] = dict(valid=pct(valid), valid_unique=pct(valid, ~multi), valid_multi=pct(valid, multi),
                          **summarize_choices(task.choice_stats(p), final_stats))
    result["by_step"] = by_step
    result["exact_top"] = float(task.exact_top.mean()) if len(task.exact_top) else None
    final_preds = task.readout(grids, probs[-1].reshape(-1, n, n))
    final_valid = task.valid(final_preds)
    result["lex_first"], result["lex_first_chance"] = task.lex_first(final_preds, final_valid)
    del probs

    # The 8 symmetries
    views, view_valid = [], []
    for k, flip in SYMMETRIES:
        if (k, flip) == (0, False):
            preds = final_preds
        else:
            g = np.ascontiguousarray(dihedral(grids, k, flip))
            p = step_probs(model, g.reshape(len(g), -1), task.token, args.model_batch, device)[-1].astype(np.float32)
            preds = np.ascontiguousarray(dihedral(task.readout(g, p.reshape(-1, n, n)), k, flip, inverse=True))
        views.append(preds)
        view_valid.append(task.valid(preds))
    result["symmetry"] = ensemble(views, view_valid, final_preds, final_valid, task.num_solutions)

    # Noise on the initial latent
    runs, run_valid = [], []
    for seed in range(args.noise_seeds):
        p = step_probs(model, inputs, task.token, args.model_batch, device, args.init_noise, seed)[-1].astype(np.float32)
        preds = task.readout(grids, p.reshape(-1, n, n))
        runs.append(preds)
        run_valid.append(task.valid(preds))
    result["init_noise"] = dict(sigma=args.init_noise, **ensemble(runs, run_valid, final_preds, final_valid, task.num_solutions))
    return result


def ensemble(answers, valid, reference, reference_valid, num_solutions):
    """Valid rate, agreement with the reference answer, any-valid, majority vote and coverage over several answers."""
    answers, valid = np.stack(answers, 1), np.stack(valid, 1)  # (Q, V, ...), (Q, V)
    q, v = valid.shape
    multi = num_solutions > 1
    keys = answers.reshape(q, v, -1)
    ref = reference.reshape(q, -1)
    same = [(keys[i, j] == ref[i]).all() for i in np.nonzero(multi & reference_valid)[0] for j in range(v) if valid[i, j]]
    majority, distinct = np.zeros(q, bool), np.zeros(q)
    for i in range(q):
        counts = Counter(keys[i, j].tobytes() for j in range(v))
        winner = counts.most_common(1)[0][0]
        majority[i] = valid[i, [keys[i, j].tobytes() == winner for j in range(v)]].any()
        distinct[i] = len({keys[i, j].tobytes() for j in range(v) if valid[i, j]})
    return dict(views=v, valid=float(valid.mean()), any_valid=float(valid.any(1).mean()), majority=float(majority.mean()),
                same_as_reference_multi=float(np.mean(same)) if same else None,
                distinct_valid_multi=float(distinct[multi].mean()) if multi.any() else None,
                coverage_multi=float((distinct[multi] / num_solutions[multi]).mean()) if multi.any() else None,
                any_valid_unique=float(valid[~multi].any(1).mean()) if (~multi).any() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("nqueens", "maze"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="name=checkpoint pairs")
    parser.add_argument("--init_noise", type=float, default=1.0, help="Std of the noise added to the initial latent")
    parser.add_argument("--noise_seeds", type=int, default=4)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--model_batch", type=int, default=4096)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data, metadata = load_split(args.data)
    inputs = np.array(data["inputs"][:args.max_queries], dtype=np.int64)
    n = int(round(np.sqrt(inputs.shape[1])))
    task = (NQueens if args.task == "nqueens" else Maze)(args.data, inputs.reshape(-1, n, n))
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for spec in args.runs:
        name, checkpoint = spec.split("=", 1)
        model, step = load_model(checkpoint, metadata, 0.0, device)
        results[name] = {"checkpoint": checkpoint, "step": step, **analyze(task, model, inputs, n, args, device)}
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(results[name], f, indent=2)
        last = results[name]["by_step"][max(results[name]["by_step"])]
        print(f"{name}: one-pass readout {100 * last['valid']:.1f}% valid, normalized top at choice points "
              f"{last['top']}, symmetry any-valid {100 * results[name]['symmetry']['any_valid']:.1f}%", flush=True)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    f = lambda x, pct=False: "-" if x is None else f"{100 * x:.1f}" if pct else f"{x:.3f}"
    exact = next(iter(results.values()))["exact_top"]
    lines = [f"# Blend or tie-break? ({args.task}, {len(inputs)} test inputs)", "",
             "One model run per input, read out without re-running: each row's top column (N-Queens) or the greedy walk "
             "(mazes). Choice points: rows the solutions disagree on / forks of the shortest paths. Normalized top: the "
             f"model's largest probability among the options over their sum; a blend of the solutions would give the exact "
             f"top, here {f(exact)} on average; ~1 means one option chosen. Mass: total probability on the options. For mazes, normalized top and same-as-final count only forks with "
             f"mass >= {MIN_FORK_MASS} (forks on the model's own route; 'forks used' gives their share). "
             "Agreed: probability on cells every solution shares. Same as final: the choice point's top option equals "
             "the last step's.", "",
             "## By supervision step", "",
             "| run | step | valid (%) | valid, one solution | valid, several | normalized top | choice points >= 0.99 (%) | "
             "mass on options | forks used (%) | agreed cells, one solution | agreed cells, several | same option as final (%) |",
             "|" + "---|" * 12]
    for nm, r in results.items():
        for s, b in r["by_step"].items():
            lines.append(f"| {nm} | {s} | {f(b['valid'], True)} | {f(b['valid_unique'], True)} | {f(b['valid_multi'], True)} | "
                         f"{f(b['top'])} | {f(b['sure'], True)} | {f(b['mass'])} | {f(b['active'], True)} | {f(b['agreed_unique'])} | "
                         f"{f(b['agreed_multi'])} | {f(b['same_as_final'], True)} |")
    lines += ["", "## Symmetric views and initial-latent noise", "",
              "Over the 8 symmetries (decoded on the transformed input, mapped back) and over noise seeds on the initial "
              "latent. Same answer: share of valid answers on multi-solution inputs equal to the untransformed / "
              "noise-free answer (when that is valid). Distinct and coverage: valid distinct answers on multi-solution inputs.", "",
              "| run | variation | valid (%) | same answer, several solutions (%) | any valid (%) | any valid, one solution (%) | "
              "majority (%) | distinct valid, several | coverage, several (%) |",
              "|" + "---|" * 9]
    for nm, r in results.items():
        for label, e in ((f"8 symmetries", r["symmetry"]), (f"{r['init_noise']['views']} init-noise seeds, sigma {r['init_noise']['sigma']}", r["init_noise"])):
            lines.append(f"| {nm} | {label} | {f(e['valid'], True)} | {f(e['same_as_reference_multi'], True)} | "
                         f"{f(e['any_valid'], True)} | {f(e['any_valid_unique'], True)} | {f(e['majority'], True)} | "
                         f"{f(e['distinct_valid_multi'])} | {f(e['coverage_multi'], True)} |")
    if args.task == "nqueens":
        lines += ["", "## Lexicographically first solution", "",
                  "Valid multi-solution answers (no symmetry, no noise) that are the first solution in column order, row by row.", "",
                  "| run | lex-first (%) | chance (%) |", "|---|---|---|"]
        for nm, r in results.items():
            lines.append(f"| {nm} | {f(r['lex_first'], True)} | {f(r['lex_first_chance'], True)} |")
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
