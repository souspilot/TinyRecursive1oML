"""
Graph Colouring with Many Solutions

Each input is a random graph on --num-nodes nodes that has a --num-colors colouring: a colouring is planted (a
random colour per node) and each pair of differently coloured nodes is joined with probability p, drawn uniformly
from [--edge-prob-min, --edge-prob-max]. Sparser graphs have more colourings, so p is the ambiguity dial.

Solutions are colourings in canonical form (colours numbered in order of first appearance, as in GRAM and
evaluators/graph_coloring_utils.py), so renaming colours never makes a new solution: every solution differs from
the others in which nodes share a colour. All are enumerated exactly. Inputs are drawn to a target distribution of
solution counts (--unique-share unique, the rest spread over log-spaced groups up to the most a graph can have),
and each stores up to --max-stored-solutions; see dataset/multi_solution.py for the four datasets written and the
held-out solutions on training inputs.

Encoding as in dataset/build_graph_coloring_dataset.py: the input is the upper triangle of the adjacency matrix
(diagonal 0, no edge 1, edge 2), the label holds 3 + colour in the first num_nodes positions and the ignore id
elsewhere. Per input, num_edges is saved too.

Usage:
    python dataset/build_graph_coloring_multi_dataset.py --num-train 100000 --num-test 2000 --output-dir data/graph-coloring-multi
"""
import os
import sys
from functools import lru_cache

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import multi_solution  # noqa: E402

cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str = "data/graph-coloring-multi"
    num_nodes: int = 10
    num_colors: int = 3
    num_train: int = 100000
    num_test: int = 2000
    edge_prob_min: float = 0.1
    edge_prob_max: float = 1.0
    unique_share: float = 1 / 3
    max_stored_solutions: int = 16
    seed: int = 42


@lru_cache(maxsize=None)
def canonical_colorings(n: int, k: int) -> np.ndarray:
    """Every colouring of n nodes with at most k colours in canonical form, as an (N, n) uint8 array."""
    rows = [[0]]
    for _ in range(n - 1):
        rows = [r + [c] for r in rows for c in range(min(max(r) + 2, k))]
    return np.array(rows, dtype=np.uint8)


def bounds_for(n: int, k: int):
    """Inclusive upper bounds of the solution-count groups: 1, 2-3, 4-15, then x4 up to the most possible."""
    most = len(canonical_colorings(n, k))
    bounds = [1, 3, 15]
    while bounds[-1] < most:
        bounds.append(min(4 * bounds[-1] + 3, most))
    return tuple(bounds)


def encode_adjacency(adj: np.ndarray) -> np.ndarray:
    rows, cols = np.triu_indices(len(adj))
    return np.where(rows == cols, 0, np.where(adj[rows, cols], 2, 1)).astype(np.uint8)


def generate_chunk(args):
    seed, config, num_candidates, needed = args
    config = DataProcessConfig(**config)  # passed as a dict: argdantic's wrapped config class can't be pickled
    rng = np.random.default_rng(seed)
    n, k = config.num_nodes, config.num_colors
    table = canonical_colorings(n, k)
    bounds = bounds_for(n, k)
    seq_len = n * (n + 1) // 2
    iu, ju = np.triu_indices(n, 1)
    out = []
    for _ in range(num_candidates):
        planted = rng.integers(k, size=n)
        p = rng.uniform(config.edge_prob_min, config.edge_prob_max)
        edge = (planted[iu] != planted[ju]) & (rng.random(len(iu)) < p)
        adj = np.zeros((n, n), dtype=bool)
        adj[iu[edge], ju[edge]] = True
        adj = adj | adj.T
        proper = ~(table[:, iu[edge]] == table[:, ju[edge]]).any(1)
        solutions = table[proper]
        if not needed[multi_solution.count_group(len(solutions), bounds)]:
            continue
        order = rng.permutation(len(solutions))[:config.max_stored_solutions]
        labels = []
        for i in order:
            label = np.zeros(seq_len, dtype=np.uint8)
            label[:n] = solutions[i] + 3
            labels.append(label)
        out.append(dict(inputs=encode_adjacency(adj), labels=labels, num_solutions=len(solutions),
                        num_edges=int(edge.sum())))
    return out


@cli.command(singleton=True)
def build_graph_coloring_multi_dataset(config: DataProcessConfig):
    rng = np.random.default_rng(config.seed)
    bounds = bounds_for(config.num_nodes, config.num_colors)
    fields = {k: getattr(config, k) for k in DataProcessConfig.model_fields}
    seen = set()
    test = multi_solution.draw(generate_chunk, fields, multi_solution.quotas_for(config.num_test, config.unique_share, len(bounds)),
                               bounds, rng, seen, chunk=1000, desc="Drawing test graphs")
    train = multi_solution.draw(generate_chunk, fields, multi_solution.quotas_for(config.num_train, config.unique_share, len(bounds)),
                                bounds, rng, seen, chunk=1000, desc="Drawing train graphs")
    multi_solution.describe("test", test, bounds, ["num_edges"])
    multi_solution.describe("train", train, bounds, ["num_edges"])
    n = config.num_nodes
    multi_solution.write_variants(config.output_dir, train, test, rng, seq_len=n * (n + 1) // 2,
                                  vocab_size=3 + config.num_colors, max_stored=config.max_stored_solutions,
                                  extra_fields=["num_edges"])


if __name__ == "__main__":
    cli()
