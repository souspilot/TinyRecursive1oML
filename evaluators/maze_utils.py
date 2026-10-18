"""Maze utilities: token encoding, exact shortest-path statistics, and verification.

Grid tokens: 0 pad, 1 wall, 2 open, 3 start, 4 goal, 5 path (labels and predictions only).
A prediction is correct when it marks exactly the cells of one shortest start-goal path and leaves every other
cell as in the input. Mazes can have many shortest paths; their number, and the share passing through each
cell, are counted exactly in BFS layers without listing the paths.
"""
from collections import deque

import numpy as np
import torch

PAD, WALL, OPEN, START, GOAL, PATH = range(6)


def bfs_counts(passable: np.ndarray, source):
    """Distance from source to every cell (-1 if unreachable) and the number of shortest paths to it."""
    h, w = passable.shape
    dist = np.full((h, w), -1, dtype=np.int64)
    count = np.zeros((h, w), dtype=np.int64)
    dist[source], count[source] = 0, 1
    queue = deque([source])
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc]:
                if dist[nr, nc] < 0:
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))
                if dist[nr, nc] == dist[r, c] + 1:
                    count[nr, nc] += count[r, c]
    return dist, count


class ShortestPaths:
    """Exact statistics of the shortest start-goal paths of one maze, given as an (n, n) token grid."""

    def __init__(self, grid: np.ndarray):
        self.grid = grid
        passable = (grid != WALL) & (grid != PAD)
        self.start = tuple(int(x) for x in np.argwhere(grid == START)[0])
        self.goal = tuple(int(x) for x in np.argwhere(grid == GOAL)[0])
        self.dist_s, self.count_s = bfs_counts(passable, self.start)
        self.dist_g, self.count_g = bfs_counts(passable, self.goal)
        self.length = int(self.dist_s[self.goal])  # in steps; -1 if the goal is unreachable
        self.num_paths = int(self.count_s[self.goal])
        self.on_path = (self.dist_s >= 0) & (self.dist_g >= 0) & (self.dist_s + self.dist_g == self.length)
        # Share of shortest paths passing through each cell
        self.share = np.where(self.on_path, self.count_s * self.count_g / max(self.num_paths, 1), 0.0)

    def _next_cells(self, cell):
        r, c = cell
        d = self.dist_s[r, c]
        return [(nr, nc) for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                if 0 <= nr < self.grid.shape[0] and 0 <= nc < self.grid.shape[1]
                and self.on_path[nr, nc] and self.dist_s[nr, nc] == d + 1]

    def sample_path(self, rng: np.random.Generator):
        """A shortest path drawn uniformly at random, as a list of cells from start to goal."""
        path = [self.start]
        while path[-1] != self.goal:
            options = self._next_cells(path[-1])
            weights = np.array([self.count_g[cell] for cell in options], dtype=np.float64)
            path.append(options[rng.choice(len(options), p=weights / weights.sum())])
        return path

    def all_paths(self):
        """Every shortest path (only sensible when num_paths is small)."""
        paths, stack = [], [[self.start]]
        while stack:
            path = stack.pop()
            if path[-1] == self.goal:
                paths.append(path)
            else:
                stack.extend(path + [cell] for cell in self._next_cells(path[-1]))
        return paths

    def max_weight_path(self, weights: np.ndarray) -> float:
        """Largest total weight of the cells on any one shortest path (dynamic programming over BFS layers)."""
        best = np.full(self.grid.shape, -np.inf)
        order = sorted(zip(*np.where(self.on_path)), key=lambda cell: -self.dist_s[cell])
        for cell in order:
            if cell == self.goal:
                best[cell] = weights[cell]
            else:
                best[cell] = weights[cell] + max(best[nxt] for nxt in self._next_cells(cell))
        return float(best[self.start])

    def label(self, path) -> np.ndarray:
        """The label grid for a path: its cells between start and goal marked PATH."""
        label = self.grid.copy()
        for cell in path[1:-1]:
            label[cell] = PATH
        return label


def _spread(mask: torch.Tensor) -> torch.Tensor:
    """Cells 4-adjacent to any cell of mask."""
    out = torch.zeros_like(mask)
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


def _reachable(source: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Cells reachable from source moving through allowed cells (flood fill)."""
    reached = source & allowed
    while True:
        grown = reached | (_spread(reached) & allowed)
        if torch.equal(grown, reached):
            return reached
        reached = grown


def shortest_lengths(grids: torch.Tensor) -> torch.Tensor:
    """Shortest start-goal path length in steps for a batch of (B, n, n) token grids; -1 if none."""
    passable = (grids != WALL) & (grids != PAD)
    goal = grids == GOAL
    frontier = reached = grids == START
    length = torch.full((grids.shape[0],), -1, dtype=torch.long, device=grids.device)
    steps = 0
    while frontier.any():
        found = (frontier & goal).any((1, 2)) & (length < 0)
        length[found] = steps
        frontier = _spread(frontier) & passable & ~reached
        reached = reached | frontier
        steps += 1
    return length


@torch.compiler.disable
def batch_verify_maze(inputs: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """Whether each prediction marks exactly one shortest start-goal path. Returns a (B,) bool tensor.

    Correct iff every non-open cell is unchanged, open cells are open or PATH, the number of PATH cells is the
    shortest length minus one, and start reaches goal through PATH cells. A connected set of that size holding
    start and goal can only be the cells of a shortest path."""
    b = inputs.shape[0]
    n = int(round(inputs.shape[-1] ** 0.5))
    grids, preds = inputs.view(b, n, n), predictions.view(b, n, n)
    path = preds == PATH
    unchanged = torch.where(grids == OPEN, (preds == OPEN) | path, preds == grids).all((1, 2))
    length = shortest_lengths(grids)
    right_size = path.sum((1, 2)) == length - 1
    connected = (_reachable(grids == START, path | (grids == START) | (grids == GOAL)) & (grids == GOAL)).any((1, 2))
    return unchanged & right_size & connected & (length > 0)
