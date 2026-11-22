"""Light up (akari) on a square grid, in the token encoding of dataset/build_lightup_multi_dataset.py.

Input token per cell (row by row): WHITE, WALL (a wall without a number) or WALL0 + n for a wall numbered n (0..4).
Answer tokens: NO_BULB or BULB (walls take NO_BULB). 0 is padding and the ignore id.

Rules: bulbs go on white cells; a bulb lights its cell and every white cell it sees along its row and column, up to
the nearest wall; every white cell is lit; no bulb sees another; a numbered wall has exactly that many bulbs on its
four sides.
"""
import numpy as np

WHITE = 1
WALL = 2
WALL0 = 3                # WALL0 + n for a wall numbered n, n = 0..4
NO_BULB = WALL0 + 5      # 8
BULB = NO_BULB + 1       # 9
VOCAB_SIZE = BULB + 1    # 10


def encode(grid: np.ndarray) -> np.ndarray:
    """grid: (n, n) ints, -2 = white, -1 = wall without a number, 0..4 = numbered wall. Returns the input tokens."""
    out = np.where(grid >= 0, WALL0 + grid, np.where(grid == -1, WALL, WHITE))
    return out.reshape(-1).astype(np.uint8)


def decode(tokens: np.ndarray) -> np.ndarray:
    """Input tokens -> (n, n) grid in the format of encode."""
    n = int(round(np.sqrt(len(tokens))))
    t = tokens.astype(np.int64).reshape(n, n)
    return np.where(t >= WALL0, t - WALL0, np.where(t == WALL, -1, -2))


def segments(white: np.ndarray):
    """(n, n) ids of each white cell's row segment and column segment (runs of white cells between walls); -1 on walls."""
    n, m = white.shape
    out = []
    for grid in (white, white.T):
        seg = -np.ones(grid.shape, dtype=np.int64)
        k = -1
        for r in range(grid.shape[0]):
            prev = False
            for c in range(grid.shape[1]):
                if grid[r, c]:
                    if not prev:
                        k += 1
                    seg[r, c] = k
                prev = grid[r, c]
        out.append(seg)
    return out[0], out[1].T


def verify_lightup(inputs: np.ndarray, pred: np.ndarray, decoded=None) -> bool:
    """inputs, pred: (n * n,) tokens. decoded: decode(inputs), if already known."""
    n = int(round(np.sqrt(len(inputs))))
    if not np.isin(pred, (NO_BULB, BULB)).all():
        return False
    grid = decoded if decoded is not None else decode(inputs)
    white = grid == -2
    b = pred.reshape(n, n) == BULB
    if (b & ~white).any():
        return False
    row, col = segments(white)
    row_bulbs = np.bincount(row[b], minlength=row.max() + 1)
    col_bulbs = np.bincount(col[b], minlength=col.max() + 1)
    if (row_bulbs > 1).any() or (col_bulbs > 1).any():
        return False
    if not ((row_bulbs[row[white]] > 0) | (col_bulbs[col[white]] > 0)).all():
        return False
    padded = np.pad(b, 1)
    sides = padded[:-2, 1:-1].astype(int) + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    numbered = grid >= 0
    return bool((sides[numbered] == grid[numbered]).all())
