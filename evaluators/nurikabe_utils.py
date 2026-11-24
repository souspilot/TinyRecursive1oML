"""Nurikabe on a square grid, in the token encoding of dataset/build_nurikabe_multi_dataset.py.

Input token per cell (row by row): EMPTY, UNKNOWN (a number whose value is not given, "?") or NUMBER0 + n for the
number n. Answer tokens: UNSHADED or SHADED. 0 is padding and the ignore id.

Rules: numbered cells are unshaded; the shaded cells form one connected area with no 2 x 2 block; every connected area
of unshaded cells (an island) holds exactly one number, and has as many cells as that number ("?" allows any size).
"""
import numpy as np

MAX_NUMBER = 60
EMPTY = 1
UNKNOWN = 2
NUMBER0 = 3                         # NUMBER0 + n for the number n, n = 1..MAX_NUMBER
UNSHADED = NUMBER0 + MAX_NUMBER + 1  # 64
SHADED = UNSHADED + 1                # 65
VOCAB_SIZE = SHADED + 1              # 66


def encode(numbers: np.ndarray) -> np.ndarray:
    """numbers: (n, n) ints, 0 = empty cell, -1 = "?", n >= 1 = number. Returns the (n * n,) input tokens."""
    out = np.where(numbers > 0, NUMBER0 + numbers, np.where(numbers < 0, UNKNOWN, EMPTY))
    return out.reshape(-1).astype(np.uint8)


def decode(tokens: np.ndarray) -> np.ndarray:
    """Input tokens -> (n, n) numbers in the format of encode."""
    n = int(round(np.sqrt(len(tokens))))
    t = tokens.astype(np.int64).reshape(n, n)
    return np.where(t >= NUMBER0, t - NUMBER0, np.where(t == UNKNOWN, -1, 0))


def _components(mask: np.ndarray):
    """Labels of the 4-connected components of mask (-1 outside it), and their count."""
    n, m = mask.shape
    label = -np.ones(mask.shape, dtype=np.int64)
    count = 0
    for r, c in zip(*np.nonzero(mask)):
        if label[r, c] >= 0:
            continue
        label[r, c] = count
        stack = [(r, c)]
        while stack:
            y, x = stack.pop()
            for ny, nx in ((y, x + 1), (y, x - 1), (y + 1, x), (y - 1, x)):
                if 0 <= ny < n and 0 <= nx < m and mask[ny, nx] and label[ny, nx] < 0:
                    label[ny, nx] = count
                    stack.append((ny, nx))
        count += 1
    return label, count


def verify_nurikabe(inputs: np.ndarray, pred: np.ndarray, decoded=None) -> bool:
    """inputs, pred: (n * n,) tokens. decoded: decode(inputs), if already known."""
    n = int(round(np.sqrt(len(inputs))))
    if not np.isin(pred, (UNSHADED, SHADED)).all():
        return False
    numbers = decoded if decoded is not None else decode(inputs)
    s = pred.reshape(n, n) == SHADED
    if (s & (numbers != 0)).any():
        return False
    if (s[:-1, :-1] & s[1:, :-1] & s[:-1, 1:] & s[1:, 1:]).any():
        return False
    if _components(s)[1] > 1:
        return False
    label, count = _components(~s)
    clue_label = label[numbers != 0]
    if len(clue_label) != count or len(np.unique(clue_label)) != count:
        return False  # an island without a number, or with two
    sizes = np.bincount(label[~s], minlength=count)
    return all(v < 0 or sizes[label[r, c]] == v for (r, c), v in np.ndenumerate(numbers) if v != 0)
