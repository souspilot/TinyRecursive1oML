"""Heyawake on a square grid, in the token encoding of dataset/build_heyawake_multi_dataset.py.

Input token per cell (row by row): 1 + wall * CLUE_SLOTS + slot, where wall = 2 * (wall on the right) + (wall below),
counting the grid edge as a wall, and slot = 0 without a room number or 1 + number on the room's first cell (in
reading order). Answer tokens: UNSHADED or SHADED. 0 is padding and the ignore id.

Rules: shaded cells never touch side by side; a room with a number has exactly that many shaded cells; the unshaded
cells form one connected area; and no straight line of unshaded cells crosses more than one room border.
"""
import numpy as np

MAX_CLUE = 33
CLUE_SLOTS = MAX_CLUE + 2           # none, 0..MAX_CLUE
UNSHADED = 1 + 4 * CLUE_SLOTS       # 141
SHADED = UNSHADED + 1               # 142
VOCAB_SIZE = SHADED + 1             # 143


def encode(room: np.ndarray, clues: dict) -> np.ndarray:
    """room: (n, n) room ids; clues: {room id: number}. Returns the (n * n,) input tokens."""
    n = room.shape[0]
    right = np.ones((n, n), dtype=np.int64); right[:, :-1] = room[:, :-1] != room[:, 1:]
    down = np.ones((n, n), dtype=np.int64); down[:-1, :] = room[:-1, :] != room[1:, :]
    slot = np.zeros((n, n), dtype=np.int64)
    flat = room.reshape(-1)
    for rid, v in clues.items():
        first = int(np.argmax(flat == rid))
        slot.flat[first] = 1 + v
    return (1 + (2 * right + down) * CLUE_SLOTS + slot).reshape(-1).astype(np.uint8)


def decode(tokens: np.ndarray):
    """Input tokens -> (room ids (n, n), {room id: number})."""
    n = int(round(np.sqrt(len(tokens))))
    t = tokens.astype(np.int64).reshape(n, n) - 1
    wall, slot = t // CLUE_SLOTS, t % CLUE_SLOTS
    right, down = wall // 2, wall % 2
    room = -np.ones((n, n), dtype=np.int64)
    clues = {}
    rid = 0
    for r in range(n):
        for c in range(n):
            if room[r, c] >= 0:
                continue
            room[r, c] = rid
            stack = [(r, c)]
            while stack:
                y, x = stack.pop()
                for ny, nx, open_ in ((y, x + 1, x + 1 < n and not right[y, x]), (y, x - 1, x > 0 and not right[y, x - 1]),
                                      (y + 1, x, y + 1 < n and not down[y, x]), (y - 1, x, y > 0 and not down[y - 1, x])):
                    if open_ and room[ny, nx] < 0:
                        room[ny, nx] = rid
                        stack.append((ny, nx))
            if slot[r, c]:
                clues[rid] = int(slot[r, c]) - 1
            rid += 1
    return room, clues


def verify_heyawake(inputs: np.ndarray, pred: np.ndarray, decoded=None) -> bool:
    """inputs, pred: (n * n,) tokens. decoded: decode(inputs), if already known."""
    n = int(round(np.sqrt(len(inputs))))
    if not np.isin(pred, (UNSHADED, SHADED)).all():
        return False
    room, clues = decoded if decoded is not None else decode(inputs)
    s = pred.reshape(n, n) == SHADED
    if (s[:, :-1] & s[:, 1:]).any() or (s[:-1, :] & s[1:, :]).any():
        return False
    counts = np.bincount(room[s], minlength=room.max() + 1)
    if any(counts[rid] != v for rid, v in clues.items()):
        return False
    # Unshaded cells connected
    u = ~s
    start = np.argwhere(u)
    if len(start) == 0:
        return False
    seen = np.zeros_like(u)
    seen[tuple(start[0])] = True
    stack = [tuple(start[0])]
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y, x + 1), (y, x - 1), (y + 1, x), (y - 1, x)):
            if 0 <= ny < n and 0 <= nx < n and u[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    if (u & ~seen).any():
        return False
    # Straight unshaded lines cross at most one room border
    for grid, rooms in ((u, room), (u.T, room.T)):
        for line, ids in zip(grid, rooms):
            crossings = 0
            for i in range(1, n):
                if not line[i]:
                    crossings = 0
                elif line[i - 1]:
                    crossings += ids[i] != ids[i - 1]
                    if crossings >= 2:
                        return False
    return True
