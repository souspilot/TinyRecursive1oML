"""Random 3-SAT as a token sequence (dataset/build_sat_dataset.py), and exact solution enumeration.

A formula over n variables with m clauses is one sequence of n + 3 * max_clauses tokens:
  positions 0..n-1      one slot per variable: FREE(i) if unassigned, POS(i) / NEG(i) if given true / false
  then 3 per clause     the clause's literals POS(j) / NEG(j), then PAD up to max_clauses clauses
The label holds TRUE / FALSE in the variable slots and the ignore id (PAD) elsewhere.

Tokens: 0 = PAD (also the ignore label), 1 = FALSE, 2 = TRUE, POS(i) = 3 + i, NEG(i) = 3 + n + i,
FREE(i) = 3 + 2n + i, so vocab_size = 3 + 3n and num_vars(vocab_size) recovers n.
"""
import numpy as np

PAD, FALSE, TRUE = 0, 1, 2


def num_vars(vocab_size: int) -> int:
    return (vocab_size - 3) // 3


def pos(i, n):
    return 3 + np.asarray(i)


def neg(i, n):
    return 3 + n + np.asarray(i)


def free(i, n):
    return 3 + 2 * n + np.asarray(i)


def encode(variables: np.ndarray, negated: np.ndarray, givens: dict, n: int, max_clauses: int) -> np.ndarray:
    """variables, negated: (m, 3) clause literals; givens: {variable: bool}. Returns the input sequence."""
    seq = np.zeros(n + 3 * max_clauses, dtype=np.uint8)
    seq[:n] = free(np.arange(n), n)
    for i, value in givens.items():
        seq[i] = pos(i, n) if value else neg(i, n)
    m = len(variables)
    seq[n:n + 3 * m] = np.where(negated, neg(variables, n), pos(variables, n)).ravel()
    return seq


def label(assignment: np.ndarray, n: int, max_clauses: int) -> np.ndarray:
    """assignment: (n,) bool. Returns the label sequence."""
    seq = np.zeros(n + 3 * max_clauses, dtype=np.uint8)
    seq[:n] = np.where(assignment, TRUE, FALSE)
    return seq


def decode(inputs: np.ndarray, n: int):
    """Inverse of encode for one sequence: (variables, negated, givens), givens as an (n,) int array of -1/0/1."""
    lits = inputs[n:].reshape(-1, 3).astype(np.int64)
    lits = lits[(lits != PAD).all(1)]
    negated = lits >= 3 + n
    variables = np.where(negated, lits - 3 - n, lits - 3)
    slots = inputs[:n].astype(np.int64)
    givens = np.where(slots < 3 + n, 1, np.where(slots < 3 + 2 * n, 0, -1))
    return variables, negated, givens


def verify(inputs: np.ndarray, preds: np.ndarray, n: int) -> bool:
    """Whether preds assigns every variable, agrees with the givens and satisfies every clause."""
    values = preds[:n]
    if not np.isin(values, (FALSE, TRUE)).all():
        return False
    assignment = values == TRUE
    variables, negated, givens = decode(inputs, n)
    if ((givens >= 0) & (assignment != (givens == 1))).any():
        return False
    return bool((assignment[variables] != negated).any(1).all())


def batch_verify(inputs: np.ndarray, preds: np.ndarray, n: int) -> np.ndarray:
    return np.array([verify(x, p, n) for x, p in zip(inputs, preds)])


class Assignments:
    """Every assignment of n variables (n >= 6), bit-packed: words[v] has bit k set iff assignment k sets v true."""

    def __init__(self, n: int):
        assert n >= 6, "bit-packing needs at least 64 assignments"
        self.n = n
        idx = np.arange(2 ** n, dtype=np.uint64)
        bits = ((idx[None, :] >> np.arange(n, dtype=np.uint64)[:, None]) & 1).astype(bool)
        self.words = np.packbits(bits, axis=1, bitorder="little").view(np.uint64)  # (n, 2^n / 64)
        self.bits = bits

    def solutions(self, variables: np.ndarray, negated: np.ndarray, givens: dict = None) -> np.ndarray:
        """Indices k of the satisfying assignments (bit v of k = value of variable v), in increasing order."""
        ones = np.uint64(0xFFFFFFFFFFFFFFFF)
        sat = np.full(self.words.shape[1], ones, dtype=np.uint64)
        flips = np.where(negated, ones, np.uint64(0))
        for (a, b, c), (fa, fb, fc) in zip(variables, flips):
            sat &= (self.words[a] ^ fa) | (self.words[b] ^ fb) | (self.words[c] ^ fc)
        for i, value in (givens or {}).items():
            sat &= self.words[i] if value else ~self.words[i]
        mask = np.unpackbits(sat.view(np.uint8), bitorder="little").astype(bool)
        return np.flatnonzero(mask)

    def values(self, solutions: np.ndarray) -> np.ndarray:
        """(len(solutions), n) bool assignments."""
        return self.bits[:, solutions].T
