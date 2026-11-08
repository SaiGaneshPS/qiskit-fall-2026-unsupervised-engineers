"""Learned Ising surrogates of the Marxan landscape.

The quantum proposal explores whatever landscape it is given. Acceptance always
uses the true Marxan energy, so the surrogate never affects correctness, only
how good the proposals are.
"""
from __future__ import annotations

import numpy as np


def all_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def snake_pairs(n_cols: int) -> list[tuple[int, int]]:
    """Hamiltonian path through a 2 x n_cols ladder (row-major indices).

    Visits (0,0),(1,0),(1,1),(0,1),(0,2),(1,2),... so every rung is kept plus every
    other horizontal edge. On hardware this is a line of qubits: no SWAPs needed.
    """
    order = []
    for c in range(n_cols):
        col = [c, n_cols + c] if c % 2 == 0 else [n_cols + c, c]
        order.extend(col)
    return [tuple(sorted((order[k], order[k + 1]))) for k in range(len(order) - 1)], order


def best_line_pairs(J_full: np.ndarray, exhaustive_max: int = 10, rng: np.random.Generator | None = None):
    """Order the units along a line of qubits to keep the strongest learned couplings.

    The qubit layout is free, so the 1D chain need not follow the map grid. We pick the
    Hamiltonian path maximising sum |J_ij| over consecutive units (exact for n <= 10,
    greedy + 2-opt otherwise). Returns (pairs, order).
    """
    from itertools import permutations
    W = np.abs(np.triu(J_full, 1))
    W = W + W.T
    n = W.shape[0]

    def score(order):
        return sum(W[order[k], order[k + 1]] for k in range(n - 1))

    if n <= exhaustive_max:
        best = max((p for p in permutations(range(n)) if p[0] < p[-1]), key=score)
    else:
        rng = rng or np.random.default_rng(0)
        best, best_s = None, -1.0
        for start in range(n):
            order = [start]
            while len(order) < n:
                order.append(max((v for v in range(n) if v not in order), key=lambda v: W[order[-1], v]))
            improved = True
            while improved:
                improved = False
                for i in range(1, n - 1):
                    for j in range(i + 1, n):
                        cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                        if score(cand) > score(order) + 1e-12:
                            order, improved = cand, True
            if score(order) > best_s:
                best, best_s = order, score(order)
    best = list(best)
    return [tuple(sorted((best[k], best[k + 1]))) for k in range(n - 1)], best


def fit_ising(x: np.ndarray, E: np.ndarray, weights: np.ndarray | None = None,
              pairs: list[tuple[int, int]] | None = None, ridge: float = 1e-6):
    """Weighted least squares E ~ c + sum h_i z_i + sum_{(i,j) in pairs} J_ij z_i z_j.

    x: (m, n) 0/1 configurations, z = 1 - 2x. Returns (h, J, c, fitted).
    """
    x = np.asarray(x)
    m, n = x.shape
    pairs = all_pairs(n) if pairs is None else pairs
    z = 1 - 2 * x.astype(float)
    cols = [np.ones(m), *z.T] + [z[:, i] * z[:, j] for i, j in pairs]
    X = np.column_stack(cols)
    w = np.ones(m) if weights is None else np.asarray(weights, float)
    sw = np.sqrt(w / w.max())
    A = X * sw[:, None]
    b = E * sw
    coef = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ b)
    h = coef[1:n + 1]
    J = np.zeros((n, n))
    for c_, (i, j) in zip(coef[n + 1:], pairs):
        J[i, j] = c_
    return h, J, coef[0], X @ coef


def boltzmann_weights(E: np.ndarray, T_fit: float) -> np.ndarray:
    return np.exp(-(E - E.min()) / T_fit)
