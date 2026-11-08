"""Proposal kernels, Metropolis transition matrices and spectral gaps.

Q[a, b] = probability of proposing configuration b from configuration a.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


# --------------------------------------------------------------------------- proposals
def local_proposal(n: int) -> np.ndarray:
    N = 2 ** n
    Q = np.zeros((N, N))
    k = np.arange(N)
    for i in range(n):
        Q[k, k ^ (1 << i)] = 1.0 / n
    return Q


def uniform_proposal(n: int) -> np.ndarray:
    N = 2 ** n
    return np.full((N, N), 1.0 / N)


def mixer_alpha(h: np.ndarray, J: np.ndarray) -> float:
    """Layden et al. normalisation alpha = ||H_mix||_F / ||H_prob||_F for H_mix = sum X."""
    n = len(h)
    return np.sqrt(n) / np.sqrt(np.sum(np.triu(J, 1) ** 2) + np.sum(h ** 2))


def _mixer_matrix(n: int) -> np.ndarray:
    N = 2 ** n
    M = np.zeros((N, N))
    k = np.arange(N)
    for i in range(n):
        M[k, k ^ (1 << i)] = 1.0
    return M


def quantum_proposal_exact(h, J, gammas, times, diag=None) -> np.ndarray:
    """Average of |<b|exp(-iHt)|a>|^2 over the listed (gamma, t) pairs.

    H = (1 - gamma) * alpha * H_prob + gamma * sum_j X_j, H_prob = diag (Ising surrogate).
    `gammas` and `times` are equal-length sequences; each pair is equally likely.
    """
    n = len(h)
    alpha = mixer_alpha(h, J)
    Hp = ising_diag(h, J) if diag is None else diag
    Hm = _mixer_matrix(n)
    Q = np.zeros((2 ** n, 2 ** n))
    pairs = list(zip(gammas, times))
    by_gamma: dict[float, list[float]] = {}
    for g, t in pairs:
        by_gamma.setdefault(float(g), []).append(float(t))
    for g, ts in by_gamma.items():
        H = (1 - g) * alpha * np.diag(Hp) + g * Hm
        w, V = eigh(H)
        Vf = V.astype(np.float32)
        for t in ts:
            # U = V e^{-iwt} V^T = A - iB with real A, B, so |U|^2 = A^2 + B^2
            A = (Vf * np.cos(w * t).astype(np.float32)) @ Vf.T
            B = (Vf * np.sin(w * t).astype(np.float32)) @ Vf.T
            Q += (A.astype(np.float64) ** 2 + B.astype(np.float64) ** 2)
    Q /= len(pairs)
    Q /= Q.sum(axis=1, keepdims=True)  # remove float32 round-off
    return 0.5 * (Q + Q.T)  # U is symmetric, so Q is too (symmetrise round-off)


def ising_energy_vector(h: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Diagonal of H = sum h_i Z_i + sum_{i<j} J_ij Z_i Z_j over all basis states."""
    n = len(h)
    k = np.arange(2 ** n, dtype=np.int64)
    z = 1 - 2 * ((k[:, None] >> np.arange(n)) & 1)
    return z @ h + np.einsum("ki,ij,kj->k", z, np.triu(J, 1), z)


def ising_diag(h, J):
    return ising_energy_vector(h, J)


# --------------------------------------------------------------------------- Metropolis
def metropolis_matrix(Q: np.ndarray, E: np.ndarray, T: float) -> np.ndarray:
    dE = E[None, :] - E[:, None]
    A = np.exp(-np.clip(dE, 0, None) / T)
    P = Q * A
    np.fill_diagonal(P, 0.0)
    P[np.diag_indices_from(P)] = 1.0 - P.sum(axis=1)
    return P


def boltzmann(E: np.ndarray, T: float) -> np.ndarray:
    w = np.exp(-(E - E.min()) / T)
    return w / w.sum()


def metropolis_gap(Q: np.ndarray, E: np.ndarray, T: float) -> float:
    """Absolute spectral gap of the Metropolis chain for a SYMMETRIC proposal Q.

    D^{1/2} P D^{-1/2} has off-diagonal entries Q_ab * exp(-|E_a - E_b| / 2T), which
    avoids dividing by Boltzmann weights that underflow at low T.
    """
    dE = np.abs(E[None, :] - E[:, None])
    S = Q * np.exp(-dE / (2 * T))
    np.fill_diagonal(S, 0.0)
    P_off = Q * np.exp(-np.clip(E[None, :] - E[:, None], 0, None) / T)
    np.fill_diagonal(P_off, 0.0)
    S[np.diag_indices_from(S)] = 1.0 - P_off.sum(axis=1)
    S = 0.5 * (S + S.T)
    lam = np.sort(eigh(S, eigvals_only=True))
    return float(1.0 - np.max(np.abs(lam[:-1])))


def spectral_gap(P: np.ndarray, pi: np.ndarray) -> float:
    """Absolute spectral gap of a general (possibly non-reversible) chain.

    Used for proposals estimated from noisy hardware, which need not be symmetric.
    """
    lam = np.linalg.eigvals(P)
    lam = lam[np.argsort(-np.abs(lam))]
    return float(1.0 - np.abs(lam[1]))


def detailed_balance_violation(P: np.ndarray, pi: np.ndarray) -> float:
    F = pi[:, None] * P
    return float(np.abs(F - F.T).sum() / 2)


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(P.T)
    i = np.argmin(np.abs(w - 1))
    p = np.real(v[:, i])
    return p / p.sum()


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())
