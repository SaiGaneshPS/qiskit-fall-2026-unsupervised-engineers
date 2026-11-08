"""Transition-matrix tomography of the quantum proposal (as in Layden et al. Fig. 3).

For every initial design s and every (gamma, t) pair, run the proposal circuit and
histogram the outputs. The resulting Q(s'|s) gives the exact Metropolis transition
matrix of the device-level algorithm, so its spectral gap, stationary distribution and
bias can be computed directly. Shared by the noisy simulation (E4) and hardware (E5).
"""
from __future__ import annotations

import numpy as np
from qiskit import transpile

from .proposal import bind_native, build_native_template, n_trotter_steps


def proposal_jobs(h, J, pairs, gammas, times, keys_per_input: int, rng: np.random.Generator,
                  twirl: bool = True):
    """Group all (s, gamma, t, key) settings by Trotter depth r.

    Returns {r: (template, settings)} where settings rows are (s_int, pair_index, key_int).
    """
    n = len(h)
    jobs: dict[int, list] = {}
    for k, t in enumerate(times):
        r = n_trotter_steps(t)
        for s in range(2 ** n):
            for _ in range(keys_per_input):
                key = int(rng.integers(0, 2 ** n)) if twirl else 0
                jobs.setdefault(r, []).append((s, k, key))
    templates = {r: build_native_template(n, pairs, r) for r in jobs}
    return {r: (templates[r], np.array(v)) for r, v in jobs.items()}


def to_bits(v: int, n: int) -> np.ndarray:
    return (v >> np.arange(n)) & 1


def parameter_table(tpl, h, J, gammas, times, settings) -> np.ndarray:
    n = len(h)
    return np.array([bind_native(tpl, h, J, gammas[k], times[k], to_bits(s, n), to_bits(key, n))
                     for s, k, key in settings])


def accumulate(Q_counts: np.ndarray, settings, counts_list, n: int, physical_to_logical=None):
    """Add measured counts into Q_counts[s, s'] after undoing the twirl key."""
    for (s, k, key), counts in zip(settings, counts_list):
        for bitstr, c in counts.items():
            out = int(bitstr.replace(" ", ""), 2)
            Q_counts[s, out ^ key] += c
    return Q_counts


def normalise(Q_counts: np.ndarray) -> np.ndarray:
    return Q_counts / Q_counts.sum(axis=1, keepdims=True)


def metropolis_from_empirical(Q: np.ndarray, E: np.ndarray, T: float) -> np.ndarray:
    """Transition matrix as the device would realise it (acceptance assumes symmetry)."""
    A = np.exp(-np.clip(E[None, :] - E[:, None], 0, None) / T)
    P = Q * A
    np.fill_diagonal(P, 0.0)
    P[np.diag_indices_from(P)] = 1.0 - P.sum(axis=1)
    return P


def chain_diagnostics(Q: np.ndarray, E: np.ndarray, T: float, X: np.ndarray) -> dict:
    """Gap, stationary-distribution bias and marginal bias of the realised chain."""
    from .mcmc import boltzmann, spectral_gap, stationary_distribution, tvd
    P = metropolis_from_empirical(Q, E, T)
    pi = boltzmann(E, T)
    st = np.clip(stationary_distribution(P), 0, None)
    st /= st.sum()
    return dict(gap=spectral_gap(P, pi), tvd_stationary=tvd(st, pi),
                max_marginal_bias=float(np.abs(st @ X - pi @ X).max()),
                asymmetry=float(0.5 * np.abs(Q - Q.T).sum(axis=1).mean()))

