"""Classical warm-up that collects training data for the learned Ising surrogate.

Only uses energy evaluations along local Metropolis chains (no enumeration), so the
pipeline scales to problems far beyond exact enumeration.

The surrogate only has to describe the landscape at the scale that matters for the
decision: designs within a few T of the best one. Large target penalties create steep
walls that would otherwise dominate a least-squares fit and, after the Hamiltonian is
normalised, hide the small cost differences between good designs from the circuit. So
(1) the warm-up ladder cools down to the decision temperature T, and (2) energies are
clipped at E_best + CAP_MULT * T before fitting (everything above looks "equally bad").
"""
from __future__ import annotations

import numpy as np

from .surrogate import boltzmann_weights, fit_ising

T_DECISION = 0.25   # decision temperature (cost units)
CAP_MULT = 16       # fit window: E_best + 16 T (= 4 cost units at T = 0.25)


def energy_scale(p, rng: np.random.Generator, m: int = 256) -> float:
    return float(np.std(p.energy(rng.integers(0, 2, (m, p.n)))))


def warmup_samples(p, rng: np.random.Generator, n_evals: int | None = None,
                   n_temps: int = 6, T_end: float = T_DECISION) -> tuple[np.ndarray, np.ndarray]:
    """Local Metropolis over a geometric ladder from the random-design energy scale to T_end."""
    n = p.n
    n_evals = 30 * n * n if n_evals is None else n_evals
    temps = np.geomspace(energy_scale(p, rng), T_end, n_temps)
    per = n_evals // n_temps
    seen: dict[bytes, float] = {}
    x = rng.integers(0, 2, n).astype(np.int8)
    e = float(p.energy(x))
    for T in temps:
        for _ in range(per):
            y = x.copy()
            y[rng.integers(n)] ^= 1
            ey = float(p.energy(y))
            seen[y.tobytes()] = ey
            if ey <= e or rng.random() < np.exp(-(ey - e) / T):
                x, e = y, ey
    X = np.array([np.frombuffer(k, dtype=np.int8) for k in seen])
    return X, np.array(list(seen.values()))


def hardware_native_surrogate(p, rng: np.random.Generator, n_evals: int | None = None,
                              T_decision: float = T_DECISION, cap_mult: float = CAP_MULT):
    """Surrogate whose couplings form a line of qubits (no SWAPs on hardware).

    Fit the full surrogate, choose the unit ordering that keeps its strongest couplings
    (max-weight Hamiltonian path), then refit with only those n - 1 couplings.
    Returns (h, J, pairs, order); order[k] is the unit placed on the k-th qubit of the line.
    """
    from .surrogate import best_line_pairs
    X, Es = warmup_samples(p, rng, n_evals, T_end=T_decision)
    Ec = np.minimum(Es, Es.min() + cap_mult * T_decision)
    _, J_full, _, _ = fit_ising(X, Ec)
    pairs, order = best_line_pairs(J_full)
    h, J, _, _ = fit_ising(X, Ec, pairs=pairs)
    return h, J, pairs, order


def learn_surrogate(p, rng: np.random.Generator, pairs=None, n_evals: int | None = None,
                    T_decision: float = T_DECISION, cap_mult: float | None = CAP_MULT):
    """Fit h, J to warm-up samples. cap_mult=None reproduces the unclipped, Boltzmann-weighted fit."""
    X, Es = warmup_samples(p, rng, n_evals, T_end=T_decision)
    if cap_mult is None:
        h, J, c, _ = fit_ising(X, Es, boltzmann_weights(Es, 0.3 * np.std(Es)), pairs=pairs)
    else:
        h, J, c, _ = fit_ising(X, np.minimum(Es, Es.min() + cap_mult * T_decision), pairs=pairs)
    return h, J, c, len(X)
