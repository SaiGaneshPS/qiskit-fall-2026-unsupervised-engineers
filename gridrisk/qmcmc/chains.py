"""Metropolis chains run in lockstep, with pluggable proposal generators.

Every proposal type is a callable  propose(X, rng) -> Y  on an (M, n) array of the
current designs of M independent chains. Acceptance always uses the TRUE Marxan
energy, so any symmetric proposal leaves the exact Boltzmann distribution invariant.
"""
from __future__ import annotations

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from .proposal import (build_template, bind_values, coupling_pairs, n_trotter_steps,
                      ordered_parameters)


def local_proposer(X, rng):
    Y = X.copy()
    Y[np.arange(len(X)), rng.integers(0, X.shape[1], len(X))] ^= 1
    return Y


def uniform_proposer(X, rng):
    return rng.integers(0, 2, X.shape).astype(X.dtype)


class QuantumProposer:
    """Samples s' ~ |<s'|V~(gamma, t)|s>|^2 by executing Qiskit circuits.

    backend: any Qiskit backend with a `run` method (AerSimulator by default, or an Aer
    simulator carrying a hardware noise model). `twirl` enables SPAM twirling.
    """

    def __init__(self, h, J, gammas, times, backend=None, pairs=None, twirl=True, seed=0):
        self.h, self.J = np.asarray(h), np.asarray(J)
        self.n = len(h)
        self.gammas, self.times = np.asarray(gammas), np.asarray(times)
        self.pairs = coupling_pairs(self.J) if pairs is None else pairs
        # parallelise across the batched circuits rather than inside each small statevector
        self.backend = backend or AerSimulator(method="statevector", seed_simulator=seed,
                                               max_parallel_experiments=0, statevector_parallel_threshold=18)
        self.twirl = twirl
        self._tpl: dict[int, tuple] = {}
        self.circuits_run = 0

    def _template(self, r):
        if r not in self._tpl:
            tpl = build_template(self.n, self.pairs, r)
            tqc = transpile(tpl.circuit, self.backend, optimization_level=1)
            self._tpl[r] = (tpl, tqc, ordered_parameters(tpl))
        return self._tpl[r]

    def __call__(self, X, rng):
        M = len(X)
        k = rng.integers(0, len(self.gammas), M)
        keys = rng.integers(0, 2, X.shape) if self.twirl else np.zeros_like(X)
        Y = np.empty_like(X)
        r_of = np.array([n_trotter_steps(t) for t in self.times[k]])
        # one backend call per MCMC step: every chain's circuit, grouped by Trotter depth
        circuits, binds, order = [], [], []
        for r in np.unique(r_of):
            idx = np.where(r_of == r)[0]
            tpl, tqc, params = self._template(int(r))
            vals = np.array([bind_values(tpl, self.h, self.J, self.gammas[k[i]], self.times[k[i]],
                                         X[i], keys[i]) for i in idx])
            circuits.append(tqc)
            binds.append({prm: vals[:, j].tolist() for j, prm in enumerate(params)})
            order.extend(idx)
        res = self.backend.run(circuits, parameter_binds=binds, shots=1).result()
        for a, i in enumerate(order):
            bitstr = next(iter(res.get_counts(a)))
            bits = np.array([int(b) for b in bitstr[::-1]], dtype=X.dtype)
            Y[i] = bits ^ keys[i]
        self.circuits_run += len(order)
        return Y


def run_chains(p, propose, T: float, steps: int, M: int, rng: np.random.Generator,
               X0: np.ndarray | None = None, record_every: int = 1, log_every: int = 0):
    """Run M Metropolis chains; returns visited designs (steps//record_every, M, n) and stats."""
    X = rng.integers(0, 2, (M, p.n)).astype(np.int8) if X0 is None else X0.copy()
    E = p.energy(X)
    traj = []
    acc = 0
    for s in range(steps):
        Y = propose(X, rng)
        Ey = p.energy(Y)
        a = rng.random(M) < np.exp(-np.clip(Ey - E, 0, None) / T)
        X[a], E[a] = Y[a], Ey[a]
        acc += a.sum()
        if (s + 1) % record_every == 0:
            traj.append(X.copy())
        if log_every and (s + 1) % log_every == 0:
            print(f"  step {s + 1}/{steps}  acceptance {acc / ((s + 1) * M):.3f}  mean E {E.mean():.3f}", flush=True)
    return np.array(traj), dict(acceptance=acc / (steps * M))


def run_parallel_tempering(p, T: float, sweeps: int, M: int, rng: np.random.Generator,
                           n_replicas: int = 8, T_max_factor: float = 20.0):
    """Classical baseline: replica-exchange MCMC with local moves.

    Each sweep costs n_replicas energy evaluations (one local move per replica), then
    attempts swaps between neighbouring temperatures. Returns the T-replica trajectory.
    """
    temps = T * np.geomspace(1.0, T_max_factor, n_replicas)
    X = rng.integers(0, 2, (M, n_replicas, p.n)).astype(np.int8)
    E = p.energy(X)
    traj = []
    for s in range(sweeps):
        Y = X.copy()
        flip = rng.integers(0, p.n, (M, n_replicas))
        np.put_along_axis(Y, flip[..., None], 1 - np.take_along_axis(Y, flip[..., None], 2), 2)
        Ey = p.energy(Y)
        a = rng.random((M, n_replicas)) < np.exp(-np.clip(Ey - E, 0, None) / temps)
        X[a], E[a] = Y[a], Ey[a]
        start = s % 2
        for k in range(start, n_replicas - 1, 2):
            d = (1 / temps[k] - 1 / temps[k + 1]) * (E[:, k] - E[:, k + 1])
            sw = rng.random(M) < np.exp(np.clip(d, None, 0))
            X[sw, k], X[sw, k + 1] = X[sw, k + 1].copy(), X[sw, k].copy()
            E[sw, k], E[sw, k + 1] = E[sw, k + 1].copy(), E[sw, k].copy()
        traj.append(X[:, 0].copy())
    return np.array(traj)


def running_marginal_error(traj: np.ndarray, exact_marg: np.ndarray, checkpoints: np.ndarray,
                           burn_frac: float = 0.0) -> np.ndarray:
    """Per-chain max |m_hat - m| over units at each checkpoint (number of recorded steps)."""
    out = []
    csum = np.cumsum(traj, axis=0, dtype=float)
    for c in checkpoints:
        b = int(burn_frac * c)
        tot = csum[c - 1] - (csum[b - 1] if b > 0 else 0)
        m_hat = tot / (c - b)
        out.append(np.abs(m_hat - exact_marg).max(axis=1))
    return np.array(out)  # (len(checkpoints), M)

