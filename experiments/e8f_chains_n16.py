"""E8f: actual Markov chains on a 16-neighbourhood St. John's region (65,536 rotation plans).

Every sampler gets the same number of energy evaluations. The exact Boltzmann marginals (the share of good
rotation plans that put each neighbourhood in the first dark group) are computed by enumeration; the error of
each chain's running estimate is tracked. Samplers:
  local, uniform           classical single proposals
  mix_flip                 classical mixture: flip one, flip two, uniform, swap the two groups
  parallel_tempering       8 replicas with local moves (each sweep costs 8 evaluations)
  quantum_line_fit         Layden et al. proposals on one hardware-native qubit line carrying a fitted Ising model
                           (Boltzmann-weighted least squares of the true energy, as in E8g), Qiskit Aer, noiseless
  quantum_line_fit_flip    the same, plus "swap the two groups" with probability 1/4 (the move mix_flip gets free)
  quantum_line_fit_flip_noisy   the same circuits with the IBM Heron noise model (Aer, trajectories)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridrisk.qmcmc.chains import (QuantumProposer, local_proposer, run_chains, run_parallel_tempering,  # noqa: E402
                                   running_marginal_error, uniform_proposer)
from gridrisk.qmcmc.mcmc import boltzmann, ising_energy_vector  # noqa: E402
from gridrisk.qmcmc.proposal import GAMMA_RANGE  # noqa: E402
from gridrisk.qmcmc.surrogate import boltzmann_weights, fit_ising  # noqa: E402
from gridrisk.rotation import IsingProblem, ising, load_city  # noqa: E402
from experiments.e8_screen import temperature_for  # noqa: E402
from experiments.e8b_hardware_friendly import line_only  # noqa: E402
from experiments.e8c_multiline import cover_lines  # noqa: E402
from experiments.e8d_stjohns_gaps import FAIRNESS  # noqa: E402

N = 16
SEED_AREA = "Downtown"
STEPS = 2000
M = 16
K_EFF = 16
DRAWS = 24
SEED = 868


def mix_flip_proposer(X, rng):
    Y = X.copy()
    M_, n = X.shape
    kind = rng.integers(0, 4, M_)
    for i in range(M_):
        if kind[i] == 0:
            Y[i, rng.integers(n)] ^= 1
        elif kind[i] == 1:
            a, b = rng.choice(n, 2, replace=False); Y[i, a] ^= 1; Y[i, b] ^= 1
        elif kind[i] == 2:
            Y[i] = rng.integers(0, 2, n)
        else:
            Y[i] ^= 1
    return Y


class WithFlip:
    """With probability 1/4 swap the two groups (flip every bit); otherwise use the wrapped proposal."""

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, X, rng):
        flip = rng.random(len(X)) < 0.25
        Y = X ^ 1
        if (~flip).any():
            Y[~flip] = self.inner(X[~flip], rng)
        return Y


class MultiLineProposer:
    """Each chain's proposal comes from one of several line circuits, chosen uniformly at random."""

    def __init__(self, proposers):
        self.proposers = proposers

    def __call__(self, X, rng):
        Y = np.empty_like(X)
        which = rng.integers(0, len(self.proposers), len(X))
        for k, prop in enumerate(self.proposers):
            sel = np.where(which == k)[0]
            if len(sel):
                Y[sel] = prop(X[sel], rng)
        return Y


def run(include_noisy=True):
    from qiskit_aer import AerSimulator
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    city = load_city()
    idx = city.subregion(SEED_AREA, N)
    h, J = ising(city, idx, fairness=FAIRNESS * rng.normal(0, 1, N))
    p = IsingProblem(h, J)
    E = ising_energy_vector(h, J)
    T = temperature_for(E, K_EFF)
    X_all = ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1).astype(float)
    exact_marg = boltzmann(E, T) @ X_all
    lines, cover = cover_lines(J, 1)
    pr = lines[0]
    hf, Jf, _, _ = fit_ising(X_all, E, weights=boltzmann_weights(E, 2 * T), pairs=pr)
    g, t = rng.uniform(*GAMMA_RANGE, DRAWS), rng.uniform(1, 6, DRAWS)
    q_ideal = QuantumProposer(hf, Jf, g, t, pairs=pr, seed=SEED)
    samplers = {"local": local_proposer, "uniform": uniform_proposer, "mix_flip": mix_flip_proposer,
                "quantum_line_fit": q_ideal, "quantum_line_fit_flip": WithFlip(q_ideal)}
    if include_noisy:
        from qiskit_ibm_runtime.fake_provider import FakeFez
        noisy = AerSimulator.from_backend(FakeFez(), seed_simulator=SEED, max_parallel_experiments=0)
        samplers["quantum_line_fit_flip_noisy"] = WithFlip(QuantumProposer(hf, Jf, g, t, pairs=pr, backend=noisy, seed=SEED))
    checkpoints = np.unique(np.geomspace(10, STEPS, 25).astype(int))
    out = dict(area=[city.names[i] for i in idx], n=N, T=T, k_eff=K_EFF, steps=STEPS, chains=M, coverage=cover,
               checkpoints=checkpoints.tolist(), exact_marginals=exact_marg.tolist(), samplers={})
    X0 = rng.integers(0, 2, (M, N)).astype(np.int8)
    for name, prop in samplers.items():
        ts = time.time()
        traj, st = run_chains(p, prop, T, STEPS, M, np.random.default_rng(SEED + 7), X0=X0)
        err = running_marginal_error(traj.astype(float), exact_marg, checkpoints)
        out["samplers"][name] = dict(median_error=np.median(err, axis=1).tolist(), acceptance=st["acceptance"],
                                     seconds=time.time() - ts)
        print(f"{name:22s} final median marginal error {np.median(err[-1]):.3f}  acceptance {st['acceptance']:.3f}  "
              f"[{time.time() - ts:.0f}s]", flush=True)
    R = 8
    traj = run_parallel_tempering(p, T, STEPS // R, M, np.random.default_rng(SEED + 7), n_replicas=R)
    cp_pt = np.maximum(checkpoints // R, 1)
    err = running_marginal_error(traj.astype(float), exact_marg, cp_pt)
    out["samplers"]["parallel_tempering"] = dict(median_error=np.median(err, axis=1).tolist(), replicas=R)
    print(f"{'parallel_tempering':22s} final median marginal error {np.median(err[-1]):.3f}", flush=True)
    out["seconds"] = time.time() - t0
    (ROOT / "results" / "e8f_chains_n16.json").write_text(json.dumps(out, indent=1))
    print(f"done in {out['seconds']:.0f}s")


if __name__ == "__main__":
    run()
