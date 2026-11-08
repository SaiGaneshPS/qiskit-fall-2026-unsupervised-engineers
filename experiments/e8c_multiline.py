"""E8c: recover the full quantum advantage with hardware-native circuits by rotating between several qubit lines.

A real chip couples only neighbouring qubits, so one circuit can carry the couplings of one line (a Hamiltonian
path through the map). Each proposal step picks one of k lines at random (each line is its own qubit layout);
the mixture of symmetric proposals is symmetric, so Metropolis stays exact. Lines are chosen greedily so that
together they cover as many map links as possible. Short evolution t in [1, 6] (at most 8 Trotter steps).
Exact spectral gaps, n = 10, against the best classical proposal (as in E8 / E8b).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridrisk.qmcmc.mcmc import (ising_energy_vector, local_proposal, metropolis_gap,  # noqa: E402
                                 quantum_proposal_exact, uniform_proposal)
from gridrisk.qmcmc.proposal import GAMMA_RANGE  # noqa: E402
from gridrisk.qmcmc.surrogate import best_line_pairs  # noqa: E402
from experiments.e8_screen import K_EFFS, family, pair_proposal, swap_proposal, temperature_for  # noqa: E402
from experiments.e8b_hardware_friendly import line_only  # noqa: E402

N_SPINS = 10
INSTANCES = 6
DRAWS = 24
MAX_LINES = 4
SEED = 828


def cover_lines(J, k):
    """Greedy: each new line maximises the total |J| of links not yet carried by an earlier line."""
    W = np.abs(np.triu(J, 1))
    lines, covered = [], np.zeros_like(W, dtype=bool)
    for _ in range(k):
        pairs, _ = best_line_pairs(np.where(covered, 1e-3 * W, W))
        lines.append(pairs)
        for i, j in pairs:
            covered[i, j] = True
    cum, seen = [], set()
    for p in lines:
        seen |= set(p)
        cum.append(float(sum(W[i, j] for i, j in seen) / W.sum()))
    return lines, cum


def run(families):
    rng = np.random.default_rng(SEED)
    n = N_SPINS
    classical = {"local": local_proposal(n), "pair": pair_proposal(n), "swap": swap_proposal(n),
                 "uniform": uniform_proposal(n)}
    classical["mix"] = (classical["local"] + classical["pair"] + classical["uniform"]) / 3
    rows, t0 = [], time.time()
    for fam in families:
        for inst in range(INSTANCES):
            h, J = family(fam, rng)
            E = ising_energy_vector(h, J)
            lines, cover = cover_lines(J, MAX_LINES)
            g, t = rng.uniform(*GAMMA_RANGE, DRAWS), rng.uniform(1, 6, DRAWS)
            per_line = [quantum_proposal_exact(h, line_only(J, p), g, t) for p in lines]
            kernels = {f"lines{k}": sum(per_line[:k]) / k for k in range(1, MAX_LINES + 1)}
            kernels["full_short"] = quantum_proposal_exact(h, J, g, t)
            for kk in K_EFFS:
                T = temperature_for(E, kk)
                cg = {name: metropolis_gap(Q, E, T) for name, Q in classical.items()}
                best = max(cg, key=cg.get)
                qg = {name: metropolis_gap(Q, E, T) for name, Q in kernels.items()}
                rows.append(dict(family=fam, instance=inst, k_eff=kk, T=T, classical=cg, quantum=qg, best_classical=best,
                                 coverage=cover, ratio={name: v / cg[best] for name, v in qg.items()}))
            print(f"{fam:10s} #{inst} cover {np.round(cover, 2)} " + " | ".join(
                f"K{r['k_eff']}: " + " ".join(f"{nm} x{v:.2f}" for nm, v in r["ratio"].items()) for r in rows[-len(K_EFFS):])
                + f"  [{time.time() - t0:.0f}s]", flush=True)
    (ROOT / "results" / "e8c_multiline.json").write_text(json.dumps(dict(n=n, rows=rows), indent=1))
    print("\nmedian ratio to best classical (wins / instances):")
    for fam in families:
        for kk in K_EFFS:
            rs = [r for r in rows if r["family"] == fam and r["k_eff"] == kk]
            print(f"  {fam:10s} K={kk:3d}: " + "  ".join(
                f"{nm} x{np.median([r['ratio'][nm] for r in rs]):.2f} ({sum(r['ratio'][nm] > 1 for r in rs)}/{len(rs)})"
                for nm in rs[0]["ratio"]))


if __name__ == "__main__":
    run(sys.argv[1:] or ["planar", "planar_bal"])
