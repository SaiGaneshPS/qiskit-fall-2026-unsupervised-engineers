"""E8d: quantum vs classical proposals on the real St. John's rotation problem (exact spectral gaps, n = 10).

Instances: 10 contiguous neighbourhoods grown from six seeds across the city (OpenStreetMap boundaries), with
the E8 energy (checkerboard borders + mild load balance + fairness fields from a random hardship history).
Classical proposals: local, pair, uniform, mix, and mix_flip (mix plus "swap the two groups", which a skeptic
would add because exchanging A and B is nearly free). Quantum proposals (short evolution t in [1, 6]):
  full      every map link in the circuit (not hardware-native)
  line      one best line of qubits (hardware-native)
  lines3    three lines covering the map, one picked at random per step (hardware-native)
  circuit3  the actual Trotterised circuits of lines3 (exact unitaries)
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
from gridrisk.qmcmc.proposal import GAMMA_RANGE, trotter_proposal_matrix  # noqa: E402
from gridrisk.rotation import ising, load_city  # noqa: E402
from experiments.e8_screen import K_EFFS, pair_proposal, temperature_for  # noqa: E402
from experiments.e8b_hardware_friendly import line_only  # noqa: E402
from experiments.e8c_multiline import cover_lines  # noqa: E402

N = 10
SEEDS = ["Downtown", "Georgestown", "Churchill Park", "Cowan Heights", "Clovelly Trails", "White Hills"]
DRAWS = 24
FAIRNESS = 0.3
SEED = 838


def classical_kernels(n):
    k = {"local": local_proposal(n), "pair": pair_proposal(n), "uniform": uniform_proposal(n)}
    k["mix"] = (k["local"] + k["pair"] + k["uniform"]) / 3
    flip = np.zeros_like(k["local"]); a = np.arange(2 ** n); flip[a, a ^ (2 ** n - 1)] = 1.0
    k["mix_flip"] = (k["local"] + k["pair"] + k["uniform"] + flip) / 4
    return k


def run():
    rng = np.random.default_rng(SEED)
    city = load_city()
    classical = classical_kernels(N)
    rows, t0 = [], time.time()
    for seed in SEEDS:
        idx = city.subregion(seed, N)
        h, J = ising(city, idx, fairness=FAIRNESS * rng.normal(0, 1, N))
        E = ising_energy_vector(h, J)
        lines, cover = cover_lines(J, 3)
        g, t = rng.uniform(*GAMMA_RANGE, DRAWS), rng.uniform(1, 6, DRAWS)
        per_line = [quantum_proposal_exact(h, line_only(J, p), g, t) for p in lines]
        kernels = {"full": quantum_proposal_exact(h, J, g, t), "line": per_line[0], "lines3": sum(per_line) / 3}
        kernels["circuit3"] = sum(trotter_proposal_matrix(h, line_only(J, p), g[:8], t[:8], pairs=p) for p in lines) / 3
        for kk in K_EFFS:
            T = temperature_for(E, kk)
            cg = {name: metropolis_gap(Q, E, T) for name, Q in classical.items()}
            best = max(cg, key=cg.get)
            qg = {name: metropolis_gap(0.5 * (Q + Q.T), E, T) for name, Q in kernels.items()}
            rows.append(dict(seed=seed, names=[city.names[i] for i in idx], k_eff=kk, T=T, coverage=cover,
                             classical=cg, quantum=qg, best_classical=best,
                             ratio={name: v / cg[best] for name, v in qg.items()}))
        print(f"{seed:16s} cover {np.round(cover, 2)} " + " | ".join(
            f"K{r['k_eff']}: best classical {r['best_classical']} " + " ".join(f"{nm} x{v:.2f}" for nm, v in r["ratio"].items())
            for r in rows[-len(K_EFFS):]) + f"  [{time.time() - t0:.0f}s]", flush=True)
    (ROOT / "results" / "e8d_stjohns_gaps.json").write_text(json.dumps(dict(n=N, fairness=FAIRNESS, rows=rows), indent=1))
    print("\nmedian ratio to best classical (wins / instances):")
    for kk in K_EFFS:
        rs = [r for r in rows if r["k_eff"] == kk]
        print(f"  K={kk:3d}: " + "  ".join(
            f"{nm} x{np.median([r['ratio'][nm] for r in rs]):.2f} (min x{np.min([r['ratio'][nm] for r in rs]):.2f}, "
            f"{sum(r['ratio'][nm] > 1 for r in rs)}/{len(rs)})" for nm in rs[0]["ratio"]))


if __name__ == "__main__":
    run()
