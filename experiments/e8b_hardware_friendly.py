"""E8b: does the quantum advantage survive the changes a real chip forces on the circuit?

For the families that won the E8 screen, exact spectral gaps (n = 10) of:
  full        the problem's own Ising Hamiltonian, all couplings, t in [2, 20] (Layden et al.)
  line        couplings kept only along the best line of qubits (no SWAP gates on hardware), t in [2, 20]
  line_short  the same line, short evolution t in [1, 6] (at most 8 Trotter steps of 0.8)
  circuit     the actual Trotterised line_short circuits (what the chip runs), exact unitary
against the best classical proposal (local, pair, swap, uniform, mix). Temperatures as in E8.
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
from gridrisk.qmcmc.surrogate import best_line_pairs  # noqa: E402
from experiments.e8_screen import K_EFFS, family, pair_proposal, swap_proposal, temperature_for  # noqa: E402

N_SPINS = 10
INSTANCES = 6
DRAWS = 32
SEED = 818


def line_only(J, pairs):
    Jl = np.zeros_like(J)
    for i, j in pairs:
        Jl[i, j] = J[i, j]
    return Jl


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
            pairs, order = best_line_pairs(J)
            Jl = line_only(J, pairs)
            g = rng.uniform(*GAMMA_RANGE, DRAWS)
            t_long, t_short = rng.uniform(2, 20, DRAWS), rng.uniform(1, 6, DRAWS)
            kernels = {"full": quantum_proposal_exact(h, J, g, t_long),
                       "line": quantum_proposal_exact(h, Jl, g, t_long),
                       "line_short": quantum_proposal_exact(h, Jl, g, t_short)}
            if n <= 10:
                kernels["circuit"] = trotter_proposal_matrix(h, Jl, g[:12], t_short[:12], pairs=pairs)
            for k in K_EFFS:
                T = temperature_for(E, k)
                cg = {name: metropolis_gap(Q, E, T) for name, Q in classical.items()}
                best = max(cg, key=cg.get)
                qg = {name: metropolis_gap(0.5 * (Q + Q.T), E, T) for name, Q in kernels.items()}
                rows.append(dict(family=fam, instance=inst, k_eff=k, T=T, classical=cg, quantum=qg, best_classical=best,
                                 ratio={name: v / cg[best] for name, v in qg.items()}))
                r = rows[-1]["ratio"]
            print(f"{fam:9s} #{inst} " + " | ".join(
                f"K{rr['k_eff']}: " + " ".join(f"{nm} x{v:.2f}" for nm, v in rr["ratio"].items())
                for rr in rows[-len(K_EFFS):]) + f"  [{time.time() - t0:.0f}s]", flush=True)
    (ROOT / "results" / "e8b_hardware_friendly.json").write_text(json.dumps(dict(n=n, rows=rows), indent=1))
    print("\nmedian ratio to best classical (wins / instances):")
    for fam in families:
        for k in K_EFFS:
            rs = [r for r in rows if r["family"] == fam and r["k_eff"] == k]
            print(f"  {fam:9s} K={k:3d}: " + "  ".join(
                f"{nm} x{np.median([r['ratio'][nm] for r in rs]):.2f} ({sum(r['ratio'][nm] > 1 for r in rs)}/{len(rs)})"
                for nm in rs[0]["ratio"]))


if __name__ == "__main__":
    run(sys.argv[1:] or ["sk", "shed"])
