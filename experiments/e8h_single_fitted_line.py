"""E8h: the chosen hardware design on all six real St. John's clusters (exact spectral gaps, n = 10).

E8d/E8g showed that mixing several qubit lines dilutes the proposal, while ONE best line carrying a FITTED
Ising model (Boltzmann-weighted least squares of the true energy on that line's links, T_fit = 2T) beats the
strongest classical proposal. Kernels (t in [1, 6], 24 draws), against local, pair, uniform, mix, mix_flip:
  line_fit            exact evolution, fitted line
  circuit_fit         the actual Trotterised circuits the chip runs (exact unitaries)
  line_fit_flip       line_fit plus "swap the two groups" with probability 1/4 (the free move mix_flip gets)
  circuit_fit_flip    the same for the circuits
Same instances and random draws as E8d.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridrisk.qmcmc.mcmc import ising_energy_vector, metropolis_gap, quantum_proposal_exact  # noqa: E402
from gridrisk.qmcmc.proposal import GAMMA_RANGE, trotter_proposal_matrix  # noqa: E402
from gridrisk.qmcmc.surrogate import best_line_pairs  # noqa: E402
from gridrisk.rotation import ising, load_city  # noqa: E402
from experiments.e8_screen import K_EFFS, temperature_for  # noqa: E402
from experiments.e8d_stjohns_gaps import DRAWS, FAIRNESS, N, SEED, SEEDS, classical_kernels  # noqa: E402
from experiments.e8g_fitted_lines import fitted_line, flip_kernel  # noqa: E402


def run():
    rng = np.random.default_rng(SEED)          # same draws as E8d
    city = load_city()
    classical = classical_kernels(N)
    F = flip_kernel(N)
    X = ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1).astype(float)
    rows, t0 = [], time.time()
    for seed in SEEDS:
        idx = city.subregion(seed, N)
        h, J = ising(city, idx, fairness=FAIRNESS * rng.normal(0, 1, N))
        g, t = rng.uniform(*GAMMA_RANGE, DRAWS), rng.uniform(1, 6, DRAWS)
        E = ising_energy_vector(h, J)
        pairs, order = best_line_pairs(J)
        for kk in K_EFFS:
            T = temperature_for(E, kk)
            hf, Jf = fitted_line(E, X, pairs, 2 * T)
            kernels = {"line_fit": quantum_proposal_exact(hf, Jf, g, t),
                       "circuit_fit": trotter_proposal_matrix(hf, Jf, g, t, pairs=pairs)}
            kernels["line_fit_flip"] = 0.75 * kernels["line_fit"] + 0.25 * F
            kernels["circuit_fit_flip"] = 0.75 * kernels["circuit_fit"] + 0.25 * F
            cg = {name: metropolis_gap(Q, E, T) for name, Q in classical.items()}
            best = max(cg, key=cg.get)
            qg = {name: metropolis_gap(0.5 * (Q + Q.T), E, T) for name, Q in kernels.items()}
            rows.append(dict(seed=seed, names=[city.names[i] for i in idx], k_eff=kk, T=T, line_order=[city.names[idx[i]] for i in order],
                             classical=cg, quantum=qg, best_classical=best, ratio={name: v / cg[best] for name, v in qg.items()}))
        print(f"{seed:16s} " + " | ".join(
            f"K{r['k_eff']}: best {r['best_classical']} " + " ".join(f"{nm} x{v:.2f}" for nm, v in r["ratio"].items())
            for r in rows[-len(K_EFFS):]) + f"  [{time.time() - t0:.0f}s]", flush=True)
    (ROOT / "results" / "e8h_single_fitted_line.json").write_text(json.dumps(dict(n=N, rows=rows), indent=1))
    print("\nmedian ratio to best classical (min, wins / instances):")
    for kk in K_EFFS:
        rs = [r for r in rows if r["k_eff"] == kk]
        print(f"  K={kk:3d}: " + "  ".join(
            f"{nm} x{np.median([r['ratio'][nm] for r in rs]):.2f} (min x{np.min([r['ratio'][nm] for r in rs]):.2f}, "
            f"{sum(r['ratio'][nm] > 1 for r in rs)}/{len(rs)})" for nm in rs[0]["ratio"]))


if __name__ == "__main__":
    run()
