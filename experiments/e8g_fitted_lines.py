"""E8g: hardware-native proposals that FIT the real St. John's landscape instead of truncating it.

E8d showed that on real neighbourhoods (very uneven loads) the load-balance term links heavy areas that are not
map neighbours; a qubit line that simply drops those links proposes poorly. Here each line carries a fitted
Ising model: weighted least squares of the true energy over all 1,024 plans, using only that line's links,
weighted towards good plans (Boltzmann weights at T_fit = 2T), as in Project 1's best-line surrogate.
Same six instances and random draws as E8d. Kernels (short evolution t in [1, 6], 24 draws):
  line_cut / lines3_cut     E8d's truncated lines (for reference)
  line_fit / lines3_fit     fitted lines
  lines3_fit_flip           lines3_fit plus "swap the two groups" with probability 1/4, the same free move the
                            strongest classical baseline (mix_flip) gets
  circuit3_fit              the actual Trotterised circuits of lines3_fit (exact unitaries, all 24 draws)
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
from gridrisk.qmcmc.surrogate import boltzmann_weights, fit_ising  # noqa: E402
from gridrisk.rotation import ising, load_city  # noqa: E402
from experiments.e8_screen import K_EFFS, temperature_for  # noqa: E402
from experiments.e8b_hardware_friendly import line_only  # noqa: E402
from experiments.e8c_multiline import cover_lines  # noqa: E402
from experiments.e8d_stjohns_gaps import DRAWS, FAIRNESS, N, SEED, SEEDS, classical_kernels  # noqa: E402


def fitted_line(E, X, pairs, T_fit):
    h, J, _, _ = fit_ising(X, E, weights=boltzmann_weights(E, T_fit), pairs=pairs)
    return h, J


def flip_kernel(n):
    F = np.zeros((2 ** n, 2 ** n)); a = np.arange(2 ** n); F[a, a ^ (2 ** n - 1)] = 1.0
    return F


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
        lines, cover = cover_lines(J, 3)
        for kk in K_EFFS:
            T = temperature_for(E, kk)
            cut = [quantum_proposal_exact(h, line_only(J, p), g, t) for p in lines]
            fits = [fitted_line(E, X, p, 2 * T) for p in lines]
            fit = [quantum_proposal_exact(hf, Jf, g, t) for hf, Jf in fits]
            kernels = {"line_cut": cut[0], "lines3_cut": sum(cut) / 3, "line_fit": fit[0], "lines3_fit": sum(fit) / 3}
            kernels["lines3_fit_flip"] = 0.75 * kernels["lines3_fit"] + 0.25 * F
            kernels["circuit3_fit"] = sum(trotter_proposal_matrix(hf, Jf, g, t, pairs=p) for (hf, Jf), p in zip(fits, lines)) / 3
            cg = {name: metropolis_gap(Q, E, T) for name, Q in classical.items()}
            best = max(cg, key=cg.get)
            qg = {name: metropolis_gap(0.5 * (Q + Q.T), E, T) for name, Q in kernels.items()}
            rows.append(dict(seed=seed, names=[city.names[i] for i in idx], k_eff=kk, T=T, classical=cg, quantum=qg,
                             best_classical=best, ratio={name: v / cg[best] for name, v in qg.items()}))
        print(f"{seed:16s} " + " | ".join(
            f"K{r['k_eff']}: best {r['best_classical']} " + " ".join(f"{nm} x{v:.2f}" for nm, v in r["ratio"].items())
            for r in rows[-len(K_EFFS):]) + f"  [{time.time() - t0:.0f}s]", flush=True)
    (ROOT / "results" / "e8g_fitted_lines.json").write_text(json.dumps(dict(n=N, rows=rows), indent=1))
    print("\nmedian ratio to best classical (min, wins / instances):")
    for kk in K_EFFS:
        rs = [r for r in rows if r["k_eff"] == kk]
        print(f"  K={kk:3d}: " + "  ".join(
            f"{nm} x{np.median([r['ratio'][nm] for r in rs]):.2f} (min x{np.min([r['ratio'][nm] for r in rs]):.2f}, "
            f"{sum(r['ratio'][nm] > 1 for r in rs)}/{len(rs)})" for nm in rs[0]["ratio"]))


if __name__ == "__main__":
    run()
