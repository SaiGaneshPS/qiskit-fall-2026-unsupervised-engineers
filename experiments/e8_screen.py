"""E8 screen: on which kinds of landscape does the quantum proposal beat the best classical proposal?

Exact spectral gaps (n = 10 spins, 1,024 states) of Metropolis chains that all target the same Boltzmann
distribution. Proposals compared:
  classical: local (flip one), pair (flip two at random), swap (one in, one out: keeps the count),
             uniform (any state), mix (equal mixture of local, pair and uniform)
  quantum:   Layden et al. (Nature 2023) proposal with the problem's own Ising Hamiltonian, averaged over
             random (gamma, t)
Temperature is set per instance so the Boltzmann distribution has a fixed effective size
exp(entropy) = K_EFF (the "handful of good answers" regime that matters for planning).

Families (all small, all native Ising):
  sk          Sherrington-Kirkpatrick spin glass (Layden's benchmark)
  shed        rolling-outage selection: cut feeders whose loads sum to a shortfall, with small
              per-feeder costs (subset-sum + fields)
  partition   balance two sides (number partitioning)
  maxcut3     antiferromagnet on a random 3-regular graph + weak fields
  tri_afm     triangular-lattice antiferromagnet + random fields
  rfim        2 x 5 ferromagnet in a random field
  cliques     two antiferromagnetic cliques of 5 + random fields
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
from gridrisk.qmcmc.proposal import draw_parameters  # noqa: E402

N_SPINS = 10
INSTANCES = 6
K_EFFS = [4, 16]
DRAWS = 32
SEED = 808


def qubo_to_ising(q, Q):
    """E(x) = sum q_i x_i + sum_{i<j} Q_ij x_i x_j  ->  h, J with x = (1 - z) / 2 (constant dropped)."""
    Q = np.triu(Q, 1)
    Qs = Q + Q.T
    h = -q / 2 - Qs.sum(axis=1) / 4
    return h, Q / 4


def family(name, rng, n=N_SPINS):
    if name == "sk":
        J = np.triu(rng.normal(0, 1, (n, n)), 1)
        return rng.normal(0, 1, n), J
    if name == "shed":
        L = rng.lognormal(np.log(12), 0.5, n)          # feeder loads, MW
        S = 0.4 * L.sum()                                # shortfall to shed
        A = 1.0 / S ** 2 * 20
        c = rng.uniform(0, 1, n)                          # per-feeder harm (critical loads, recent cuts)
        q = A * (L ** 2 - 2 * S * L) + c
        Q = 2 * A * np.outer(L, L)
        return qubo_to_ising(q, Q)
    if name == "partition":
        w = rng.uniform(1, 10, n)
        return 0.01 * rng.normal(0, 1, n), np.triu(np.outer(w, w), 1) * 2
    if name == "maxcut3":
        import networkx as nx
        G = nx.random_regular_graph(3, n, seed=int(rng.integers(1e9)))
        J = np.zeros((n, n))
        for a, b in G.edges():
            J[min(a, b), max(a, b)] = 1.0
        return 0.3 * rng.normal(0, 1, n), J
    if name == "tri_afm":
        J = np.zeros((n, n)); cols = n // 2
        idx = lambda r, c: r * cols + c
        for r in range(2):
            for c in range(cols):
                if c + 1 < cols:
                    J[idx(r, c), idx(r, c + 1)] = 1.0
                if r == 0:
                    J[idx(0, c), idx(1, c)] = 1.0
                    if c + 1 < cols:
                        J[idx(0, c + 1), idx(1, c)] = 1.0
        J = np.triu(J + J.T, 1)
        return 0.5 * rng.normal(0, 1, n), J
    if name == "rfim":
        J = np.zeros((n, n)); cols = n // 2
        for r in range(2):
            for c in range(cols):
                i = r * cols + c
                if c + 1 < cols:
                    J[i, i + 1] = -1.0
                if r == 0:
                    J[i, i + cols] = -1.0
        return rng.normal(0, 1.5, n), J
    if name in ("planar", "planar_bal"):
        # neighbourhood map: Delaunay triangulation of random centres; neighbours should be in different groups
        from scipy.spatial import Delaunay
        P = rng.uniform(0, 1, (n, 2))
        J = np.zeros((n, n))
        for s in Delaunay(P).simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = sorted((int(s[a]), int(s[b])))
                    J[i, j] = rng.uniform(0.5, 1.5)          # shared border length
        if name == "planar_bal":                              # mild balance of the load shed by each group
            L = rng.lognormal(0, 0.4, n); L /= L.mean()
            J = J + 0.15 * np.triu(np.outer(L, L), 1)
        return 0.3 * rng.normal(0, 1, n), J
    if name == "cliques":
        J = np.zeros((n, n))
        for g in (range(0, n // 2), range(n // 2, n)):
            for a in g:
                for b in g:
                    if a < b:
                        J[a, b] = 1.0
        for _ in range(3):                                 # a few links between the two groups
            a, b = rng.integers(0, n // 2), rng.integers(n // 2, n)
            J[a, b] = rng.choice([-1.0, 1.0]) * 0.5
        return 0.5 * rng.normal(0, 1, n), J
    raise KeyError(name)


def pair_proposal(n):
    N = 2 ** n; Q = np.zeros((N, N)); k = np.arange(N)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j in pairs:
        Q[k, k ^ ((1 << i) | (1 << j))] += 1.0 / len(pairs)
    return Q


def swap_proposal(n):
    """Pick a 1 and a 0 uniformly and exchange them (count-preserving, symmetric)."""
    N = 2 ** n; Q = np.zeros((N, N))
    for a in range(N):
        ones = [i for i in range(n) if a >> i & 1]
        zeros = [i for i in range(n) if not a >> i & 1]
        if not ones or not zeros:
            Q[a, a] = 1.0
            continue
        w = 1.0 / (len(ones) * len(zeros))
        for i in ones:
            for j in zeros:
                Q[a, a ^ (1 << i) ^ (1 << j)] += w
    return Q


def temperature_for(E, k_eff):
    """T such that exp(entropy of the Boltzmann distribution) = k_eff (bisection on log T)."""
    lo, hi = 1e-4 * np.ptp(E), 10 * np.ptp(E)
    for _ in range(80):
        T = np.sqrt(lo * hi)
        w = np.exp(-(E - E.min()) / T); p = w / w.sum()
        H = -np.sum(p[p > 0] * np.log(p[p > 0]))
        lo, hi = (T, hi) if np.exp(H) < k_eff else (lo, T)
    return float(np.sqrt(lo * hi))


def run():
    rng = np.random.default_rng(SEED)
    n = N_SPINS
    classical = {"local": local_proposal(n), "pair": pair_proposal(n), "swap": swap_proposal(n),
                 "uniform": uniform_proposal(n)}
    classical["mix"] = (classical["local"] + classical["pair"] + classical["uniform"]) / 3
    rows = []
    t0 = time.time()
    for fam in ["sk", "shed", "partition", "maxcut3", "tri_afm", "rfim", "cliques"]:
        for inst in range(INSTANCES):
            h, J = family(fam, rng)
            E = ising_energy_vector(h, J)
            g, t = draw_parameters(rng, DRAWS)
            Qq = quantum_proposal_exact(h, J, g, t)
            for k in K_EFFS:
                T = temperature_for(E, k)
                gaps = {name: metropolis_gap(Q, E, T) for name, Q in classical.items()}
                gaps["quantum"] = metropolis_gap(Qq, E, T)
                best_c = max(v for kk, v in gaps.items() if kk != "quantum")
                rows.append(dict(family=fam, instance=inst, k_eff=k, T=T, gaps=gaps,
                                 ratio_vs_best_classical=gaps["quantum"] / best_c,
                                 best_classical=max((kk for kk in gaps if kk != "quantum"), key=gaps.get)))
            print(f"{fam:9s} #{inst}  " + "  ".join(f"K{r['k_eff']}: x{r['ratio_vs_best_classical']:.2f} ({r['best_classical']})"
                                                     for r in rows[-len(K_EFFS):]) + f"  [{time.time() - t0:.0f}s]", flush=True)
    out = ROOT / "results" / "e8_screen.json"
    out.write_text(json.dumps(dict(n=n, instances=INSTANCES, k_effs=K_EFFS, draws=DRAWS, rows=rows), indent=1))
    print("\nmedian quantum / best-classical gap ratio (wins out of instances):")
    for fam in dict.fromkeys(r["family"] for r in rows):
        for k in K_EFFS:
            rs = [r["ratio_vs_best_classical"] for r in rows if r["family"] == fam and r["k_eff"] == k]
            print(f"  {fam:9s} K={k:3d}: median x{np.median(rs):.2f}  min x{np.min(rs):.2f}  wins {sum(x > 1 for x in rs)}/{len(rs)}")


if __name__ == "__main__":
    run()
