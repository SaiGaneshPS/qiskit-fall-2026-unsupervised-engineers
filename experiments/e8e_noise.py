"""E8e: does the quantum advantage on the St. John's rotation problem survive realistic hardware noise?

Instance: 8 neighbourhoods around Downtown (hardware size). Hardware-native proposal: three qubit lines that
together cover the map's links, one picked at random per step; short evolution t in [1, 6].
Proposal-matrix tomography (as in Layden et al. Fig. 3): for every start state s and every (line, gamma, t),
run the circuit with SPAM twirling and histogram the outputs. This gives the Metropolis chain the device would
actually run, so its spectral gap, stationary-distribution bias and marginal bias follow exactly.

Kernels: ideal_exact (Trotter circuits, no noise), noisy (IBM Heron noise model, density matrix, twirled), and
noisy_budget (the noisy kernel re-estimated with the shot budget planned for hardware). Compared with the
classical proposals local, pair, uniform, mix and mix_flip at the E8 temperatures.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import transpile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridrisk.qmcmc.mcmc import ising_energy_vector, metropolis_gap  # noqa: E402
from gridrisk.qmcmc.proposal import GAMMA_RANGE, trotter_proposal_matrix  # noqa: E402
from gridrisk.qmcmc.tomography import accumulate, chain_diagnostics, normalise, parameter_table, proposal_jobs  # noqa: E402
from gridrisk.rotation import ising, load_city  # noqa: E402
from experiments.e8_screen import K_EFFS, temperature_for  # noqa: E402
from experiments.e8b_hardware_friendly import line_only  # noqa: E402
from experiments.e8c_multiline import cover_lines  # noqa: E402
from experiments.e8d_stjohns_gaps import FAIRNESS, classical_kernels  # noqa: E402

N = 8
SEED_AREA = "Downtown"
LINES = 3
DRAWS_PER_LINE = 8
KEYS = 2
SHOTS = 256            # per (start, line, draw, key) in simulation
BUDGET_SHOTS = 32      # per (start, line, draw, key) planned for hardware
SEED = 848


def path_order(pairs, n):
    """Vertex order along a line given its consecutive pairs."""
    nb = {i: [] for i in range(n)}
    for a, b in pairs:
        nb[a].append(b); nb[b].append(a)
    start = next(i for i in range(n) if len(nb[i]) == 1)
    order, prev = [start], None
    while len(order) < n:
        nxt = [v for v in nb[order[-1]] if v != prev][0]
        prev = order[-1]; order.append(nxt)
    return order


def line_layout(backend, n):
    """A path of n physical qubits with the lowest summed two-qubit error (greedy walk from every start)."""
    import networkx as nx
    target = backend.target
    G = nx.Graph()
    for a, b in backend.coupling_map.get_edges():
        err = None
        for op in ("cz", "ecr", "cx"):
            if op in target and (a, b) in target[op] and target[op][(a, b)] is not None:
                err = target[op][(a, b)].error
        G.add_edge(a, b, w=err if err is not None else 0.01)
    best, best_cost = None, np.inf
    for start in G.nodes:
        path = [start]
        while len(path) < n:
            nb = [(G[path[-1]][v]["w"], v) for v in G[path[-1]] if v not in path]
            if not nb:
                break
            path.append(min(nb)[1])
        if len(path) == n:
            cost = sum(G[path[i]][path[i + 1]]["w"] for i in range(n - 1))
            if cost < best_cost:
                best, best_cost = path, cost
    return best


def instance():
    rng = np.random.default_rng(SEED)
    city = load_city()
    idx = city.subregion(SEED_AREA, N)
    h, J = ising(city, idx, fairness=FAIRNESS * rng.normal(0, 1, N))
    lines, cover = cover_lines(J, LINES)
    g = rng.uniform(*GAMMA_RANGE, (LINES, DRAWS_PER_LINE))
    t = rng.uniform(1, 6, (LINES, DRAWS_PER_LINE))
    return city, idx, h, J, lines, cover, g, t


def tomography(sim, phys_path, h, J, lines, g, t, shots, rng):
    """Proposal matrix of the line mixture, estimated from circuit runs on `sim` (counts per setting)."""
    n = len(h)
    Q = np.zeros((2 ** n, 2 ** n))
    twoq = {}
    for li, pairs in enumerate(lines):
        Jl = line_only(J, pairs)
        order = path_order(pairs, n)
        layout = [None] * n
        for k, unit in enumerate(order):
            layout[unit] = phys_path[k]
        Qc = np.zeros_like(Q)
        for r, (tpl, settings) in sorted(proposal_jobs(h, Jl, pairs, g[li], t[li], KEYS, rng).items()):
            tqc = transpile(tpl.circuit, sim, initial_layout=layout, optimization_level=1, seed_transpiler=SEED)
            twoq[(li, r)] = int(sum(v for k, v in tqc.count_ops().items() if k in ("cz", "ecr", "cx")))
            vals = parameter_table(tpl, h, Jl, g[li], t[li], settings)
            params = [p for p in tpl.parameters if p in set(tqc.parameters)]
            pos = {p: j for j, p in enumerate(tpl.parameters)}
            res = sim.run([tqc], parameter_binds=[{p: vals[:, pos[p]].tolist() for p in params}], shots=shots).result()
            accumulate(Qc, settings, [res.get_counts(i) for i in range(len(settings))], n)
        Q += normalise(Qc) / len(lines)
        print(f"  line {li}: done", flush=True)
    return Q, twoq


def subsample(Q, per_row, rng):
    """Re-estimate each row of Q from `per_row` samples (the hardware shot budget)."""
    out = np.zeros_like(Q)
    for s in range(len(Q)):
        out[s] = rng.multinomial(per_row, Q[s] / Q[s].sum()) / per_row
    return out


def run():
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeFez
    t0 = time.time()
    city, idx, h, J, lines, cover, g, t = instance()
    E = ising_energy_vector(h, J)
    X = ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1).astype(float)
    fake = FakeFez()
    sim = AerSimulator.from_backend(fake, method="density_matrix", seed_simulator=SEED)
    phys = line_layout(fake, N)
    rng = np.random.default_rng(SEED + 1)
    Q_exact = sum(trotter_proposal_matrix(h, line_only(J, p), g[i], t[i], pairs=p) for i, p in enumerate(lines)) / LINES
    Q_noisy, twoq = tomography(sim, phys, h, J, lines, g, t, SHOTS, rng)
    per_row = LINES * DRAWS_PER_LINE * KEYS * BUDGET_SHOTS
    Q_budget = subsample(Q_noisy, per_row, rng)
    classical = classical_kernels(N)
    out = dict(area=[city.names[i] for i in idx], n=N, lines=[[list(map(int, p)) for p in l] for l in lines],
               coverage=cover, draws_per_line=DRAWS_PER_LINE, keys=KEYS, shots=SHOTS, budget_shots=BUDGET_SHOTS,
               budget_samples_per_row=per_row, backend=fake.name, physical_path=phys,
               two_qubit_gates={f"line{k[0]}_r{k[1]}": v for k, v in sorted(twoq.items())}, results=[])
    for kk in K_EFFS:
        T = temperature_for(E, kk)
        row = dict(k_eff=kk, T=T, classical={nm: metropolis_gap(Q, E, T) for nm, Q in classical.items()})
        best = max(row["classical"], key=row["classical"].get)
        row["best_classical"] = best
        for nm, Q in {"ideal_exact": Q_exact, "noisy": Q_noisy, "noisy_budget": Q_budget}.items():
            row[nm] = chain_diagnostics(Q, E, T, X)
            row[nm]["ratio_vs_best_classical"] = row[nm]["gap"] / row["classical"][best]
        out["results"].append(row)
        print(f"K={kk}: best classical {best} gap {row['classical'][best]:.3e} | " + "  ".join(
            f"{nm} gap {row[nm]['gap']:.3e} (x{row[nm]['ratio_vs_best_classical']:.2f}, stationary TVD {row[nm]['tvd_stationary']:.3f})"
            for nm in ("ideal_exact", "noisy", "noisy_budget")), flush=True)
    out["seconds"] = time.time() - t0
    (ROOT / "results" / "e8e_noise.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(ROOT / "results" / "e8e_kernels.npz", ideal=Q_exact, noisy=Q_noisy, budget=Q_budget)
    print(f"done in {out['seconds']:.0f}s; two-qubit gates per (line, Trotter steps): {out['two_qubit_gates']}")


if __name__ == "__main__":
    run()
