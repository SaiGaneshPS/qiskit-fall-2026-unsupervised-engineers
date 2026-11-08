"""E8i: the St. John's rotation sampler at hardware size (8 neighbourhoods, 256 rotation plans).

Design fixed by E8d-E8h: ONE line of qubits carrying a fitted Ising model; short evolution t in [1, 6]
(at most 8 Trotter steps, at most 98 two-qubit gates); SPAM twirling; plus the free "swap the two groups" move
(probability 1/4) that the strongest classical proposal also gets.

Stages
  check      exact spectral gaps for the six E8d seed areas at n = 8 (no circuits)
  simulate   proposal-matrix tomography on the IBM Heron noise model (Aer density matrix) for AREA
  hardware   the same tomography on an IBM Quantum device (one SamplerV2 job), then the same analysis
The device-level Metropolis chain follows exactly from the measured proposal matrix, so its spectral gap,
stationary-distribution bias and marginal bias are reported against the classical proposals.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import transpile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridrisk.qmcmc.mcmc import ising_energy_vector, metropolis_gap, quantum_proposal_exact  # noqa: E402
from gridrisk.qmcmc.proposal import GAMMA_RANGE, trotter_proposal_matrix  # noqa: E402
from gridrisk.qmcmc.surrogate import best_line_pairs  # noqa: E402
from gridrisk.qmcmc.tomography import accumulate, chain_diagnostics, normalise, parameter_table, proposal_jobs  # noqa: E402
from gridrisk.rotation import ising, load_city  # noqa: E402
from experiments.e8_screen import K_EFFS, temperature_for  # noqa: E402
from experiments.e8d_stjohns_gaps import FAIRNESS, SEEDS, classical_kernels  # noqa: E402
from experiments.e8e_noise import line_layout, path_order, subsample  # noqa: E402
from experiments.e8g_fitted_lines import fitted_line, flip_kernel  # noqa: E402

N = 8
AREA = "Downtown"
DRAWS = 24
KEYS = 2
SIM_SHOTS = 256
HW_SHOTS = 32
K_FIT = 16             # the fitted line is fitted once, at the K_EFF = 16 temperature, and used for both K
SEED = 888


def setup(area, rng):
    city = load_city()
    idx = city.subregion(area, N)
    h, J = ising(city, idx, fairness=FAIRNESS * rng.normal(0, 1, N))
    E = ising_energy_vector(h, J)
    X = ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1).astype(float)
    pairs, order = best_line_pairs(J)
    hf, Jf = fitted_line(E, X, pairs, 2 * temperature_for(E, K_FIT))
    g, t = rng.uniform(*GAMMA_RANGE, DRAWS), rng.uniform(1, 6, DRAWS)
    return city, idx, h, J, E, X, pairs, hf, Jf, g, t


def check():
    classical = classical_kernels(N)
    F = flip_kernel(N)
    for area in SEEDS:
        rng = np.random.default_rng(SEED)
        city, idx, h, J, E, X, pairs, hf, Jf, g, t = setup(area, rng)
        Qc = trotter_proposal_matrix(hf, Jf, g, t, pairs=pairs)
        ks = {"line_fit": quantum_proposal_exact(hf, Jf, g, t), "circuit_fit": Qc, "circuit_fit_flip": 0.75 * Qc + 0.25 * F}
        out = []
        for kk in K_EFFS:
            T = temperature_for(E, kk)
            cg = {nm: metropolis_gap(Q, E, T) for nm, Q in classical.items()}
            best = max(cg, key=cg.get)
            out.append(f"K{kk}: best {best} " + " ".join(f"{nm} x{metropolis_gap(0.5 * (Q + Q.T), E, T) / cg[best]:.2f}"
                                                    for nm, Q in ks.items()))
        print(f"{area:16s} " + " | ".join(out), flush=True)


def settings_by_depth(hf, Jf, pairs, g, t, rng):
    return sorted(proposal_jobs(hf, Jf, pairs, g, t, KEYS, rng).items())


def analyse(name, Q, E, X, extra, per_row=None):
    """Gaps vs the exact classical proposals and, if per_row is given, vs classical proposals estimated from the
    same number of samples per row as the quantum one (finite sampling adds spurious moves that can flatter any
    chain, so this is the like-for-like comparison). The swap move is exact on both sides."""
    classical = classical_kernels(N)
    F = flip_kernel(N)
    rng = np.random.default_rng(SEED + 3)
    sampled = {}
    if per_row:
        mix = (classical["local"] + classical["pair"] + classical["uniform"]) / 3
        sampled = {nm: subsample(Qc, per_row, rng) for nm, Qc in classical.items() if nm != "mix_flip"}
        sampled["mix_flip"] = 0.75 * subsample(mix, per_row, rng) + 0.25 * F
    rows = []
    for kk in K_EFFS:
        T = temperature_for(E, kk)
        row = dict(k_eff=kk, T=T, classical={nm: metropolis_gap(Qc, E, T) for nm, Qc in classical.items()})
        best = max(row["classical"], key=row["classical"].get)
        row["best_classical"] = best
        if sampled:
            from gridrisk.qmcmc.mcmc import spectral_gap
            from gridrisk.qmcmc.tomography import metropolis_from_empirical
            row["classical_same_samples"] = {nm: spectral_gap(metropolis_from_empirical(Qc, E, T), None)
                                             for nm, Qc in sampled.items()}
            best_s = max(row["classical_same_samples"], key=row["classical_same_samples"].get)
            row["best_classical_same_samples"] = best_s
        for nm, Qk in {**extra, name: Q, name + "_flip": 0.75 * Q + 0.25 * F}.items():
            row[nm] = chain_diagnostics(Qk, E, T, X)
            row[nm]["ratio_vs_best_classical"] = row[nm]["gap"] / row["classical"][best]
            if sampled:
                row[nm]["ratio_vs_best_classical_same_samples"] = (
                    row[nm]["gap"] / row["classical_same_samples"][row["best_classical_same_samples"]])
        rows.append(row)
        print(f"K={kk}: best classical {best} {row['classical'][best]:.3e} | " + "  ".join(
            f"{nm} x{row[nm]['ratio_vs_best_classical']:.2f}"
            + (f" (same-samples x{row[nm]['ratio_vs_best_classical_same_samples']:.2f})" if sampled else "")
            + f" (TVD {row[nm]['tvd_stationary']:.3f})"
            for nm in row if isinstance(row[nm], dict) and "gap" in row[nm]), flush=True)
    return rows


def layout_for(backend, pairs):
    phys = line_layout(backend, N)
    order = path_order(pairs, N)
    layout = [None] * N
    for k, unit in enumerate(order):
        layout[unit] = phys[k]
    return phys, layout


def simulate(area=AREA):
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeFez
    rng = np.random.default_rng(SEED)
    city, idx, h, J, E, X, pairs, hf, Jf, g, t = setup(area, rng)
    fake = FakeFez()
    sim = AerSimulator.from_backend(fake, method="density_matrix", seed_simulator=SEED)
    phys, layout = layout_for(fake, pairs)
    Qc = np.zeros((2 ** N, 2 ** N))
    for r, (tpl, st) in settings_by_depth(hf, Jf, pairs, g, t, np.random.default_rng(SEED + 1)):
        tqc = transpile(tpl.circuit, sim, initial_layout=layout, optimization_level=1, seed_transpiler=SEED)
        vals = parameter_table(tpl, hf, Jf, g, t, st)
        params = [p for p in tpl.parameters if p in set(tqc.parameters)]
        pos = {p: j for j, p in enumerate(tpl.parameters)}
        res = sim.run([tqc], parameter_binds=[{p: vals[:, pos[p]].tolist() for p in params}], shots=SIM_SHOTS).result()
        accumulate(Qc, st, [res.get_counts(i) for i in range(len(st))], N)
        print(f"  r={r}: {len(st)} settings", flush=True)
    Q_noisy = normalise(Qc)
    Q_budget = subsample(Q_noisy, DRAWS * KEYS * HW_SHOTS, np.random.default_rng(SEED + 2))
    Q_exact = trotter_proposal_matrix(hf, Jf, g, t, pairs=pairs)
    rows = analyse("noisy", Q_noisy, E, X, {"ideal_exact": Q_exact, "noisy_budget": Q_budget})
    out = dict(area=[city.names[i] for i in idx], n=N, line=[city.names[idx[i]] for i in path_order(pairs, N)],
               backend=fake.name, physical_path=phys, draws=DRAWS, keys=KEYS, shots=SIM_SHOTS, budget_shots=HW_SHOTS, results=rows)
    (ROOT / "results" / "e8i_simulate.json").write_text(json.dumps(out, indent=1))


def hardware(area=AREA, backend_name="ibm_quebec"):
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    rng = np.random.default_rng(SEED)
    city, idx, h, J, E, X, pairs, hf, Jf, g, t = setup(area, rng)
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    phys, layout = layout_for(backend, pairs)
    groups = settings_by_depth(hf, Jf, pairs, g, t, np.random.default_rng(SEED + 1))
    pubs, meta, twoq = [], [], {}
    for r, (tpl, st) in groups:
        tqc = transpile(tpl.circuit, backend, initial_layout=layout, optimization_level=1, seed_transpiler=SEED)
        twoq[r] = int(sum(v for k, v in tqc.count_ops().items() if k in ("cz", "ecr", "cx")))
        vals = parameter_table(tpl, hf, Jf, g, t, st)
        pos = {p: j for j, p in enumerate(tpl.parameters)}
        vals_t = np.column_stack([vals[:, pos[p]] for p in tqc.parameters])
        pubs.append((tqc, vals_t))
        meta.append(st)
    print(f"{sum(len(s) for s in meta)} settings in {len(pubs)} groups; two-qubit gates by Trotter steps {twoq}", flush=True)
    sampler = SamplerV2(mode=backend)
    sampler.options.twirling.enable_gates = True
    sampler.options.dynamical_decoupling.enable = True
    t0 = time.time()
    job = sampler.run(pubs, shots=HW_SHOTS)
    print(f"submitted job {job.job_id()} to {backend.name}; waiting...", flush=True)
    res = job.result()
    Qc = np.zeros((2 ** N, 2 ** N))
    for pr, st in zip(res, meta):
        ba = pr.data.c
        accumulate(Qc, st, [ba.get_counts(loc=i) for i in range(len(st))], N)
    Q_hw = normalise(Qc)
    Q_exact = trotter_proposal_matrix(hf, Jf, g, t, pairs=pairs)
    rows = analyse("hardware", Q_hw, E, X, {"ideal_exact": Q_exact}, per_row=DRAWS * KEYS * HW_SHOTS)
    usage = None
    try:
        usage = job.usage() or service.job(job.job_id()).usage()
    except Exception:
        pass
    out = dict(area=[city.names[i] for i in idx], n=N, line=[city.names[idx[i]] for i in path_order(pairs, N)],
               backend=backend.name, job_id=job.job_id(), layout=layout, physical_path=phys, draws=DRAWS, keys=KEYS,
               shots_per_setting=HW_SHOTS, two_qubit_gates_by_trotter_steps=twoq, qpu_usage_seconds=usage,
               wall_seconds=time.time() - t0, results=rows,
               options="SamplerV2, gate twirling, dynamical decoupling, optimization_level 1 on a fixed line layout; SPAM twirling keys")
    (ROOT / "results" / "e8i_hardware.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(ROOT / "results" / "e8i_hardware_Q.npz", hardware=Q_hw, ideal=Q_exact)
    print(f"QPU seconds {usage}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["check", "simulate", "hardware"])
    ap.add_argument("--area", default=AREA)
    ap.add_argument("--backend", default="ibm_quebec")
    a = ap.parse_args()
    {"check": check, "simulate": lambda: simulate(a.area), "hardware": lambda: hardware(a.area, a.backend)}[a.stage]()
