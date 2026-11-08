"""E3: the quantum core on the real instance: every outage pattern and demand level in one 24-qubit state.

Instance: the 2026-27 island system (with Holyrood) in a random December-March hour of any of 46 weather
years, built by gridrisk.instances from the AI demand model, the AI's out-of-sample errors, weather-driven
wind and the exact outage convolution of all small units. Qubits: 5 net-load levels, 6 big-unit outages
(Labrador-Island Link, Bay d'Espoir U7, Holyrood U3, Hardwoods, Stephenville, and the Bay d'Espoir U1-6
frazil-ice shutdown whose probability depends on how cold the hour is), a 6-qubit sum register, the
shortfall flag and 6 clean ancillas.

1. Correctness: Aer statevector P(flag) vs classical enumeration of the same instance vs the full engine.
2. End-to-end maximum-likelihood amplitude estimation on Aer, using the *actual* 24-qubit circuits
   Q^m A for m = 0, 1, 2, 4, 8, 16, 32 (30,000 recorded shots each, split into 300 trials of 100 shots).
   Relative RMSE vs oracle queries, against Monte Carlo at the same number of samples.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e2_reliability import demand_hours, forecast_errors, frazil_p, wind_cf  # noqa: E402
from gridrisk import fleet  # noqa: E402
from gridrisk.instances import build_instance, lolh_from_instance, season_hours  # noqa: E402
from gridrisk.quantum import (mlae_circuits, mlae_estimate_fine, oracle_queries, resource_counts,  # noqa: E402
                              state_preparation, statevector_probability)
from gridrisk.reliability import lole  # noqa: E402

POWERS = [0, 1, 2, 4, 8, 16, 32]
SHOTS_PER_TRIAL = 100
TRIALS = 300
SEED = 2026
BIG_TODAY = ["Labrador-Island Link", "Bay d'Espoir U7", "Holyrood U3", "Hardwoods GT", "Stephenville GT"]


def today_instance():
    vals, w, _ = forecast_errors()
    dh, _ = demand_hours(2026)
    cf = wind_cf()
    p = fleet.today()
    inst = build_instance(p, dh, (vals, w), cf, BIG_TODAY, frazil=(459.0, frazil_p()))
    full = lole(p, dh, (vals, w), wind_cf=cf, common_mode=[("Bay d'Espoir", frazil_p())])
    ss = season_hours(dh.assign(l=full["lolp"].values))
    return inst, float(ss.l.groupby(ss.winter).sum().mean())


def run():
    inst, engine_lolh = today_instance()
    A, flag = state_preparation(inst)
    p_exact = inst.exact()
    t = time.time()
    p_sv = statevector_probability(A, flag)
    t_sv = time.time() - t
    res = resource_counts(A, flag)
    print(f"instance: {A.num_qubits} qubits; P(shortfall in a winter hour) exact {p_exact:.6e}  statevector {p_sv:.6e}"
          f"  -> LOLH {lolh_from_instance(inst):.3f} h vs full engine {engine_lolh:.3f} h")
    print("resources:", res)

    from qiskit import transpile
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="statevector")
    circs = mlae_circuits(A, flag, POWERS)
    shots = SHOTS_PER_TRIAL * TRIALS
    hits = np.zeros((TRIALS, len(POWERS)), int)
    runtime = []
    for j, c in enumerate(circs):
        t = time.time()
        r = sim.run(transpile(c, sim, optimization_level=0), shots=shots, memory=True, seed_simulator=SEED + j).result()
        mem = np.array([int(x) for x in r.get_memory()])
        hits[:, j] = mem.reshape(TRIALS, SHOTS_PER_TRIAL).sum(axis=1)
        runtime.append(time.time() - t)
        print(f"  m={POWERS[j]:2d}: P(flag) measured {mem.mean():.5f}  theory {np.sin((2 * POWERS[j] + 1) * np.arcsin(np.sqrt(p_exact))) ** 2:.5f}"
              f"  ({runtime[-1]:.0f} s)", flush=True)
    rng = np.random.default_rng(SEED)
    curve = []
    for M in range(1, len(POWERS) + 1):
        pw = POWERS[:M]
        sh = [SHOTS_PER_TRIAL] * M
        q = oracle_queries(pw, sh)
        est = np.array([mlae_estimate_fine(hits[t, :M], sh, pw) for t in range(TRIALS)])
        mc = rng.binomial(q, p_exact, size=TRIALS) / q
        rel = lambda v: float(np.sqrt(np.mean((v - p_exact) ** 2)) / p_exact)
        curve.append(dict(powers=pw, queries=q, qae_rel_rmse=rel(est), mc_rel_rmse=rel(mc),
                          mc_rel_rmse_theory=float(np.sqrt((1 - p_exact) / (p_exact * q))),
                          qae_median=float(np.median(est))))
        print(f"  powers {pw}: queries {q:7d}  MLAE rel. RMSE {curve[-1]['qae_rel_rmse']:.3f}   MC {curve[-1]['mc_rel_rmse']:.3f}")
    out = dict(n_qubits=A.num_qubits, units=inst.names, caps_quanta=inst.cap, quantum_mw=inst.quantum_mw,
               p_level=inst.p_level.tolist(), p_exact=p_exact, p_statevector=p_sv, statevector_seconds=t_sv,
               lolh_instance=lolh_from_instance(inst), lolh_full_engine=engine_lolh, resources=res,
               powers=POWERS, shots_per_trial=SHOTS_PER_TRIAL, trials=TRIALS, measured_p_flag=(hits.sum(0) / shots).tolist(),
               aer_seconds=runtime, curve=curve)
    (ROOT / "results" / "e3_quantum_core.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    run()
