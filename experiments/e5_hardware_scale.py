"""E5: the quantum core at hardware scale: an exact 3-qubit reduction of a real LOLP instance.

Instance: 2030, Holyrood's oil units retired and nothing added, Labrador-Island Link at Hydro's planning
outage rate (5%). Reduction (gridrisk.quantum.three_qubit_reduction):
    q0 = "high-load hour" (net-load level in the top band), q1 = Labrador-Island Link forced out,
    q2 = shortfall, with P(q2 = 1 | q0, q1) = P(rest of the fleet cannot cover | ...) computed exactly,
so P(q2 = 1) equals the full instance's probability. A = RY, UCRY, UCRY; Q has 4-60 two-qubit gates.

Noisy runs use the IBM Heron r2 (FakeFez) noise model with the density-matrix method (exact noisy output
probabilities); repeated experiments are then drawn by binomial sampling. Each trial estimates the risk with
  * classical Monte Carlo at the same number of oracle queries,
  * plain MLAE (assumes a noiseless device),
  * noise-aware MLAE (Tanaka et al. 2021) on readout-corrected counts.
With --hardware the same circuits run on IBM Quantum (SamplerV2) and the counts are analysed the same way;
device, layout, job id and options are saved alongside the results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e2_reliability import demand_hours, forecast_errors, frazil_p, set_lil, wind_cf  # noqa: E402
from gridrisk import fleet  # noqa: E402
from gridrisk.instances import build_instance, lolh_from_instance  # noqa: E402
from gridrisk.quantum import (mlae_estimate, noise_aware_mlae, oracle_queries, three_qubit_circuits,  # noqa: E402
                              three_qubit_reduction)

POWERS = [0, 1, 2, 3, 4]
SHOTS = 1000
TRIALS = 400
SEED = 55
BIG = ["Labrador-Island Link", "Bay d'Espoir U7", "Hardwoods GT"]


def hardware_instance():
    vals, w, _ = forecast_errors()
    dh, _ = demand_hours(2030)
    p = set_lil(fleet.no_holyrood_2030(), 0.05)
    inst = build_instance(p, dh, (vals, w), wind_cf(), BIG, frazil=(459.0, frazil_p()))
    lil = inst.names.index("Labrador-Island Link")
    # "high-load" band: the levels from which a shortfall is possible even with the LIL in service
    cond = []
    for b in range(len(inst.p_level)):
        pats, pr = inst._pattern_probs(b)
        s = pats @ np.asarray(inst.cap) + inst.step * b >= inst.threshold
        cond.append(pr[(pats[:, lil] == 0) & s].sum() / max(pr[pats[:, lil] == 0].sum(), 1e-300))
    split = int(np.argmax(np.array(cond) > 1e-3))
    return inst, three_qubit_reduction(inst, lil, split), split


def readout_correct(p_meas: float, e01: float, e10: float) -> float:
    return float(np.clip((p_meas - e01) / max(1e-9, 1 - e01 - e10), 0, 1))


def calibration_circuits():
    from qiskit import QuantumCircuit
    c0 = QuantumCircuit(3, 1); c0.measure(2, 0)
    c1 = QuantumCircuit(3, 1); c1.x(2); c1.measure(2, 0)
    return [c0, c1]


def pick_layout(backend):
    """Three connected physical qubits a - b - c with the lowest 2q + readout error; flag qubit in the middle."""
    target = backend.target
    best, best_err = None, np.inf
    nbr = {}
    for a, b in backend.coupling_map.get_edges():
        nbr.setdefault(a, set()).add(b)
    gate = "cz" if "cz" in target else ("ecr" if "ecr" in target else "cx")
    for mid, ns in nbr.items():
        ns = sorted(ns)
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                a, c = ns[i], ns[j]
                err = 0.0
                for u, v in [(a, mid), (mid, a), (c, mid), (mid, c)]:
                    if (u, v) in target[gate] and target[gate][(u, v)] is not None:
                        err += target[gate][(u, v)].error or 0
                m = target["measure"][(mid,)] if (mid,) in target["measure"] else None
                err += (m.error or 0) if m else 0
                if err < best_err:
                    best, best_err = [a, c, mid], err
    return best


def exact_noisy_probs(circs, backend, layout):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    sim = AerSimulator.from_backend(backend, method="density_matrix")
    tc = transpile(circs, backend, initial_layout=layout, optimization_level=3, seed_transpiler=SEED)
    run_c = []
    for c in tc:
        c2 = c.remove_final_measurements(inplace=False)
        c2.save_probabilities_dict(qubits=[layout[2]])      # only the flag qubit: lets Aer drop idle qubits
        run_c.append(c2)
    res = sim.run(run_c, shots=1).result()
    probs = [float(sum(v for k, v in res.data(i)["probabilities"].items() if int(str(k), 0) == 1)) for i in range(len(run_c))]
    return probs, tc


def analyse(p_true, probs_by_power, e01, e10, rng, trials=TRIALS):
    rows = []
    for M in range(1, len(POWERS) + 1):
        pw = POWERS[:M]; shots = [SHOTS] * M
        q = oracle_queries(pw, shots)
        pl, na = [], []
        for _ in range(trials):
            hits = [int(rng.binomial(SHOTS, probs_by_power[k])) for k in range(M)]
            pl.append(mlae_estimate(hits, shots, pw))
            corr = [int(round(SHOTS * readout_correct(h / SHOTS, e01, e10))) for h in hits]
            na.append(noise_aware_mlae(corr, shots, pw)[0])
        mc = rng.binomial(q, p_true, size=trials) / q
        rel = lambda v: float(np.sqrt(np.mean((np.array(v) - p_true) ** 2)) / p_true)
        rows.append(dict(powers=pw, queries=q, mc_rel_rmse=rel(mc), plain_mlae_rel_rmse=rel(pl),
                         noise_aware_mlae_rel_rmse=rel(na), noise_aware_mean=float(np.mean(na))))
    return rows


def run(hardware=False, backend_name=None):
    from qiskit_ibm_runtime.fake_provider import FakeFez
    inst, red, split = hardware_instance()
    p_true = red["p"]
    assert abs(p_true - inst.exact()) < 1e-12
    theta = np.arcsin(np.sqrt(p_true))
    circs = three_qubit_circuits(red, POWERS)
    ideal = [float(np.sin((2 * m + 1) * theta) ** 2) for m in POWERS]
    fake = FakeFez()
    layout = pick_layout(fake)
    probs, tc = exact_noisy_probs(circs, fake, layout)
    ro = float(fake.target["measure"][(layout[2],)].error)
    e01 = e10 = ro                                            # symmetric readout error from the calibration snapshot
    p_read = [e01 * (1 - x) + (1 - e10) * x for x in probs]
    rng = np.random.default_rng(SEED)
    out = dict(instance="2030, Holyrood retired, nothing added, LIL 5% EqFOR", p_true=p_true,
               lolh_implied=lolh_from_instance(inst, p_true), level_split=split,
               reduction=dict(p_cold=red["p_cold"], p_out={str(k): v for k, v in red["p_out"].items()},
                              cond={str(k): v for k, v in red["cond"].items()}),
               powers=POWERS, shots_per_power=SHOTS, trials=TRIALS, layout=layout, ideal=ideal,
               fake_fez=dict(p_gate_noise=probs, p_with_readout=p_read, readout_error=ro,
                             two_qubit_gates=[int(sum(v for k, v in c.count_ops().items() if k in ("cz", "cx", "ecr"))) for c in tc],
                             depth=[int(c.depth()) for c in tc], comparison=analyse(p_true, p_read, e01, e10, rng)))
    if hardware:
        out["hardware"] = run_hardware(circs, p_true, backend_name)
        hw = out["hardware"]
        print(f"HARDWARE {hw['backend']} job {hw['job_id']}: measured {np.round(hw['measured'], 4)}  "
              f"predicted {np.round(hw['predicted_from_device_noise_model'], 4)}  readout e01={hw['readout']['e01']:.4f} e10={hw['readout']['e10']:.4f}")
        for r in hw["comparison"]:
            print(f"  powers {r['powers']}: queries {r['queries']:6d}  MC {r['mc_rel_rmse_theory']:.3f}  plain {r['plain_mlae_rel_rmse']:.3f}  "
                  f"noise-aware {r['noise_aware_mlae_rel_rmse']:.3f}")
    path = ROOT / "results" / ("e5_hardware.json" if hardware else "e5_hardware_scale_noise.json")
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"P(shortfall) = {p_true:.4f} (LOLH {out['lolh_implied']:.1f} h); layout {layout}; "
          f"2q gates per power {out['fake_fez']['two_qubit_gates']}")
    print("reduction:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["reduction"].items()})
    print("ideal          ", np.round(ideal, 4))
    print("FakeFez (+RO)  ", np.round(p_read, 4))
    for r in out["fake_fez"]["comparison"]:
        print(f"  powers {r['powers']}: queries {r['queries']:6d}  MC {r['mc_rel_rmse']:.3f}  plain MLAE {r['plain_mlae_rel_rmse']:.3f}  "
              f"noise-aware MLAE {r['noise_aware_mlae_rel_rmse']:.3f}")
    return out


HW_TRIALS = 10      # independent repeats of SHOTS shots per circuit, so the error can be measured on hardware


def run_hardware(circs, p_true, backend_name):
    """Run the same circuits on an IBM Quantum device: HW_TRIALS x SHOTS shots per circuit plus two readout
    calibration circuits, in one SamplerV2 job. Each repeat gives one estimate per budget; the spread of the
    repeats around the exact value gives the relative RMSE, compared with Monte Carlo at the same queries.
    The device's own calibration is also turned into a noise model to predict the result."""
    import time as _time
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    service = QiskitRuntimeService()
    backend = service.backend(backend_name) if backend_name else service.least_busy(operational=True, simulator=False)
    layout = pick_layout(backend)
    allc = circs + calibration_circuits()
    tc = transpile(allc, backend, initial_layout=layout, optimization_level=3, seed_transpiler=SEED)
    predicted, _ = exact_noisy_probs(circs, backend, layout)            # noise model built from this device's calibration
    ro = float(backend.target["measure"][(layout[2],)].error or 0.0)
    predicted = [ro * (1 - x) + (1 - ro) * x for x in predicted]
    sampler = SamplerV2(mode=backend)
    sampler.options.twirling.enable_gates = True
    sampler.options.dynamical_decoupling.enable = True
    n = SHOTS * HW_TRIALS
    t0 = _time.time()
    job = sampler.run([(c,) for c in tc], shots=n)
    print(f"submitted job {job.job_id()} to {backend.name}; waiting...", flush=True)
    res = job.result()
    bits = [np.array([int(b) for b in r.data.c.get_bitstrings()]) for r in res]
    e01, e10 = float(bits[-2].mean()), float(1 - bits[-1].mean())
    p_meas = [float(b.mean()) for b in bits[:-2]]
    rel = lambda v: float(np.sqrt(np.mean((np.array(v) - p_true) ** 2)) / p_true)
    rows = []
    for M in range(1, len(POWERS) + 1):
        pw = POWERS[:M]
        na, pl = [], []
        for t in range(HW_TRIALS):
            hits = [int(bits[k][t * SHOTS:(t + 1) * SHOTS].sum()) for k in range(M)]
            corr = [int(round(SHOTS * readout_correct(h / SHOTS, e01, e10))) for h in hits]
            na.append(noise_aware_mlae(corr, [SHOTS] * M, pw)[0])
            pl.append(mlae_estimate(hits, [SHOTS] * M, pw))
        q = oracle_queries(pw, [SHOTS] * M)
        rows.append(dict(powers=pw, queries=q, noise_aware_mlae_rel_rmse=rel(na), plain_mlae_rel_rmse=rel(pl),
                         noise_aware_mean=float(np.mean(na)), mc_rel_rmse_theory=float(np.sqrt((1 - p_true) / (p_true * q)))))
    usage = None
    try:
        usage = job.usage()
    except Exception:
        pass
    return dict(backend=backend.name, job_id=job.job_id(), layout=layout, shots_per_circuit=n, trials=HW_TRIALS,
                shots_per_trial=SHOTS, measured=p_meas, predicted_from_device_noise_model=predicted, ideal_note="see 'ideal'",
                readout=dict(e01=e01, e10=e10), comparison=rows, wall_seconds=time_or_none(t0), qpu_usage_seconds=usage,
                two_qubit_gates=[int(sum(v for k, v in c.count_ops().items() if k in ("cz", "cx", "ecr"))) for c in tc],
                options="SamplerV2, gate twirling, dynamical decoupling, optimization_level 3; readout calibration circuits in the same job")


TWIN_COND = {"low": {(0, 0): 0.0004, (0, 1): 0.0011, (1, 0): 0.0007, (1, 1): 0.0016}}
TWIN_COND["high"] = {k: 1 - v for k, v in TWIN_COND["low"].items()}


def twin_reduction(red: dict, cond: dict) -> dict:
    """Same circuit structure as `red`, different flag rotations: an exactly known answer near 0 or near 1."""
    p = sum((red["p_cold"] if c else 1 - red["p_cold"]) * (red["p_out"][c] if o else 1 - red["p_out"][c]) * cond[(c, o)]
            for c in (0, 1) for o in (0, 1))
    return dict(p_cold=red["p_cold"], p_out=red["p_out"], cond=cond, p=p)


def run_hardware_twin(backend_name=None):
    """Twin-circuit calibration on real hardware (designed before seeing its results).

    For every Grover power m, the device maps the ideal flag probability P to an observed one. Two twin circuits
    with identical gates and layout but exactly known answers near 0 and near 1 measure that map (assumed affine
    per m: P_obs = a_m P + b_m, which covers readout error and depolarising-like noise after twirling). The target
    circuit's readings are then corrected, P = (P_obs - b_m) / a_m, and fed to standard maximum-likelihood AE.
    Results are reported with the calibration shared across many estimates (amortised) and paid per estimate.
    """
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    inst, red, split = hardware_instance()
    p_true = red["p"]
    reds = {"target": red, "low": twin_reduction(red, TWIN_COND["low"]), "high": twin_reduction(red, TWIN_COND["high"])}
    ideal = {k: [float(np.sin((2 * m + 1) * np.arcsin(np.sqrt(r["p"]))) ** 2) for m in POWERS] for k, r in reds.items()}
    service = QiskitRuntimeService()
    backend = service.backend(backend_name) if backend_name else service.least_busy(operational=True, simulator=False)
    layout = pick_layout(backend)
    circs = {k: three_qubit_circuits(r, POWERS) for k, r in reds.items()}
    order = ["target", "low", "high"]
    allc = [c for k in order for c in circs[k]]
    tc = transpile(allc, backend, initial_layout=layout, optimization_level=3, seed_transpiler=SEED)
    twoq = [int(sum(v for kk, v in c.count_ops().items() if kk in ("cz", "cx", "ecr"))) for c in tc]
    L = len(POWERS)
    cz = {k: twoq[i * L:(i + 1) * L] for i, k in enumerate(order)}
    print("two-qubit gates per power:", cz, flush=True)
    sampler = SamplerV2(mode=backend)
    sampler.options.twirling.enable_gates = True
    sampler.options.dynamical_decoupling.enable = True
    n = SHOTS * HW_TRIALS
    job = sampler.run([(c,) for c in tc], shots=n)
    print(f"submitted job {job.job_id()} to {backend.name}; waiting...", flush=True)
    res = job.result()
    bits = [np.array([int(b) for b in r.data.c.get_bitstrings()]) for r in res]
    obs = {k: [bits[i * L + j] for j in range(L)] for i, k in enumerate(order)}
    pooled = {k: [float(b.mean()) for b in v] for k, v in obs.items()}

    def affine(p_low_obs, p_high_obs, m):
        a = (p_high_obs - p_low_obs) / (ideal["high"][m] - ideal["low"][m])
        return a, p_low_obs - a * ideal["low"][m]

    rel = lambda v: float(np.sqrt(np.mean((np.array(v) - p_true) ** 2)) / p_true)
    rows = []
    for M in range(1, L + 1):
        pw = POWERS[:M]
        est_amort, est_paid = [], []
        for t in range(HW_TRIALS):
            sl = slice(t * SHOTS, (t + 1) * SHOTS)
            hits_amort, hits_paid = [], []
            for m in range(M):
                pt = obs["target"][m][sl].mean()
                a1, b1 = affine(pooled["low"][m], pooled["high"][m], m)                       # shared calibration
                a2, b2 = affine(obs["low"][m][sl].mean(), obs["high"][m][sl].mean(), m)       # calibration paid per estimate
                hits_amort.append(int(round(SHOTS * float(np.clip((pt - b1) / a1, 0, 1)))))
                hits_paid.append(int(round(SHOTS * float(np.clip((pt - b2) / a2, 0, 1)))))
            est_amort.append(mlae_estimate(hits_amort, [SHOTS] * M, pw))
            est_paid.append(mlae_estimate(hits_paid, [SHOTS] * M, pw))
        q = oracle_queries(pw, [SHOTS] * M)
        rows.append(dict(powers=pw, queries=q, queries_with_twins=3 * q,
                         twin_amortised_rel_rmse=rel(est_amort), twin_paid_rel_rmse=rel(est_paid),
                         twin_amortised_mean=float(np.mean(est_amort)),
                         mc_rel_rmse_same_queries=float(np.sqrt((1 - p_true) / (p_true * q))),
                         mc_rel_rmse_3x_queries=float(np.sqrt((1 - p_true) / (p_true * 3 * q)))))
    usage = None
    try:
        usage = job.usage()
    except Exception:
        pass
    out = dict(backend=backend.name, job_id=job.job_id(), layout=layout, p_true=p_true, twins={k: reds[k]["p"] for k in ("low", "high")},
               shots_per_circuit=n, trials=HW_TRIALS, shots_per_trial=SHOTS, ideal=ideal, measured=pooled,
               corrected_target=[float(np.clip((pooled["target"][m] - affine(pooled["low"][m], pooled["high"][m], m)[1])
                                               / affine(pooled["low"][m], pooled["high"][m], m)[0], 0, 1)) for m in range(L)],
               affine={str(m): dict(zip(("a", "b"), affine(pooled["low"][m], pooled["high"][m], m))) for m in range(L)},
               comparison=rows, two_qubit_gates=cz, qpu_usage_seconds=usage,
               options="SamplerV2, gate twirling, dynamical decoupling, optimization_level 3, same layout and transpilation for target and twins")
    (ROOT / "results" / "e5_hardware_twin.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"TWIN {out['backend']} job {out['job_id']}  p_true {p_true:.4f}")
    print("  measured target ", np.round(pooled["target"], 4), "\n  corrected target", np.round(out["corrected_target"], 4),
          "\n  ideal target    ", np.round(ideal["target"], 4))
    for r in rows:
        print(f"  powers {r['powers']}: queries {r['queries']:6d}  MC {r['mc_rel_rmse_same_queries']:.3f}  twin (shared calib.) "
              f"{r['twin_amortised_rel_rmse']:.3f} | with twins paid: queries {r['queries_with_twins']:6d}  MC {r['mc_rel_rmse_3x_queries']:.3f}  "
              f"twin {r['twin_paid_rel_rmse']:.3f}")
    return out


def time_or_none(t0):
    import time as _time
    return float(_time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", action="store_true")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--twin", action="store_true", help="twin-circuit calibrated run on hardware")
    a = ap.parse_args()
    if a.twin:
        run_hardware_twin(a.backend)
    else:
        run(a.hardware, a.backend)
