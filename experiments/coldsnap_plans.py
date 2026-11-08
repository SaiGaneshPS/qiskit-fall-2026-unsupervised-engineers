"""The bottleneck: checking plans through a multi-day cold snap, where the exact classical shortcut stops working.

Single-hour risk has a fast exact formula. A cold snap does not: outages last for days, so every unit's state must be
followed from day to day, the number of possible histories doubles with every unit and every day, and planners fall
back on Monte Carlo, which needs very many runs for rare events. Here the quantum method is applied to exactly this
model (gridrisk.coldsnap, built in E6) for four 2030 plans at two link outage rates.

Question for each plan: what is the chance of at least one day short of power in the winter's worst 3-day cold snap?
We call a plan safe if that chance is at most 10%, close to the classic "one day in ten years" reliability standard.

For each plan and link rate:
  * truth: exact enumeration (still possible at this small size: 2 units over 3 days);
  * quantum: the real 22-qubit circuits Q^m A, m = 0, 1, 2, 4, 8, run on Qiskit Aer (20,000 shots per power), split
    into 200 independent trials and fitted by maximum likelihood (amplitude estimation, end to end);
  * classical: Monte Carlo with the same number of runs, and the runs each method needs for a confident call.
Plans differ in supply (Unit 8, the turbine, batteries, smart thermostats, wind counted at its output in the 1% of
hours with the highest demand) and in demand (heat pumps, at the snap's coldest temperature).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e4_rarity_advantage import qae_curve, queries_for  # noqa: E402
from experiments.e6_cold_snap import build, cold_snaps  # noqa: E402
from experiments.turbine_decision import bits, pick  # noqa: E402
from gridrisk import fleet, scenarios  # noqa: E402
from gridrisk.coldsnap import circuit  # noqa: E402
from gridrisk.portfolio import OPTIONS, battery_firm_mw  # noqa: E402
from gridrisk.quantum import mlae_circuits, mlae_estimate_fine, oracle_queries  # noqa: E402
from gridrisk.reliability import Unit  # noqa: E402

SAFE = 0.10
RATES = {"1.1%": 0.011, "5%": 0.05}
POWERS = [0, 1, 2, 4, 8]
SHOTS, TRIALS = 100, 200
SEED = 1107
OUT = ROOT / "results" / "coldsnap_plans.json"
WIND_PEAK_CF = {"wa": 0.522, "wb": 0.528, "ww": 0.355}      # results/e7_wind_sites.json, top 1% demand hours
EXISTING_WIND = 27 * 0.522 + 27 * 0.528                       # Fermeuse and St. Lawrence, 27 MW each


def plans() -> dict:
    strong = pick("3pct", False, "lolh")
    return {"Hydro's plan: Unit 8 + turbine": bits(["ct", "u8"]),
            "cheapest plan without the turbine": pick("1pct", False, "net_cost"),
            "strongest plan without the turbine": strong,
            "same, plus the turbine": strong | bits(["ct"])}


def instance(x, rate: float, snaps, ua: float):
    on = {k: bool(v) for (k, *_), v in zip(OPTIONS, x)}
    units = list(fleet.no_holyrood_2030().units)
    if on["ct"]:
        units.append(Unit("Avalon CT (new, diesel)", 150, fleet.CT_DAUFOP, "ct", "diesel"))
    if on["u8"]:
        units.append(Unit("Bay d'Espoir U8 (new)", 150, fleet.HYDRO_DAFOR, "hydro", cold_sensitive=True))
    bat = 25 * on["bat_small"] + 50 * on["bat_large"]
    if bat:
        units.append(Unit("Batteries (firm part)", battery_firm_mw(bat), 0.03, "battery"))
    if on["dr"]:
        units.append(Unit("Demand response", 40.0, 0.10, "demand_response"))
    wind = EXISTING_WIND + sum(100 * WIND_PEAK_CF[k] for k, key in (("wa", "wind_avalon"), ("wb", "wind_burin"), ("ww", "wind_west")) if on[key])
    units.append(Unit("Wind at cold peaks", wind, 0.0, "wind"))
    n_bb = 15000 * on["bb_small"] + 35000 * on["bb_large"]
    n_oil = 10000 * on["oil_small"] + 30000 * on["oil_large"]
    adj = []
    for s in snaps:
        d = float(scenarios.demand_delta_mw(np.array([s["tmin"]]), ua, baseboard_to_hp=n_bb, oil_to_hp=n_oil, oil_backup_below_c=-10)[0])
        adj.append(dict(s, peaks=[v + d for v in s["peaks"]]))
    return build(rate, adj, portfolio=fleet.Portfolio("plan", units))


def aer_trials(A, flag, rng):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="statevector")
    circs = mlae_circuits(A, flag, POWERS)
    hits = np.zeros((TRIALS, len(POWERS)), int)
    secs = []
    for j, c in enumerate(circs):
        t0 = time.time()
        mem = sim.run(transpile(c, sim, optimization_level=0), shots=SHOTS * TRIALS, memory=True, seed_simulator=SEED + j).result().get_memory()
        secs.append(time.time() - t0)
        bits_ = np.array([int(b) for b in mem]).reshape(TRIALS, SHOTS)
        hits[:, j] = bits_.sum(1)
    return hits, secs


def run():
    e2 = json.loads((ROOT / "results" / "e2_reliability.json").read_text())
    ua = e2["heat_pumps_2026"]["ua_kw_per_degree_per_home"]
    snaps = cold_snaps()
    rng = np.random.default_rng(SEED)
    out = dict(safe=SAFE, powers=POWERS, shots=SHOTS, trials=TRIALS, cases=[])
    for name, x in plans().items():
        for tag, rate in RATES.items():
            inst = instance(x, rate, snaps, ua)
            p = inst.exact()
            A, flag = circuit(inst)
            hits, secs = aer_trials(A, flag, rng)
            curve = []
            for k in range(1, len(POWERS) + 1):
                pw = POWERS[:k]
                est = np.array([mlae_estimate_fine(h[:k], [SHOTS] * k, pw) for h in hits])
                q = oracle_queries(pw, [SHOTS] * k)
                curve.append(dict(powers=pw, queries=q, qae_rel_rmse=float(np.sqrt(np.mean((est - p) ** 2)) / p),
                                  mc_rel_rmse=float(np.sqrt((1 - p) / (q * p)) / p * p), qae_mean=float(est.mean())))
            full = np.array([mlae_estimate_fine(h, [SHOTS] * len(POWERS), POWERS) for h in hits])
            q_full = oracle_queries(POWERS, [SHOTS] * len(POWERS))
            sd_q = float(full.std())
            mc = rng.binomial(q_full, p, TRIALS) / q_full
            sd_mc = float(np.sqrt(p * (1 - p) / q_full))
            call = lambda e, sd: "safe" if e + 2 * sd <= SAFE else "unsafe" if e - 2 * sd > SAFE else "unsure"
            truth = "safe" if p <= SAFE else "unsafe"
            need = max(abs(p - SAFE) / p / 2, 0.01)
            n_mc = (1 - p) / (p * need ** 2)
            max_K = int(np.clip(np.ceil(np.log2(1.0 / (need * np.sqrt(p)))) + 2, 3, 15))
            n_qae = queries_for(qae_curve(p, rng, max_K), need)
            case = dict(plan=name, link=tag, p_exact=p, p_independent_days=inst.exact(persistence=False),
                        qubits=A.num_qubits, cx=int(A.count_ops().get("cx", 0)), aer_seconds=secs,
                        measured_flag=(hits.sum(0) / (SHOTS * TRIALS)).tolist(),
                        theory_flag=[float(np.sin((2 * m + 1) * np.arcsin(np.sqrt(p))) ** 2) for m in POWERS],
                        curve=curve, runs=q_full, qae_mean=float(full.mean()), qae_sd=sd_q, mc_sd=sd_mc,
                        truth=truth, qae_calls={c: int(sum(call(e, sd_q) == c for e in full)) for c in ("safe", "unsafe", "unsure")},
                        mc_calls={c: int(sum(call(e, sd_mc) == c for e in mc)) for c in ("safe", "unsafe", "unsure")},
                        need=need, n_mc_needed=n_mc, n_qae_needed=n_qae, built=[OPTIONS[j][1] for j in range(len(OPTIONS)) if x[j]])
            out["cases"].append(case)
            print(f"{name[:34]:34s} link {tag:4s} | exact {p:.4f} ({truth}) | quantum {full.mean():.4f} +/- {sd_q:.4f} "
                  f"vs MC +/- {sd_mc:.4f} at {q_full:,} runs | to decide: MC {n_mc:,.0f} vs QAE {n_qae:,.0f} | "
                  f"{A.num_qubits} qubits, {case['cx']:,} CX, Aer {sum(secs):.0f} s", flush=True)
            OUT.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        e2 = json.loads((ROOT / "results" / "e2_reliability.json").read_text())
        snaps = cold_snaps()
        for name, x in plans().items():
            for tag, rate in RATES.items():
                inst = instance(x, rate, snaps, e2["heat_pumps_2026"]["ua_kw_per_degree_per_home"])
                print(f"{name[:34]:34s} {tag:4s} exact {inst.exact():.4f}  independent-days {inst.exact(persistence=False):.4f}  y0 {inst.y0.tolist()} w {inst.w_sev.tolist()}")
    else:
        run()
