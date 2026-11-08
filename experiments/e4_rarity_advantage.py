"""E4: the safer the grid, the harder it is to prove it is safe, and the bigger the quantum advantage.

Every portfolio from E2 becomes a quantum LOLP instance (gridrisk.instances). For each one we compare the
number of samples / oracle queries needed to estimate P(shortfall in a winter hour) to 10% relative RMSE:
  * Monte Carlo:            N = (1 - p) / (p eps^2)   (exact formula)
  * tuned importance sampling (classical): outage probabilities raised and demand levels tilted, best of a
    grid of tilts chosen with the true per-sample variance from a 200,000-sample pilot; N = var / (p eps)^2
  * maximum-likelihood amplitude estimation: doubling Grover powers, 100 shots per power, 300 trials, using
    the ideal circuit statistics P(flag | m) = sin^2((2m+1) theta). E3 checks this law on the real 24-qubit
    circuits on Aer; here it lets us cover instances whose circuits would take hours to simulate.
Queries are counted as applications of A (one Monte Carlo sample = one simulated winter hour).
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e2_reliability import demand_hours, forecast_errors, frazil_p, set_lil, wind_cf  # noqa: E402
from gridrisk import fleet, scenarios, ml  # noqa: E402
from gridrisk.instances import build_instance, lolh_from_instance  # noqa: E402
from gridrisk.quantum import mlae_estimate_fine, oracle_queries, resource_counts, state_preparation  # noqa: E402
from gridrisk.reliability import Unit  # noqa: E402

EPS = 0.10
SHOTS = 100
TRIALS = 300
SEED = 44
TILTS = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
LAMS = [0.0, 0.1, 0.2, 0.3, 0.45]
IS_PILOT = 200_000


def qae_curve(p, rng, max_K):
    theta = np.arcsin(np.sqrt(p))
    pts = []
    for K in range(0, max_K + 1):
        powers = [0] + [2 ** j for j in range(K)]
        shots = [SHOTS] * len(powers)
        probs = [np.sin((2 * m + 1) * theta) ** 2 for m in powers]
        est = np.array([mlae_estimate_fine([int(rng.binomial(s, pr)) for s, pr in zip(shots, probs)], shots, powers)
                        for _ in range(TRIALS)])
        pts.append((oracle_queries(powers, shots), float(np.sqrt(np.mean((est - p) ** 2)) / p)))
    return pts


def queries_for(pts, eps):
    q = np.array([a for a, _ in pts], float)
    e = np.array([b for _, b in pts], float)
    below = np.where(e <= eps)[0]
    if len(below) == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return float(q[0])
    x0, x1, y0, y1 = np.log(q[i - 1]), np.log(q[i]), np.log(e[i - 1]), np.log(e[i])
    return float(np.exp(x0 + (np.log(eps) - y0) * (x1 - x0) / (y1 - y0)))


def instances():
    vals, w, _ = forecast_errors()
    err = (vals, w)
    cf = wind_cf()
    fz = (459.0, frazil_p())
    dh26, _ = demand_hours(2026)
    dh30, _ = demand_hours(2030)
    lin = ml.LinearDegreeHour().fit(ml.dataset(), ml.dataset().demand_mw)
    ua = scenarios.ua_per_home_kw(float(lin.m.coef_[0] + lin.m.coef_[1]))
    bb26 = dh26.copy(); bb26["demand_mw"] += scenarios.demand_delta_mw(dh26.T_island.values, ua, baseboard_to_hp=50_000)
    bb30 = dh30.copy(); bb30["demand_mw"] += scenarios.demand_delta_mw(dh30.T_island.values, ua, baseboard_to_hp=50_000)
    u8 = Unit("Bay d'Espoir U8 (new)", 150, fleet.HYDRO_DAFOR, "hydro", cold_sensitive=True)
    base30 = fleet.no_holyrood_2030()
    big26 = ["Labrador-Island Link", "Bay d'Espoir U7", "Holyrood U3", "Hardwoods GT", "Stephenville GT"]
    big30 = ["Labrador-Island Link", "Bay d'Espoir U7", "Hardwoods GT", "Bay d'Espoir U8 (new)", "Avalon CT (new, diesel)"]
    cases = [
        ("2030 Holyrood retired, nothing added (LIL 5%)", set_lil(base30, 0.05), dh30, 0),
        ("2030 Holyrood retired, nothing added (LIL 1.1%)", set_lil(base30, 0.011), dh30, 0),
        ("2030 Hydro's plan (LIL 5%)", set_lil(fleet.hydro_plan_2030(), 0.05), dh30, 0),
        ("2026-27 system, Holyrood at 34% outage rate", None, dh26, 0),
        ("2030 U8 + 300 MW wind + 50k heat pumps (LIL 1.1%)", set_lil(base30.plus(u8), 0.011), bb30, 300),
        ("2026-27 system", fleet.today(), dh26, 0),
        ("2026-27 system + 50k heat pumps", fleet.today(), bb26, 0),
        ("2030 U8 + 600 MW wind + 50k heat pumps (LIL 1.1%)", set_lil(base30.plus(u8), 0.011), bb30, 600),
        ("2026-27 system + 50k heat pumps + 300 MW wind", fleet.today(), bb26, 300),
    ]
    from experiments.e2_reliability import with_holyrood_rate
    out = []
    for label, p, dh, wadd in cases:
        if p is None:
            p = with_holyrood_rate(fleet.today(), 0.34)
        p = replace(p, wind_mw=54.0 + wadd)
        big = [b for b in (big26 if "2026" in label else big30) if any(u.name == b for u in p.units)]
        out.append((label, build_instance(p, dh, err, cf, big, frazil=fz)))
    return out


def best_is(inst, rng, p):
    best = None
    for tilt in TILTS:
        for lam in LAMS:
            w = inst.importance_sample(IS_PILOT, rng, tilt, lam)
            rv = float(np.mean(w ** 2) / p ** 2 - 1)
            if w.sum() > 0 and (best is None or rv < best[2]):
                best = (tilt, lam, rv)
    return best


def run():
    rng = np.random.default_rng(SEED)
    rows = []
    for label, inst in instances():
        p = inst.exact()
        max_K = int(np.clip(np.ceil(np.log2(1.0 / (EPS * np.sqrt(p)))) + 2, 3, 13))
        pts = qae_curve(p, rng, max_K)
        n_qae = queries_for(pts, EPS)
        n_mc = (1 - p) / (p * EPS ** 2)
        tilt, lam, rv = best_is(inst, rng, p)
        n_is = rv / EPS ** 2
        A, flag = state_preparation(inst)
        res = resource_counts(A, flag)
        rows.append(dict(label=label, p=p, lolh=lolh_from_instance(inst), n_mc=n_mc, n_is=n_is, is_tilt=tilt, is_lambda=lam,
                         n_qae=n_qae, advantage_vs_mc=n_mc / n_qae, advantage_vs_is=n_is / n_qae, curve=pts,
                         units=inst.names, resources=res))
        print(f"{label:52s} p {p:.2e} (LOLH {rows[-1]['lolh']:6.2f})  MC {n_mc:11,.0f}  IS {n_is:8,.0f}  "
              f"QAE {n_qae:8,.0f}  vs MC {n_mc / n_qae:5.1f}x  vs IS {n_is / n_qae:5.2f}x  ({res['qubits']} qubits)", flush=True)
    ps = np.array([r["p"] for r in rows]); adv = np.array([r["advantage_vs_mc"] for r in rows])
    slope = float(np.polyfit(np.log10(ps), np.log10(adv), 1)[0])
    # precision crossover vs the best classical method: QAE ~ n_qae * EPS / eps, IS ~ n_is * (EPS / eps)^2,
    # so QAE needs fewer queries for any target precision finer than eps* = EPS * n_is / n_qae
    for r in rows:
        r["eps_crossover_vs_is"] = float(EPS * r["n_is"] / r["n_qae"])
    out = dict(eps=EPS, shots_per_power=SHOTS, trials=TRIALS, rows=rows, fitted_log_slope_advantage_vs_p=slope)
    (ROOT / "results" / "e4_rarity_advantage.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"advantage vs Monte Carlo ~ p^{slope:.2f}   (theory: p^-0.5)")
    return out


if __name__ == "__main__":
    run()
