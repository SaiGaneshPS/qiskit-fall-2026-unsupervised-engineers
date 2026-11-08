"""Do we need the Avalon combustion turbine? The decision behind Lights On, Oil Off.

Part 1, break-even: blackout hours in 2030 (Holyrood's oil units retired) for four plans, as the
Labrador-Island Link's winter outage rate goes from 1% (about its measured rate) to 5% (Hydro's planning
value). Plans come from the E7 table of all 4,096 plans; blackout hours from the same validated engine.

Part 2, how sure do we need to be: a plan passes if its expected blackout hours stay under 2.8 h/yr. Near
the line the estimate has to be precise: we ask for a standard error of half the distance to the line (a
two-sigma call), and count the samples each method needs for that call on the real quantum instances:
Monte Carlo, tuned importance sampling, and maximum-likelihood amplitude estimation (ideal circuit
statistics, 300 trials per budget, as in E4).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e2_reliability import demand_hours, forecast_errors, frazil_p  # noqa: E402
from experiments.e4_rarity_advantage import best_is, qae_curve, queries_for  # noqa: E402
from experiments.e7_every_good_plan import IDX, load_table, site_cf  # noqa: E402
from gridrisk import fleet, ml, scenarios  # noqa: E402
from gridrisk.instances import build_instance, lolh_from_instance  # noqa: E402
from gridrisk.portfolio import OPTIONS, WIND_SITE, PlanModel, battery_firm_mw  # noqa: E402
from gridrisk.reliability import Portfolio, Unit  # noqa: E402

RATES = [0.011, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
CHECKS = [0.011, 0.02, 0.025, 0.03, 0.032, 0.034, 0.036, 0.04, 0.045]   # link rates for the sample counts
CRITERION = fleet.LOLH_CRITERION_H
SEED = 2030
OUT = ROOT / "results" / "turbine_decision.json"


def bits(keys) -> np.ndarray:
    x = np.zeros(len(OPTIONS), dtype=np.int8)
    x[[IDX[k] for k in keys]] = 1
    return x


def pick(tag: str, turbine: bool, by: str) -> np.ndarray:
    """Cheapest passing plan (by="net_cost") or most reliable plan (by="lolh") with or without the turbine."""
    t = load_table()
    t = t[(t.scenario == tag) & ((t.idx & 1) == int(turbine))]
    if by == "net_cost":
        t = t[t.lolh <= CRITERION]
    i = int(t.sort_values(by).idx.iloc[0])
    return ((i >> np.arange(len(OPTIONS))) & 1).astype(np.int8)


def plans() -> dict:
    strong = pick("3pct", False, "lolh")
    return {
        "Hydro's plan: Unit 8 + turbine": bits(["ct", "u8"]),
        "cheapest plan without the turbine": pick("1pct", False, "net_cost"),
        "strongest plan without the turbine": strong,
        "same, plus the turbine": strong | bits(["ct"]),
    }


def models(rates) -> dict:
    vals, w, _ = forecast_errors()
    dh, _ = demand_hours(2030)
    ds = ml.dataset()
    lin = ml.LinearDegreeHour().fit(ds, ds.demand_mw)
    ua = scenarios.ua_per_home_kw(float(lin.m.coef_[0] + lin.m.coef_[1]))
    cfs = {s: site_cf(s) for s in ("fermeuse", "stlawrence", "portauport")}
    return {r: PlanModel(dh, (vals, w), cfs, ua, r) for r in rates}


def crossing(rates, lolh) -> float | None:
    """Link outage rate at which a plan's blackout hours reach the criterion (linear interpolation)."""
    for r0, r1, a, b in zip(rates, rates[1:], lolh, lolh[1:]):
        if a <= CRITERION < b:
            return float(r0 + (CRITERION - a) * (r1 - r0) / (b - a))
    return None if lolh[0] > CRITERION else float("inf")


def instance(m: PlanModel, x, rate: float):
    c = m.choices(x)
    units = [fleet.lil(fleet.LIL_MW, rate) if u.name.startswith("Labrador") else u for u in fleet.no_holyrood_2030().units]
    if c["on"]["ct"]:
        units.append(Unit("Avalon CT (new, diesel)", 150, fleet.CT_DAUFOP, "ct", "diesel"))
    if c["on"]["u8"]:
        units.append(Unit("Bay d'Espoir U8 (new)", 150, fleet.HYDRO_DAFOR, "hydro", cold_sensitive=True))
    if c["bat_mw"]:
        units.append(Unit("Batteries (firm part)", battery_firm_mw(c["bat_mw"]), 0.03, "battery"))
    if c["on"]["dr"]:
        units.append(Unit("Demand response", 40.0, 0.10, "demand_response"))
    cf = 27 * m.wind_cf["fermeuse"] + 27 * m.wind_cf["stlawrence"]
    for k in c["winds"]:
        cf = cf + 100 * m.wind_cf[WIND_SITE[k]]
    big = [b for b in ("Labrador-Island Link", "Bay d'Espoir U7", "Hardwoods GT", "Bay d'Espoir U8 (new)",
                       "Avalon CT (new, diesel)") if any(u.name == b for u in units)]
    return build_instance(Portfolio("plan", units, wind_mw=1.0), m.demand(c["n_bb"], c["n_oil"]), m.err, cf, big,
                          frazil=(459.0, frazil_p()))


def samples_needed(inst, rng) -> dict:
    p = inst.exact()
    lolh = lolh_from_instance(inst)
    eps = max(abs(lolh - CRITERION) / lolh / 2, 0.01)
    max_K = int(np.clip(np.ceil(np.log2(1.0 / (eps * np.sqrt(p)))) + 2, 3, 15))
    n_qae = queries_for(qae_curve(p, rng, max_K), eps)
    _, _, rv = best_is(inst, rng, p)
    return dict(p=p, lolh=lolh, eps=eps, n_mc=(1 - p) / (p * eps ** 2), n_is=rv / eps ** 2, n_qae=n_qae)


def run():
    P = plans()
    M = models(sorted(set(RATES) | set(CHECKS)))
    curves = {}
    for name, x in P.items():
        curves[name] = [M[r].lolh(x) for r in RATES]
        print(f"{name:40s}", " ".join(f"{v:6.2f}" for v in curves[name]), flush=True)
    t = load_table()
    cost = {tag: {"with turbine": None, "without": None} for tag in ("1pct", "3pct", "5pct")}
    for tag in cost:
        s = t[(t.scenario == tag) & (t.lolh <= CRITERION)]
        for lab, ct in (("with turbine", 1), ("without", 0)):
            g = s[(s.idx & 1) == ct].sort_values("net_cost")
            if len(g):
                b = g.iloc[0]
                cost[tag][lab] = dict(net_cost=float(b.net_cost), lolh=float(b.lolh), co2_kt=float(b.co2_kt), passing=len(g),
                                      built=[OPTIONS[j][1] for j in range(len(OPTIONS)) if (int(b.idx) >> j) & 1])
    rng = np.random.default_rng(SEED)
    checks = []
    x = P["strongest plan without the turbine"]
    for r in CHECKS:
        row = dict(plan="strongest plan without the turbine", link_rate=r, **samples_needed(instance(M[r], x, r), rng))
        checks.append(row)
        print(f"link {r:.1%}: LOLH {row['lolh']:.2f}  precision {row['eps']:.1%}  MC {row['n_mc']:,.0f}  "
              f"IS {row['n_is']:,.0f}  QAE {row['n_qae']:,.0f}", flush=True)
    out = dict(rates=RATES, criterion=CRITERION, plans={k: [OPTIONS[j][1] for j in range(len(OPTIONS)) if v[j]] for k, v in P.items()},
               lolh=curves, break_even={k: crossing(RATES, v) for k, v in curves.items()}, cheapest=cost, checks=checks)
    OUT.write_text(json.dumps(out, indent=1))
    print("break-even link outage rate:", out["break_even"])


if __name__ == "__main__":
    run()
