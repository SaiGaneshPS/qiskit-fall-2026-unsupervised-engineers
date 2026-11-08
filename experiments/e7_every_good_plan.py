"""E7: Every good plan. Both quantum tools on one question: what should NL build so Holyrood's oil can retire,
the lights stay on, CO2 falls and household bills drop?

Twelve yes/no options (gridrisk.portfolio), 4,096 plans, each scored in $ per year (capital, operations,
household bill savings, wind energy value, CO2, blackout cost from the validated reliability engine).
Many plans are nearly as good, so the decision output is each option's "must-have score": its share of the
good plans (Boltzmann weights at temperature T, in $10M per year), i.e. irreplaceability.

Stages (run in order):
  tables    score all 4,096 plans for three Labrador-Island Link outage rates (5% planning, 3%, 1.1% measured)
  analysis  best plans, must-have scores, and exact convergence rates (absolute spectral gaps) of the sampler
            for 12, 10 and 8 open options: one change at a time / random plan / quantum proposal (learned map) /
            quantum on a line of qubits / scrambled-map control
  chains    12 qubits on Qiskit Aer: 2,000-step chains estimate the must-have scores, against exact values
  verify    quantum check of the chosen plan's blackout risk (amplitude estimation vs Monte Carlo queries)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e2_reliability import WIND_COLD_CUTOUT_C, demand_hours, forecast_errors, frazil_p  # noqa: E402
from gridrisk import data, fleet, ml, scenarios  # noqa: E402
from gridrisk.portfolio import OPTIONS, PlanModel, PlanProblem  # noqa: E402
from gridrisk.reliability import wind_power_curve  # noqa: E402

SEED = 1225
LIL = {"5pct": 0.05, "3pct": 0.03, "1pct": 0.011}
TEMPS = [0.1, 0.25, 0.5]                     # $10M per year: plans within about $1M, $2.5M, $5M of the best
IDX = {k: i for i, (k, *_) in enumerate(OPTIONS)}
SITUATIONS = {
    12: {"all options open": {}},
    10: {"Unit 8 delayed, demand response in place": {IDX["u8"]: 0, IDX["dr"]: 1},
         "Avalon CT approved, small battery not funded": {IDX["ct"]: 1, IDX["bat_small"]: 0}},
    8: {"CT approved, Unit 8 delayed, demand response in place, no small battery":
        {IDX["ct"]: 1, IDX["u8"]: 0, IDX["dr"]: 1, IDX["bat_small"]: 0},
        "Unit 8 built, no CT, no west-coast wind, small oil program running":
        {IDX["ct"]: 0, IDX["u8"]: 1, IDX["wind_west"]: 0, IDX["oil_small"]: 1}},
}
TABLE = ROOT / "results" / "e7_plan_table.csv"


def site_cf(site: str) -> pd.Series:
    w = data.load_weather(site)
    cf = wind_power_curve(w.wind_speed_100m.values)
    cf[w.temperature_2m.values < WIND_COLD_CUTOUT_C] = 0.0
    return pd.Series(cf, index=w.index)


def make_models() -> dict:
    vals, w, _ = forecast_errors()
    dh, _ = demand_hours(2030)
    ds = ml.dataset()
    lin = ml.LinearDegreeHour().fit(ds, ds.demand_mw)
    ua = scenarios.ua_per_home_kw(float(lin.m.coef_[0] + lin.m.coef_[1]))
    cfs = {s: site_cf(s) for s in ("fermeuse", "stlawrence", "portauport")}
    return {tag: PlanModel(dh, (vals, w), cfs, ua, eq) for tag, eq in LIL.items()}


def all_bits() -> np.ndarray:
    k = np.arange(2 ** len(OPTIONS), dtype=np.int64)
    return ((k[:, None] >> np.arange(len(OPTIONS))) & 1).astype(np.int8)


def stage_tables():
    models = make_models()
    X = all_bits()
    rows = []
    t0 = time.time()
    for tag, m in models.items():
        for i, x in enumerate(X):
            b = m.breakdown(x)
            rows.append(dict(scenario=tag, idx=i, **{k: v for k, v in b.items() if k != "built"}))
            if i % 512 == 511:
                print(f"  {tag}: {i + 1}/4096 plans  ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TABLE, index=False)
    m = next(iter(models.values()))
    (ROOT / "results" / "e7_per_home.json").write_text(json.dumps(dict(per_home=m.per_home, wind_mwh=m.wind_mwh), indent=1))
    print("tables written", len(df))


def load_table() -> pd.DataFrame:
    return pd.read_csv(TABLE)


def stage_analysis():
    from gridrisk.qmcmc.mcmc import (boltzmann, local_proposal, metropolis_gap, quantum_proposal_exact,
                                     uniform_proposal)
    from gridrisk.qmcmc.warmup import hardware_native_surrogate, learn_surrogate
    tab = load_table()
    X = all_bits()
    rng0 = np.random.default_rng(SEED)
    g = 0.25 + (np.arange(6) + 0.5) * (0.35 / 6)
    gammas, times = np.repeat(g, 4), rng0.uniform(2.0, 20.0, 24)    # same randomised (gamma, t) set as Project 1
    out = dict(options=[o[1] for o in OPTIONS], scenarios={}, gaps=[])
    for tag in LIL:
        t = tab[tab.scenario == tag].sort_values("idx")
        cost = t.net_cost.values
        E = cost / 10.0
        o = np.argsort(cost)
        nb = np.array([E[np.arange(len(E)) ^ (1 << i)] for i in range(len(OPTIONS))])
        locmin = (E[None, :] < nb).all(0)
        out["scenarios"][tag] = dict(
            best=[dict(cost=float(cost[i]), lolh=float(t.lolh.values[i]), co2_kt=float(t.co2_kt.values[i]),
                       household_savings=float(t.household_savings.values[i]),
                       options=[OPTIONS[j][1] for j in range(len(OPTIONS)) if X[i, j]]) for i in o[:6]],
            meets_criterion=int((t.lolh.values <= fleet.LOLH_CRITERION_H).sum()),
            within_5M=int((cost <= cost.min() + 5).sum()), local_minima=int(locmin.sum()),
            local_minima_within_10M=int((locmin & (cost <= cost.min() + 10)).sum()),
            must_have={str(T): (boltzmann(E, T) @ X).round(4).tolist() for T in TEMPS})
        print(f"{tag}: best ${cost.min():.1f}M/yr  LOLH {t.lolh.values[o[0]]:.2f}  plans meeting 2.8 h: "
              f"{out['scenarios'][tag]['meets_criterion']}  within $5M: {out['scenarios'][tag]['within_5M']}  "
              f"local minima {locmin.sum()}", flush=True)
        for size, sits in SITUATIONS.items():
            for sit, fixed in sits.items():
                t0 = time.time()
                p = PlanProblem(cost, fixed)
                Ev = p.energy_vector()
                h, J, _, _ = learn_surrogate(p, np.random.default_rng(SEED + size))
                hl, Jl, _, _ = hardware_native_surrogate(p, np.random.default_rng(SEED + 1 + size))
                perm = np.random.default_rng(SEED + 2).permutation(p.n)
                hp, Jp = h[perm], np.triu((J + J.T)[np.ix_(perm, perm)], 1)
                props = {"local": local_proposal(p.n), "uniform": uniform_proposal(p.n),
                         "quantum": quantum_proposal_exact(h, J, gammas, times),
                         "q_line": quantum_proposal_exact(hl, Jl, gammas, times),
                         "q_scrambled": quantum_proposal_exact(hp, Jp, gammas, times)}
                for T in TEMPS:
                    for name, Q in props.items():
                        out["gaps"].append(dict(scenario=tag, n=p.n, situation=sit, T=T, proposal=name,
                                                gap=float(metropolis_gap(Q, Ev, T))))
                print(f"   n={p.n:2d} {sit[:60]:60s} {time.time() - t0:5.0f}s", flush=True)
                (ROOT / "results" / "e7_plans_summary.json").write_text(json.dumps(out, indent=1))
    df = pd.DataFrame(out["gaps"])
    piv = df.pivot_table(index=["scenario", "n", "situation", "T"], columns="proposal", values="gap").reset_index()
    piv["best_classical"] = piv[["local", "uniform"]].max(axis=1)
    piv["speedup"] = piv.quantum / piv.best_classical
    out["gap_summary"] = dict(cases=int(len(piv)), quantum_wins=int((piv.quantum > piv.best_classical).sum()),
                              line_wins=int((piv.q_line > piv.best_classical).sum()),
                              scrambled_wins=int((piv.q_scrambled > piv.best_classical).sum()),
                              median_speedup=float(piv.speedup.median()), max_speedup=float(piv.speedup.max()),
                              median_speedup_by_T={str(T): float(g.speedup.median()) for T, g in piv.groupby("T")},
                              median_steps_by_T={str(T): {k: float(np.median(1 / g[k])) for k in ("local", "uniform", "quantum", "q_line")}
                                                 for T, g in piv.groupby("T")})
    (ROOT / "results" / "e7_plans_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["gap_summary"], indent=1))


def stage_chains(tag="3pct", T=0.25, steps=2000, M=12, n_replicas=8):
    from gridrisk.qmcmc.chains import (QuantumProposer, local_proposer, run_chains, run_parallel_tempering,
                                       running_marginal_error, uniform_proposer)
    from gridrisk.qmcmc.mcmc import boltzmann
    from gridrisk.qmcmc.warmup import learn_surrogate
    tab = load_table()
    cost = tab[tab.scenario == tag].sort_values("idx").net_cost.values
    p = PlanProblem(cost)
    E = p.energy_vector(); X = p.all_configs()
    exact = boltzmann(E, T) @ X
    rng = np.random.default_rng(SEED)
    h, J, _, _ = learn_surrogate(p, rng)
    g = 0.25 + (np.arange(6) + 0.5) * (0.35 / 6)
    gammas, times = np.repeat(g, 4), np.random.default_rng(SEED).uniform(2.0, 20.0, 24)
    X0 = rng.integers(0, 2, (M, p.n)).astype(np.int8)
    out = dict(scenario=tag, T=T, steps=steps, M=M, exact=exact.tolist(), warmup_evals=30 * p.n ** 2)
    trajs = {}
    for name, prop in [("local", local_proposer), ("uniform", uniform_proposer)]:
        trajs[name], st = run_chains(p, prop, T, steps, M, np.random.default_rng(SEED + 1), X0)
    qp = QuantumProposer(h, J, gammas, times, seed=SEED)
    tq = time.time()
    trajs["quantum"], st = run_chains(p, qp, T, steps, M, np.random.default_rng(SEED + 2), X0, log_every=250)
    out["quantum_circuits"] = qp.circuits_run
    out["quantum_sim_seconds"] = time.time() - tq
    pt = run_parallel_tempering(p, T, steps // n_replicas, M, np.random.default_rng(SEED + 3), n_replicas)
    trajs["parallel_tempering"] = np.repeat(pt, n_replicas, axis=0)
    cp = np.unique(np.geomspace(n_replicas, steps, 25).astype(int))
    out["checkpoints"] = cp.tolist()
    for name, tr in trajs.items():
        err = running_marginal_error(tr, exact, cp)
        out[f"{name}_err_median"] = np.median(err, axis=1).tolist()
    (ROOT / "results" / f"e7_plans_chains_{tag}.json").write_text(json.dumps(out, indent=1))
    for name in trajs:
        print(f"  {name:20s} final median max-error {out[name + '_err_median'][-1]:.3f}")


def stage_verify():
    """Blackout-risk check of each scenario's best plan with the quantum LOLP instance (E3/E4 machinery)."""
    from experiments.e4_rarity_advantage import EPS, best_is, qae_curve, queries_for
    from gridrisk.instances import build_instance, lolh_from_instance
    from gridrisk.reliability import Portfolio, Unit
    from gridrisk.portfolio import battery_firm_mw, WIND_SITE
    summ = json.loads((ROOT / "results" / "e7_plans_summary.json").read_text())
    models = make_models()
    X = all_bits()
    tab = load_table()
    rng = np.random.default_rng(SEED)
    res = {}
    for tag, m in models.items():
        cost = tab[tab.scenario == tag].sort_values("idx").net_cost.values
        x = X[int(np.argmin(cost))]
        c = m.choices(x)
        base = fleet.no_holyrood_2030()
        units = [fleet.lil(fleet.LIL_MW, LIL[tag]) if u.name.startswith("Labrador") else u for u in base.units]
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
        p = Portfolio("best plan", units, wind_mw=1.0)
        big = [b for b in ("Labrador-Island Link", "Bay d'Espoir U7", "Hardwoods GT", "Bay d'Espoir U8 (new)",
                           "Avalon CT (new, diesel)") if any(u.name == b for u in units)]
        inst = build_instance(p, m.demand(c["n_bb"], c["n_oil"]), m.err, cf, big, frazil=(459.0, frazil_p()))
        pe = inst.exact()
        max_K = int(np.clip(np.ceil(np.log2(1.0 / (EPS * np.sqrt(pe)))) + 2, 3, 13))
        n_qae = queries_for(qae_curve(pe, rng, max_K), EPS)
        n_mc = (1 - pe) / (pe * EPS ** 2)
        tilt, lam, rv = best_is(inst, rng, pe)
        res[tag] = dict(options=[OPTIONS[j][1] for j in range(len(OPTIONS)) if x[j]], p=pe, lolh_instance=lolh_from_instance(inst),
                        lolh_engine=float(tab[(tab.scenario == tag)].sort_values("idx").lolh.values[int(np.argmin(cost))]),
                        n_mc=n_mc, n_qae=n_qae, n_is=rv / EPS ** 2, qubits=None)
        from gridrisk.quantum import state_preparation
        A, _ = state_preparation(inst)
        res[tag]["qubits"] = A.num_qubits
        print(f"{tag}: best plan blackout risk p={pe:.2e} (LOLH {res[tag]['lolh_instance']:.2f} vs engine {res[tag]['lolh_engine']:.2f})  "
              f"MC {n_mc:,.0f}  IS {rv / EPS ** 2:,.0f}  QAE {n_qae:,.0f}  ({A.num_qubits} qubits)", flush=True)
    summ["verify"] = res
    (ROOT / "results" / "e7_plans_summary.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "tables"
    {"tables": stage_tables, "analysis": stage_analysis, "chains": stage_chains, "verify": stage_verify}[stage]()
