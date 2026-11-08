"""E6: surviving a 3-day cold snap (chronological extension, where exact classical methods stop scaling).

System: 2030 with Hydro's plan (Holyrood retired; new Avalon CT and Bay d'Espoir U8). Cold snaps: for each of the
46 replayed winters (AI demand at the 2030 level), the 3-day window with the highest daily peaks; snaps are grouped
into 4 severity classes. Shortfall on a day: peak demand + Hydro's 70 MW minimum regulating reserve exceeds
available supply (all units except the two big risks at their expected availability).
Big risks with persistence (illustrative parameters, labelled as assumptions in the notebook):
  * Labrador-Island Link outage: stationary rate = EqFOR (5% planning or 1.1% measured), P(still out next day) = 0.6,
    more likely in severe snaps (icing).
  * Bay d'Espoir U1-6 frazil shutdown: starts only in the two most severe classes (0.5% / 2% per day; about one event
    in 50 winters, cf. "first since 1967"), P(still out next day) = 0.75 (Jan 2026: about 4 days).
Checks: circuit (Aer statevector) vs exact enumeration vs Monte Carlo; persistence vs independent days; circuit
resources vs days and units; queries for a 10% estimate (MC vs amplitude estimation, ideal circuit statistics).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.e2_reliability import demand_hours  # noqa: E402
from experiments.e4_rarity_advantage import EPS, qae_curve, queries_for  # noqa: E402
from gridrisk import fleet  # noqa: E402
from gridrisk.coldsnap import ColdSnapInstance, circuit  # noqa: E402
from gridrisk.quantum import statevector_probability  # noqa: E402

DAYS = 3
RESERVE_MW = 70.0
Q_MW = 50.0
SEED = 66
LIL_STAY, FRAZIL_STAY = 0.6, 0.75
SEV_MULT = np.array([0.5, 0.8, 1.2, 1.5])
FRAZIL_DAILY = np.array([0.0, 0.0, 0.005, 0.02])


def cold_snaps(year=2030, days=DAYS):
    dh, _ = demand_hours(year)
    loc = dh.index.tz_convert("America/St_Johns")
    d = dh.assign(day=loc.tz_localize(None).normalize())
    daily = d.groupby(["winter", "day"]).agg(peak=("demand_mw", "max"), tmin=("T_island", "min")).reset_index()
    snaps = []
    for w, g in daily.groupby("winter"):
        g = g.sort_values("day")
        roll = g.peak.rolling(days).mean().values
        i = int(np.nanargmax(roll))
        win = g.iloc[i - days + 1: i + 1]
        snaps.append(dict(winter=int(w), start=str(win.day.iloc[0].date()), peaks=win.peak.tolist(), tmin=float(win.tmin.min())))
    return snaps


def build(eqfor: float, snaps, units=("Labrador-Island Link", "frazil"), portfolio=None, days=DAYS):
    p = portfolio or fleet.hydro_plan_2030()
    mean = np.array([np.mean(s["peaks"]) for s in snaps])
    order = np.argsort(mean)
    classes = np.array_split(order, 4)
    traj = np.array([np.mean([snaps[i]["peaks"] for i in c], axis=0) for c in classes])     # (4, days)
    p_sev = np.array([len(c) for c in classes], float) / len(snaps)
    lil = fleet.lil(fleet.LIL_MW, eqfor)
    firm = sum(u.mw * (1 - u.efor) for u in p.units if not u.name.startswith("Labrador"))
    big_caps, names, first, stay, new = [], [], [], [], []
    for name in units:
        if name == "frazil":
            big_caps.append(int(round(459.0 / Q_MW))); names.append("Bay d'Espoir U1-6 frazil")
            first.append(FRAZIL_DAILY); new.append(FRAZIL_DAILY); stay.append(FRAZIL_STAY)
        elif name.startswith("Labrador"):
            big_caps.append(int(round(lil.mw / Q_MW))); names.append("Labrador-Island Link")
            m = SEV_MULT / SEV_MULT.mean()
            first.append(np.clip(eqfor * m, 0, 1))
            new.append(np.clip(eqfor * (1 - LIL_STAY) / (1 - eqfor) * m, 0, 1)); stay.append(LIL_STAY)
        else:
            u = next(x for x in p.units if x.name == name)
            big_caps.append(max(1, int(round(u.mw / Q_MW)))); names.append(name)
            firm -= u.mw * (1 - u.efor)
            first.append(np.full(4, u.efor)); new.append(np.full(4, u.efor * 0.5 / (1 - u.efor))); stay.append(0.5)
    real_big_mw = Q_MW * sum(c for c, n in zip(big_caps, names) if "frazil" not in n)
    Y = np.ceil((traj + RESERVE_MW - firm - real_big_mw) / Q_MW)                     # (4, days) in quanta
    # severity enters linearly through two bits: Y(s) = y0 + w0 b0 + w1 b1 (least squares, weights >= 0)
    X = np.array([[1, s & 1, (s >> 1) & 1] for s in range(4)], float)
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    y0 = np.round(coef[0]).astype(int)
    w = np.clip(np.round(coef[1:].T), 0, None).astype(int)                         # (days, 2)
    inst = ColdSnapInstance(names=names, cap=big_caps, p_sev=p_sev, first=np.array(first), stay=np.array(stay),
                            new=np.array(new), y0=y0[:days], w_sev=w[:days], quantum_mw=Q_MW)
    inst.labels = dict(firm_expected_mw=firm, class_trajectories_mw=traj.round(0).tolist(), Y_quanta=Y.tolist(),
                       Y_linear=[inst.y(s).tolist() for s in range(4)])
    return inst


def run():
    snaps = cold_snaps()
    rng = np.random.default_rng(SEED)
    out = dict(snaps=snaps, days=DAYS, reserve_mw=RESERVE_MW, cases={})
    for tag, eq in [("lil_5pct", 0.05), ("lil_1pct", 0.011)]:
        inst = build(eq, snaps)
        A, flag = circuit(inst)
        p = inst.exact()
        p_iid = inst.exact(persistence=False)
        p_sv = statevector_probability(A, flag)
        p_mc = float(inst.sample(400_000, rng).mean())
        max_K = int(np.clip(np.ceil(np.log2(1.0 / (EPS * np.sqrt(p)))) + 2, 3, 13))
        n_qae = queries_for(qae_curve(p, rng, max_K), EPS)
        n_mc = (1 - p) / (p * EPS ** 2)
        out["cases"][tag] = dict(p_any_shortfall=p, p_statevector=p_sv, p_mc_400k=p_mc, p_independent_days=p_iid,
                                 qubits=A.num_qubits, cx=int(A.count_ops().get("cx", 0)), n_mc=n_mc, n_qae=n_qae,
                                 advantage=n_mc / n_qae, labels=inst.labels)
        print(f"LIL {eq:.1%}: P(at least one shortfall day in the winter's worst 3-day cold snap) exact {p:.5f}  "
              f"circuit {p_sv:.5f}  MC {p_mc:.5f}  | independent-day model {p_iid:.5f}  | {A.num_qubits} qubits, "
              f"{out['cases'][tag]['cx']} CX  | MC {n_mc:,.0f} vs QAE {n_qae:,.0f} queries ({n_mc / n_qae:.1f}x)")
    # resource scaling: circuit size vs days and number of persistent units (built and transpiled, not simulated)
    extra = ["Labrador-Island Link", "frazil", "Bay d'Espoir U7", "Bay d'Espoir U8 (new)", "Avalon CT (new, diesel)", "Holyrood CT"]
    scaling = []
    for k in (2, 4, 6):
        for d in (1, 2, 3, 5, 7):
            snaps_d = cold_snaps(days=d) if d != DAYS else snaps
            inst = build(0.05, snaps_d, units=tuple(extra[:k]), days=d)
            A, _ = circuit(inst)
            scaling.append(dict(units=k, days=d, qubits=A.num_qubits, cx=int(A.count_ops().get("cx", 0)),
                                classical_enumeration_states=float(4 * 2.0 ** (k * d)), transfer_matrix_states=float(4 * 2 ** k)))
            print(f"  units {k} days {d}: {A.num_qubits} qubits, {scaling[-1]['cx']:,} CX; exact enumeration "
                  f"{scaling[-1]['classical_enumeration_states']:.1e} trajectories", flush=True)
    out["scaling"] = scaling
    (ROOT / "results" / "e6_cold_snap.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    run()
