"""Small tables for the notebook, read from the saved results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHORT = {"Avalon combustion turbine, 150 MW diesel": "turbine", "Bay d'Espoir Unit 8, 150 MW hydro": "Unit 8",
         "Wind, Avalon (Fermeuse area), 100 MW": "wind Avalon", "Wind, Burin (St. Lawrence area), 100 MW": "wind Burin",
         "Wind, west coast (Port au Port), 100 MW": "wind west coast", "Battery, 25 MW for 4 h": "battery 25 MW",
         "Battery, 50 MW for 4 h": "battery 50 MW", "15,000 baseboard homes to heat pumps": "15k baseboard homes",
         "35,000 baseboard homes to heat pumps": "35k baseboard homes", "10,000 oil homes to heat pumps": "10k oil homes",
         "30,000 oil homes to heat pumps": "30k oil homes", "Smart-thermostat demand response, 40 MW": "smart thermostats"}


def load(name: str) -> dict:
    return json.loads((ROOT / "results" / name).read_text())


def ai() -> pd.DataFrame:
    e1 = load("e1_demand_model.json")
    rows = {}
    for name, short in zip(e1["test"], ["linear degree-hour (utility standard)", "boosted trees", "boosted trees + cold tail (ours)"]):
        rows[short] = {
            "held-out winter, MAPE %": e1["test"][name]["mape"],
            "daily peak, MAPE %": e1["test"][name]["daily_peak_mape"],
            "top 1% hours, MAE MW": e1["test"][name]["mae_top1pct"],
            "colder than training, MAE MW": e1["extrapolation_below_minus8C"][name]["mae"],
            "4 Feb 2023 record (actual 1,980 MW), predicted MW": e1["stress_holdout_2022_23"][name]["record_pred"],
        }
    return pd.DataFrame(rows).T.round(1)


def validation() -> pd.DataFrame:
    v = load("e2_reliability.json")["validation_2026"]
    return pd.DataFrame({
        "ours (h/yr)": [v["ours_h"]["holyrood_20pct"], v["ours_h"]["holyrood_34pct"]],
        "NL Hydro (h/yr)": [v["hydro_published_h"]["holyrood_20pct"], v["hydro_published_h"]["holyrood_34pct"]],
    }, index=["2026-27 grid, Holyrood at its planning outage rate (20%)", "same, Holyrood as bad as winter 2021-22 (34%)"]).round(2)


def heat_pumps() -> pd.DataFrame:
    e2 = load("e2_reliability.json")
    hp = e2["heat_pumps_2026"]["scenarios"]
    t = pd.DataFrame(hp).T.rename(columns={"delta_median_winter_peak_mw": "change at a typical winter peak (MW)",
                                          "delta_at_coldest_hour_mw": "change at the coldest hour (MW)",
                                          "lolh": "blackout hours / yr"})
    today = e2["validation_2026"]["ours_h"]["holyrood_20pct"]
    t.loc["today, no change"] = [0.0, 0.0, today]
    return t.iloc[[-1, 0, 1, 2]].round(2)


def turbine() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = load("turbine_decision.json")
    reach = pd.DataFrame({
        "built": [", ".join(SHORT[o] for o in v) for v in d["plans"].values()],
        "blackout h/yr, link 1.1%": [v[0] for v in d["lolh"].values()],
        "blackout h/yr, link 5%": [v[-1] for v in d["lolh"].values()],
        "safe up to a link outage rate of": [f"{x:.1%}" if x and x != float("inf") else "beyond 5%" for x in d["break_even"].values()],
    }, index=list(d["plans"]))
    label = {"1pct": "link 1.1% (measured)", "3pct": "link 3%", "5pct": "link 5% (Hydro planning)"}
    cost = {}
    for tag, v in d["cheapest"].items():
        w, wo = v["with turbine"], v["without"]
        cost[label[tag]] = {"cheapest passing plan with the turbine, CAD M/yr": f"{w['net_cost']:.1f}" if w else "none passes",
                            "cheapest passing plan without it, CAD M/yr": f"{wo['net_cost']:.1f}" if wo else "none passes",
                            "plans that pass without the turbine": int(wo["passing"]) if wo else 0}
    return reach.round(2), pd.DataFrame(cost).T


def core() -> pd.DataFrame:
    e3 = load("e3_quantum_core.json")
    best = min(e3["curve"], key=lambda r: r["qae_rel_rmse"])
    return pd.DataFrame({"": {
        "qubits": e3["n_qubits"],
        "two-qubit gates in A / in one boost": f"{e3['resources']['cx_A']:,} / {e3['resources']['cx_Q']:,}",
        "P(blackout hour), circuit": f"{e3['p_statevector']:.6e}",
        "P(blackout hour), classical enumeration": f"{e3['p_exact']:.6e}",
        "blackout h/yr, circuit vs full engine": f"{e3['lolh_instance']:.2f} vs {e3['lolh_full_engine']:.2f}",
        f"error at {best['queries']:,} samples, amplitude estimation vs Monte Carlo":
            f"{best['qae_rel_rmse']:.1%} vs {best['mc_rel_rmse_theory']:.1%}",
    }})


def samples() -> tuple[pd.DataFrame, pd.DataFrame]:
    e4 = load("e4_rarity_advantage.json")
    t4 = pd.DataFrame(e4["rows"]).sort_values("p", ascending=False)
    t4["label"] = t4.label.str.replace("LIL", "link").str.replace("U8", "Unit 8")
    t4 = t4.set_index("label")[["lolh", "n_mc", "n_is", "n_qae"]].rename_axis(None)
    t4.columns = ["blackout h/yr", "Monte Carlo", "importance sampling", "amplitude estimation"]
    c = pd.DataFrame(load("turbine_decision.json")["checks"])
    c.index = [f"link {r:.1%}" for r in c.link_rate]
    c = c[["lolh", "eps", "n_mc", "n_is", "n_qae"]]
    c.columns = ["blackout h/yr", "precision needed", "Monte Carlo", "importance sampling", "amplitude estimation"]
    for t in (t4, c):
        for col in ("Monte Carlo", "importance sampling", "amplitude estimation"):
            t[col] = t[col].map("{:,.0f}".format)
    c["precision needed"] = c["precision needed"].map("{:.1%}".format)
    return t4.round(2), c.round(2)


def hardware() -> pd.DataFrame:
    tw = load("e5_hardware_twin.json")
    t = pd.DataFrame(tw["comparison"]).set_index("queries")
    t = t[["mc_rel_rmse_same_queries", "twin_amortised_rel_rmse", "mc_rel_rmse_3x_queries", "twin_paid_rel_rmse"]]
    t.columns = ["Monte Carlo", "ibm_quebec, calibration shared", "Monte Carlo, 3x runs", "ibm_quebec, calibration paid (3x runs)"]
    t.index.name = "runs"
    return t.round(3)


def cold_snap() -> pd.DataFrame:
    e6 = load("e6_cold_snap.json")
    rows = {}
    for k, lab in (("lil_5pct", "link 5%"), ("lil_1pct", "link 1.1%")):
        v = e6["cases"][k]
        rows[lab] = {"exact": v["p_any_shortfall"], "circuit": v["p_statevector"], "Monte Carlo (400k)": v["p_mc_400k"],
                     "if days were independent": v["p_independent_days"], "qubits": v["qubits"]}
    t = pd.DataFrame(rows).T.round(4)
    t["qubits"] = t.qubits.astype(int)
    return t


def coldsnap_plans() -> pd.DataFrame:
    d = load("coldsnap_plans.json")
    rows = {}
    for c in d["cases"]:
        rows[f"{c['plan']}, link {c['link']}"] = {
            "exact risk": f"{c['p_exact']:.2%}",
            "quantum estimate": f"{c['qae_mean']:.2%}",
            "error, Monte Carlo": f"±{c['mc_sd'] / c['p_exact']:.0%}",
            "error, quantum": f"±{c['qae_sd'] / c['p_exact']:.0%}",
            "right calls, Monte Carlo": f"{c['mc_calls'][c['truth']]} of 200",
            "right calls, quantum": f"{c['qae_calls'][c['truth']]} of 200",
        }
    return pd.DataFrame(rows).T
