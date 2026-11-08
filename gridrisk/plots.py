"""Figures for the notebook and slides."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
C_Q, C_MC, C_IS, C_ML, C_BASE, C_RED, C_WIND = "#0f6e7a", "#8c8c8c", "#c9a227", "#e07b39", "#6b6b9e", "#c0392b", "#3a8fd6"


def _load(name):
    return json.loads((ROOT / "results" / name).read_text())


# ------------------------------------------------------------------------------------------------ AI layer
def demand_model(axs=None):
    e1 = _load("e1_demand_model.json")
    pr = pd.read_csv(ROOT / "results" / "e1_test_predictions.csv", parse_dates=["time_utc"]).set_index("time_utc")
    if axs is None:
        _, axs = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw=dict(width_ratios=[2.2, 1, 1]))
    win = pr.loc["2026-01-15":"2026-02-05"]
    loc = win.index.tz_convert("America/St_Johns")
    axs[0].plot(loc, win.actual, color="k", lw=1.2, label="actual island demand")
    axs[0].plot(loc, win.ours, color=C_ML, lw=1.0, label="AI model (never saw this winter)")
    axs[0].plot(loc, win.linear, color=C_BASE, lw=0.8, ls="--", label="linear degree-hour baseline")
    axs[0].axvspan(pd.Timestamp("2026-01-22", tz="America/St_Johns"), pd.Timestamp("2026-01-26", tz="America/St_Johns"),
                   color=C_RED, alpha=0.08, label="Power Watch / Warning (Bay d'Espoir frazil ice)")
    axs[0].set_ylabel("MW"); axs[0].legend(fontsize=7, loc="upper left"); axs[0].tick_params(axis="x", labelsize=7)
    t = e1["test"]
    names = list(t.keys()); short = ["linear\ndegree-hour", "boosting,\nno tail", "boosting +\nphysics tail (ours)"]
    axs[0].set_title(f"Held-out winter 2025-26: MAPE ours {t[names[2]]['mape']:.1f}% vs linear {t[names[0]]['mape']:.1f}%", fontsize=9)
    ex = e1["extrapolation_below_minus8C"]
    axs[1].bar(short, [ex[n]["mae"] for n in names], color=[C_BASE, C_MC, C_ML])
    for i, n in enumerate(names):
        axs[1].text(i, ex[n]["mae"] + 1, f"bias {ex[n]['bias']:+.0f}", ha="center", fontsize=7)
    axs[1].set_ylabel("MAE (MW)"); axs[1].tick_params(axis="x", labelsize=7)
    axs[1].set_title(f"Colder than ever seen in training\n({ex[names[0]]['n_cold_hours']} hours below -8 C)", fontsize=9)
    st = e1["stress_holdout_2022_23"]
    axs[2].bar(short, [st[n]["record_pred"] for n in names], color=[C_BASE, C_MC, C_ML])
    axs[2].axhline(st[names[0]]["record_actual"], color="k", ls="--", lw=1)
    axs[2].text(1, st[names[0]]["record_actual"] + 15, "actual flash-freeze record, 4 Feb 2023", ha="center", fontsize=7)
    axs[2].set_ylim(1300, 2100); axs[2].set_ylabel("MW"); axs[2].tick_params(axis="x", labelsize=7)
    axs[2].set_title("Negative result: the flash-freeze hour\nis ~320 MW above any weather model", fontsize=9)
    return axs


def weather_years(ax=None):
    e2 = _load("e2_reliability.json")
    ax = ax or plt.gca()
    from experiments.e2_reliability import demand_hours
    dh, _ = demand_hours(2026)
    pk = dh.groupby("winter").demand_mw.max()
    ax.bar(pk.index, pk.values, color=C_ML, alpha=0.8, label="peak demand, today's load under that winter's weather")
    ax.axhline(1788, color="k", ls=":", lw=1); ax.text(1980, 1795, "NL Hydro 2026 peak forecast", fontsize=7)
    ax.set_ylim(1500, 1950); ax.set_ylabel("MW"); ax.set_xlabel("winter (start year)")
    ax.set_title("46 winters of weather replayed on today's grid (AI demand model)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    return ax


def wind_at_peak(ax=None):
    e2 = _load("e2_reliability.json")
    ax = ax or plt.gca()
    w = e2["wind_at_peak"]
    labels = ["all hours", "top 10%\ndemand hours", "top 1%", "top 0.1%"]
    vals = [w["top_1.0"], w["top_0.1"], w["top_0.01"], w["top_0.001"]]
    ax.bar(labels, vals, color=C_WIND)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{100 * v:.0f}%", ha="center", fontsize=8)
    ax.set_ylabel("wind capacity factor"); ax.set_ylim(0, 0.75)
    ax.set_title("Fermeuse wind: it blows harder when St. John's is coldest", fontsize=9)
    return ax


def heat_pumps(axs=None):
    from gridrisk import scenarios
    e2 = _load("e2_reliability.json")
    hp = e2["heat_pumps_2026"]
    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    T = np.linspace(-24, 15, 200)
    ua = hp["ua_kw_per_degree_per_home"]
    axs[0].plot(T, scenarios.demand_delta_mw(T, ua, oil_to_hp=40_000), color=C_RED, label="40k oil homes -> heat pump")
    axs[0].plot(T, scenarios.demand_delta_mw(T, ua, oil_to_hp=40_000, oil_backup_below_c=-10), color=C_RED, ls="--",
                label="... keeping the oil furnace below -10 C")
    axs[0].plot(T, scenarios.demand_delta_mw(T, ua, baseboard_to_hp=50_000), color=C_Q, label="50k baseboard homes -> heat pump")
    axs[0].axhline(0, color="k", lw=0.6); axs[0].set_xlabel("outdoor temperature (C)"); axs[0].set_ylabel("change in island demand (MW)")
    axs[0].legend(fontsize=7); axs[0].set_title("Heat pumps change the grid's worst hours", fontsize=9)
    sc = hp["scenarios"]
    labels = ["today", "+40k oil\n-> HP", "+40k oil -> HP\n(oil backup)", "+50k baseboard\n-> HP"]
    vals = [e2["validation_2026"]["ours_h"]["holyrood_20pct"]] + [sc[k]["lolh"] for k in sc]
    axs[1].bar(labels, vals, color=[C_MC, C_RED, "#e8a19a", C_Q])
    axs[1].axhline(2.8, color="k", ls="--", lw=1); axs[1].text(-0.4, 2.85, "criterion 2.8 h/yr", fontsize=7)
    for i, v in enumerate(vals):
        axs[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    axs[1].set_ylabel("expected blackout hours / year"); axs[1].tick_params(axis="x", labelsize=7)
    axs[1].set_title("2026-27 grid: loss-of-load hours", fontsize=9)
    return axs


def decision_2030(ax=None):
    e2 = _load("e2_reliability.json")
    ax = ax or plt.gca()
    pf = e2["decision_2030"]["portfolios"]
    labels = list(pf.keys())
    y = np.arange(len(labels))
    a = [pf[k]["lil_5pct"]["lolh"] for k in labels]
    b = [pf[k]["lil_1pct"]["lolh"] for k in labels]
    ax.barh(y + 0.2, a, height=0.38, color=C_MC, label="Labrador-Island Link outage rate 5% (Hydro planning)")
    ax.barh(y - 0.2, b, height=0.38, color=C_Q, label="1.1% (measured, 12 months to Sep 2026)")
    ax.axvline(2.8, color="k", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("expected blackout hours / year (log)")
    for i, k in enumerate(labels):
        ax.text(max(a[i], b[i]) * 1.15, i, f"CO2 -{pf[k]['co2_avoided_kt_per_yr']:.0f} kt/yr", va="center", fontsize=7)
    ax.set_xlim(0.3, 600)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ax.set_title("2030: retiring Holyrood's oil. Criterion: 2.8 h/yr (dashed)", fontsize=9)
    return ax


# ------------------------------------------------------------------------------------------------ quantum
def core_check(axs=None):
    e3 = _load("e3_quantum_core.json")
    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    pw = np.array(e3["powers"])
    th = np.arcsin(np.sqrt(e3["p_exact"]))
    mm = np.linspace(0, pw.max(), 300)
    axs[0].plot(mm, np.sin((2 * mm + 1) * th) ** 2, color=C_Q, lw=1, label="theory sin^2((2m+1) theta)")
    axs[0].plot(pw, e3["measured_p_flag"], "o", color=C_ML, label=f"Aer, real {e3['n_qubits']}-qubit circuits")
    axs[0].set_xlabel("Grover power m"); axs[0].set_ylabel("P(shortfall flag)")
    axs[0].set_title("Amplification of a 1-in-1,700 risk on the full circuit", fontsize=9); axs[0].legend(fontsize=7)
    c = e3["curve"]
    q = [r["queries"] for r in c]
    axs[1].loglog(q, [r["qae_rel_rmse"] for r in c], "o-", color=C_Q, label="amplitude estimation (Aer samples)")
    axs[1].loglog(q, [r["mc_rel_rmse_theory"] for r in c], "--", color=C_MC, label="Monte Carlo, same queries")
    axs[1].set_xlabel("oracle queries"); axs[1].set_ylabel("relative RMSE")
    axs[1].set_title("End-to-end on the real instance (300 trials)", fontsize=9); axs[1].legend(fontsize=7)
    return axs


def rarity_advantage(axs=None):
    e4 = _load("e4_rarity_advantage.json")
    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    rows = sorted(e4["rows"], key=lambda r: -r["p"])
    p = np.array([r["p"] for r in rows])
    axs[0].loglog(p, [r["n_mc"] for r in rows], "o-", color=C_MC, label="Monte Carlo")
    axs[0].loglog(p, [r["n_is"] for r in rows], "s-", color=C_IS, label="tuned importance sampling (classical)")
    axs[0].loglog(p, [r["n_qae"] for r in rows], "D-", color=C_Q, label="quantum amplitude estimation")
    axs[0].invert_xaxis(); axs[0].set_xlabel("P(shortfall in a winter hour)  (rarer ->)")
    axs[0].set_ylabel("samples / queries for 10% error"); axs[0].legend(fontsize=7)
    axs[0].set_title("The safer the grid, the more samples it takes to prove it", fontsize=9)
    axs[1].semilogx(p, [r["advantage_vs_mc"] for r in rows], "D-", color=C_Q, label="vs Monte Carlo")
    axs[1].semilogx(p, [r["advantage_vs_is"] for r in rows], "s-", color=C_IS, label="vs tuned importance sampling")
    axs[1].axhline(1, color="k", lw=0.6); axs[1].invert_xaxis()
    for r in rows:
        axs[1].annotate(f"{r['lolh']:.1f} h", (r["p"], r["advantage_vs_mc"]), fontsize=6, xytext=(2, 4), textcoords="offset points")
    axs[1].set_yscale("log"); axs[1].set_xlabel("P(shortfall in a winter hour)"); axs[1].set_ylabel("quantum query advantage (x)")
    axs[1].set_title(f"Advantage vs Monte Carlo ~ p^{e4['fitted_log_slope_advantage_vs_p']:.2f} (theory p^-0.5)", fontsize=9)
    axs[1].legend(fontsize=7)
    return axs


def hardware_scale(axs=None):
    e5 = _load("e5_hardware_scale_noise.json")
    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    pw = e5["powers"]
    axs[0].plot(pw, e5["ideal"], "o-", color=C_Q, label="ideal")
    axs[0].plot(pw, e5["fake_fez"]["p_with_readout"], "s--", color=C_RED, label="IBM Heron noise model (FakeFez)")
    for m, g in zip(pw, e5["fake_fez"]["two_qubit_gates"]):
        axs[0].annotate(f"{g} CZ", (m, e5["ideal"][m]), fontsize=7, xytext=(-8, 6), textcoords="offset points")
    axs[0].set_xlabel("Grover power m"); axs[0].set_ylabel("P(shortfall flag)"); axs[0].legend(fontsize=7)
    axs[0].set_title("3-qubit exact reduction: 2030, Holyrood retired, nothing added", fontsize=9)
    c = e5["fake_fez"]["comparison"]
    q = [r["queries"] for r in c]
    axs[1].plot(q, [r["mc_rel_rmse"] for r in c], "o-", color=C_MC, label="Monte Carlo")
    axs[1].plot(q, [r["plain_mlae_rel_rmse"] for r in c], "s-", color=C_IS, label="plain MLAE (ignores noise)")
    axs[1].plot(q, [r["noise_aware_mlae_rel_rmse"] for r in c], "D-", color=C_Q, label="noise-aware MLAE")
    axs[1].set_ylim(0, 0.25); axs[1].set_xlabel("oracle queries"); axs[1].set_ylabel("relative RMSE"); axs[1].legend(fontsize=7)
    axs[1].set_title("Under realistic noise, at equal budget", fontsize=9)
    return axs


def cold_snap(axs=None):
    e6 = _load("e6_cold_snap.json")
    if axs is None:
        _, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    lab = e6["cases"]["lil_5pct"]["labels"]
    traj = np.array(lab["class_trajectories_mw"])
    days = np.arange(1, traj.shape[1] + 1)
    for i, tr in enumerate(traj):
        axs[0].plot(days, tr + e6["reserve_mw"], "o-", color=plt.get_cmap("Blues")(0.35 + 0.18 * i), label=f"severity class {i}")
    firm = lab["firm_expected_mw"]
    for y, txt, col in [(firm + 700, "supply, all up", C_Q), (firm + 700 - 459, "Bay d'Espoir frazil shutdown", C_ML),
                        (firm, "Labrador-Island Link out", C_RED)]:
        axs[0].axhline(y, color=col, ls="--", lw=1)
        axs[0].text(days[-1], y - 22, txt, fontsize=7, color=col, ha="right", va="top")
    axs[0].set_xticks(days); axs[0].set_xlabel("day of the cold snap"); axs[0].set_ylabel("daily peak + 70 MW reserve (MW)")
    axs[0].set_title("Worst 3-day cold snap of each of 46 winters (2030 load)", fontsize=9); axs[0].legend(fontsize=6, loc="center right")
    cases = e6["cases"]
    x = np.arange(2)
    axs[1].bar(x - 0.2, [cases[k]["p_any_shortfall"] for k in ("lil_5pct", "lil_1pct")], 0.38, color=C_Q, label="outages persist (chronological)")
    axs[1].bar(x + 0.2, [cases[k]["p_independent_days"] for k in ("lil_5pct", "lil_1pct")], 0.38, color=C_MC, label="days treated as independent")
    axs[1].set_xticks(x); axs[1].set_xticklabels(["LIL outage rate 5%", "LIL 1.1%"])
    axs[1].set_ylabel("P(at least one shortfall day)"); axs[1].legend(fontsize=7)
    axs[1].set_title("Chronology changes the answer", fontsize=9)
    sc = pd.DataFrame(e6["scaling"])
    for k, g in sc.groupby("units"):
        axs[2].semilogy(g.days, g.cx, "D-", color=plt.get_cmap("Greens")(0.4 + 0.1 * k), label=f"circuit CX, {k} persistent units")
        axs[2].semilogy(g.days, g.classical_enumeration_states, ":", color=plt.get_cmap("Greys")(0.4 + 0.1 * k),
                        label=f"exact enumeration, {k} units")
    axs[2].set_xlabel("days in the cold snap"); axs[2].set_ylabel("size (log)")
    axs[2].set_title("Circuit grows linearly; exact enumeration exponentially", fontsize=9); axs[2].legend(fontsize=6)
    return axs


def hardware_real(axs=None):
    """Real IBM hardware (ibm_quebec): signal per Grover power and error vs queries, with the twin-calibrated run."""
    h = _load("e5_hardware.json")["hardware"]
    ideal = _load("e5_hardware.json")["ideal"]
    tf = ROOT / "results" / "e5_hardware_twin.json"
    tw = json.loads(tf.read_text()) if tf.exists() else None
    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    m = np.arange(len(ideal))
    axs[0].plot(m, ideal, "o-", color=C_Q, label="ideal (perfect quantum computer)")
    axs[0].plot(m, h["predicted_from_device_noise_model"], "s--", color=C_MC, label="predicted by ibm_quebec's noise model")
    axs[0].plot(m, h["measured"], "D-", color=C_RED, label=f"measured on {h['backend']} (run 1)")
    if tw:
        axs[0].plot(m, tw["measured"]["target"], "D:", color=C_RED, alpha=0.6, label="measured (run 2, twin job)")
        axs[0].plot(m, tw["corrected_target"], "^-", color=C_IS, label="run 2 after twin correction")
    axs[0].set_xlabel("boosting rounds (Grover power m)"); axs[0].set_ylabel("P(shortfall flag)")
    axs[0].set_title("HARDWARE: the boosting works on a real chip", fontsize=9); axs[0].legend(fontsize=7)
    c = h["comparison"]
    q = np.array([r["queries"] for r in c])
    axs[1].plot(q[1:], [r["mc_rel_rmse_theory"] for r in c][1:], "o-", color=C_MC, label="Monte Carlo, same runs")
    axs[1].plot(q[1:], [r["noise_aware_mlae_rel_rmse"] for r in c][1:], "D-", color=C_RED, label="hardware, noise-aware fit (run 1)")
    if tw:
        tc = tw["comparison"]
        axs[1].plot(q[1:], [r["twin_amortised_rel_rmse"] for r in tc][1:], "^-", color=C_IS, label="hardware, twin-calibrated (shared calibration)")
        axs[1].plot([r["queries_with_twins"] for r in tc][1:], [r["twin_paid_rel_rmse"] for r in tc][1:], "v--", color=C_IS, alpha=0.7,
                    label="hardware, twin-calibrated (calibration paid)")
        axs[1].plot([r["queries_with_twins"] for r in tc][1:], [r["mc_rel_rmse_3x_queries"] for r in tc][1:], "o:", color=C_MC, alpha=0.7)
    axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].set_xlabel("runs (oracle queries)"); axs[1].set_ylabel("relative error")
    axs[1].set_title("HARDWARE: error over 10 repeats vs Monte Carlo", fontsize=9); axs[1].legend(fontsize=6.5)
    return axs


def plans_must_have(ax=None, T="0.25"):
    s = _load("e7_plans_summary.json")
    ax = ax or plt.gca()
    opts = s["options"]
    sc = {k: np.array(v["must_have"][T]) for k, v in s["scenarios"].items()}
    order = np.argsort(-sc["1pct"])
    y = np.arange(len(opts))
    cols = [("5pct", C_MC, "link fails 5% (Hydro planning)"), ("3pct", C_IS, "3%"), ("1pct", C_Q, "1.1% (measured)")]
    h = 0.26
    for j, (k, c, lab) in enumerate(cols):
        ax.barh(y + (j - 1) * h, sc[k][order], height=h, color=c, label=lab)
    ax.set_yticks(y); ax.set_yticklabels([opts[i] for i in order], fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0, 1.05); ax.set_xlabel("must-have score: share of good plans that include it")
    ax.axvline(0.5, color="k", lw=0.6, ls=":")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"2030: which investments appear in the good plans (within about ${float(T) * 10:.1f}M/yr of the best)", fontsize=9)
    return ax


def plans_sampler(axs=None):
    s = _load("e7_plans_summary.json")
    g = s["gap_summary"]["median_steps_by_T"]
    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    Ts = sorted(g, key=float)
    x = np.arange(len(Ts)); w = 0.2
    for j, (k, c, lab) in enumerate([("local", C_MC, "one change at a time"), ("uniform", C_IS, "random plan"),
                                     ("q_line", C_BASE, "quantum, line of qubits"), ("quantum", C_Q, "quantum, learned map")]):
        axs[0].bar(x + (j - 1.5) * w, [g[T][k] for T in Ts], width=w, color=c, label=lab)
    axs[0].set_yscale("log"); axs[0].set_xticks(x)
    axs[0].set_xticklabels([f"T = ${float(T) * 10:g}M/yr" if float(T) >= 0.25 else f"T = ${float(T) * 10:g}M/yr (picky)" for T in Ts], fontsize=8)
    axs[0].set_ylabel("steps to forget the starting plan (log)")
    gs = s["gap_summary"]
    axs[0].set_title(f"Exact convergence: quantum faster in {gs['quantum_wins']} of {gs['cases']} cases", fontsize=9)
    axs[0].legend(fontsize=7)
    f = ROOT / "results" / "e7_plans_chains_3pct.json"
    if f.exists():
        c = json.loads(f.read_text())
        cp = c["checkpoints"]
        for k, col, lab in [("local", C_MC, "one change at a time"), ("uniform", C_IS, "random plans"),
                            ("parallel_tempering", C_BASE, "parallel tempering"), ("quantum", C_Q, "quantum suggestions")]:
            axs[1].loglog(cp, c[k + "_err_median"], "o-", ms=3, color=col, label=lab)
        axs[1].set_xlabel("plans evaluated"); axs[1].set_ylabel("largest error in must-have scores")
        axs[1].set_title("12 qubits on Qiskit Aer (link 3%, median of 12 runs)", fontsize=9); axs[1].legend(fontsize=7)
    else:
        axs[1].axis("off")
    return axs


def turbine(ax=None):
    """Blackout hours vs Labrador-Island Link outage rate for four 2030 plans."""
    d = _load("turbine_decision.json")
    ax = ax or plt.gca()
    r = 100 * np.array(d["rates"])
    style = {"Hydro's plan: Unit 8 + turbine": (C_MC, "-"), "cheapest plan without the turbine": (C_ML, "-"),
             "strongest plan without the turbine": (C_Q, "-"), "same, plus the turbine": (C_BASE, "--")}
    for name, v in d["lolh"].items():
        col, ls = style[name]
        ax.plot(r, v, ls, marker="o", ms=3, color=col, label=name)
        be = d["break_even"][name]
        if be and np.isfinite(be):
            ax.plot(100 * be, d["criterion"], "o", ms=7, mfc="white", color=col)
    ax.axhline(d["criterion"], color="k", ls="--", lw=1)
    ax.text(r[-1], d["criterion"] + 0.15, "limit 2.8 h/yr", ha="right", fontsize=7)
    for x, txt in ((1.1, "link's measured rate,\nlast 12 months"), (5.0, "Hydro's planning\nvalue")):
        ax.axvline(x, color="k", lw=0.6, ls=":")
        ax.text(x, 9.85, txt, ha="center", va="top", fontsize=7, bbox=dict(fc="white", ec="none", pad=1))
    ax.set_xlabel("Labrador-Island Link winter outage rate (%)"); ax.set_ylabel("expected blackout hours / year")
    ax.set_ylim(0, 10); ax.set_yticks(range(0, 10, 2)); ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, 0.88))
    ax.set_title("2030 without Holyrood: how much link trouble each plan can take", fontsize=9)
    return ax


def decision_samples(ax=None):
    """Samples each method needs to call pass/fail for the strongest no-turbine plan, by link outage rate."""
    d = _load("turbine_decision.json")
    ax = ax or plt.gca()
    c = pd.DataFrame(d["checks"])
    x = 100 * c.link_rate
    for k, col, mk, lab in (("n_mc", C_MC, "o", "Monte Carlo"), ("n_is", C_IS, "s", "tuned importance sampling"),
                            ("n_qae", C_Q, "D", "quantum amplitude estimation (ideal)")):
        ax.semilogy(x, c[k], marker=mk, ms=4, color=col, label=lab)
    be = d["break_even"]["strongest plan without the turbine"]
    ax.axvline(100 * be, color=C_Q, lw=0.8, ls=":")
    ax.text(100 * be, ax.get_ylim()[1], " plan sits on the limit", fontsize=7, va="top", color=C_Q)
    ax.set_xlabel("Labrador-Island Link winter outage rate (%)"); ax.set_ylabel("samples to call pass or fail (log)")
    ax.set_title("Strongest plan without the turbine: cost of a confident answer", fontsize=9); ax.legend(fontsize=7)
    return ax


def coldsnap_plans(axs=None):
    """The bottleneck: quantum vs Monte Carlo on the multi-day cold-snap model, four plans at two link rates."""
    d = _load("coldsnap_plans.json")
    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
    cases = d["cases"]
    short = {"Hydro's plan: Unit 8 + turbine": "Hydro's plan", "cheapest plan without the turbine": "cheapest, no turbine",
             "strongest plan without the turbine": "strongest, no turbine", "same, plus the turbine": "strongest + turbine"}
    labels = [f"{short[c['plan']]}, link {c['link']}" for c in cases]
    y = np.arange(len(cases))[::-1]
    for yy, c in zip(y, cases):
        axs[0].errorbar(c["p_exact"] * 100, yy + 0.18, xerr=2 * c["mc_sd"] * 100, fmt="o", color=C_ML, ms=5, capsize=3, lw=1.6)
        axs[0].errorbar(c["qae_mean"] * 100, yy - 0.18, xerr=2 * c["qae_sd"] * 100, fmt="D", color=C_Q, ms=5, capsize=3, lw=1.6)
        axs[0].plot(c["p_exact"] * 100, yy, "|", color="k", ms=16, mew=2)
    axs[0].axvline(d["safe"] * 100, color=C_RED, ls="--", lw=1.2)
    axs[0].text(d["safe"] * 100 + 0.2, y[0] + 0.45, "safe line, 10%", color=C_RED, fontsize=8)
    axs[0].set_yticks(y); axs[0].set_yticklabels(labels, fontsize=8)
    axs[0].set_xlabel("chance of a day short of power in the winter's worst 3-day cold snap (%)")
    axs[0].plot([], [], "o", color=C_ML, label=f"Monte Carlo, {cases[0]['runs']:,} runs (range = 2 sd)")
    axs[0].plot([], [], "D", color=C_Q, label=f"quantum, same runs, real 22-qubit circuits on Aer")
    axs[0].plot([], [], "|", color="k", ms=10, mew=2, label="exact answer")
    axs[0].legend(fontsize=7, loc="lower right")
    axs[0].set_title("Same number of runs: quantum lands on the exact answer, Monte Carlo spreads wide", fontsize=9)
    e_mc = [100 * c["mc_sd"] / c["p_exact"] for c in cases]; e_q = [100 * c["qae_sd"] / c["p_exact"] for c in cases]
    axs[1].barh(y + 0.18, e_mc, height=0.36, color=C_ML, label="Monte Carlo")
    axs[1].barh(y - 0.18, e_q, height=0.36, color=C_Q, label="quantum")
    for yy, a, b in zip(y, e_mc, e_q):
        axs[1].text(a + 1, yy + 0.18, f"{a / b:.1f}x", va="center", fontsize=7)
    axs[1].set_yticks(y); axs[1].set_yticklabels([])
    axs[1].set_xlabel("typical error with the same 3,500 runs (% of the answer)"); axs[1].legend(fontsize=7, loc="lower right")
    axs[1].set_title("Error with the same runs (lower is better)", fontsize=9)
    return axs


def circuit_diagram(path=None):
    """High-level drawing of A for a small instance (register blocks, before flattening)."""
    from gridrisk.quantum import LolpInstance, state_preparation
    pf = np.full((3, 4, 2), 0.05); pf[2, :, :] = np.array([0, 0, 0.01, 0.02])[:, None]
    inst = LolpInstance(names=["LIL", "BdE U7", "frazil"], cap=[14, 3, 9], p_fail=pf, p_level=np.array([.6, .25, .1, .05]),
                        base=24, step=1)
    A, _ = state_preparation(inst, flat=False)
    fig = A.draw("mpl", fold=-1, scale=0.6)
    if path:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


FIGURES = {  # name: (function, figure size, number of panels, panel width ratios)
    "demand_model": (demand_model, (15, 3.8), 3, [2.2, 1, 1]),
    "weather_years": (weather_years, (9, 3.2), 1, None),
    "wind_at_peak": (wind_at_peak, (5, 3.4), 1, None),
    "heat_pumps": (heat_pumps, (10, 3.6), 2, None),
    "decision_2030": (decision_2030, (10, 4.2), 1, None),
    "core_check": (core_check, (10, 3.6), 2, None),
    "rarity_advantage": (rarity_advantage, (11, 3.8), 2, None),
    "hardware_scale": (hardware_scale, (10, 3.6), 2, None),
    "cold_snap": (cold_snap, (15, 3.8), 3, None),
    "hardware_real": (hardware_real, (11, 3.8), 2, None),
    "plans_must_have": (plans_must_have, (10, 5.2), 1, None),
    "plans_sampler": (plans_sampler, (11, 3.8), 2, None),
    "turbine": (turbine, (7.5, 4.4), 1, None),
    "winters_and_wind": (lambda axs: (weather_years(axs[0]), wind_at_peak(axs[1])), (14, 3.4), 2, [2, 1]),
    "decision_samples": (decision_samples, (7.5, 4.0), 1, None),
    "coldsnap_plans": (coldsnap_plans, (13, 4.6), 2, [1.5, 1]),
}


def show(*names):
    """Draw one or more of the figures above in a notebook."""
    for name in names:
        fn, size, n, ratios = FIGURES[name]
        fig, axs = plt.subplots(1, n, figsize=size, **(dict(gridspec_kw=dict(width_ratios=ratios)) if ratios else {}))
        fn(axs)
        fig.tight_layout()
        plt.show()


def save_all():
    FIG.mkdir(exist_ok=True)
    for name, (fn, size, n, ratios) in FIGURES.items():
        try:
            fig, axs = plt.subplots(1, n, figsize=size, **(dict(gridspec_kw=dict(width_ratios=ratios)) if ratios else {}))
            fn(axs)
            fig.tight_layout()
            fig.savefig(FIG / f"{name}.png", dpi=150)
            print("saved", name)
        except FileNotFoundError as e:
            print("skipped", name, "(missing", Path(e.filename).name + ")")
        finally:
            plt.close("all")
    circuit_diagram(FIG / "circuit_A.png")
    plt.close("all")
