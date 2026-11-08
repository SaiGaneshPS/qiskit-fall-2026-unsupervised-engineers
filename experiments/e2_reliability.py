"""E2: will the lights stay on? The island's loss-of-load hours (LOLH) under 46 winters of weather.

Pipeline: ERA5 weather 1980-2025 -> AI demand model (E1) at today's load level -> demand level anchored to
NL Hydro's peak forecast for the study year -> exact capacity-outage convolution of the fleet, mixed over the
AI's out-of-sample forecast errors -> P(shortfall) every hour -> LOLH per winter (criterion: 2.8 h/yr).

1. Validation: the 2026-27 system vs NL Hydro's own published LOLH (1.8 h; 4.1 h with Holyrood at 34% DAUFOP).
2. The insight: wind at the Fermeuse site blows harder in the coldest, highest-demand hours.
3. Heat pumps: oil -> heat pump adds load at the worst hours; baseboard -> heat pump removes it.
4. The 2030 decision: retire Holyrood's oil; how much wind / heat-pump conversion keeps LOLH <= 2.8 h
   without new diesel, and what each path does to CO2.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridrisk import data, fleet, ml, scenarios  # noqa: E402
from gridrisk.reliability import Unit, error_distribution, lole, wind_power_curve  # noqa: E402

FIRST_WINTER, LAST_WINTER = 1980, 2025
COLD_C = -10.0
# January 2026: frazil ice shut all seven Bay d'Espoir units, the first such shutdown since 1967; units 1-6
# were back after ~4 days. One ~96 h event in ~59 winters, spread over the ~268 island-cold hours per winter.
FRAZIL_EVENT_HOURS, FRAZIL_RETURN_YEARS = 96, 59
WIND_COLD_CUTOUT_C = -20.0           # standard cold-weather packages stop turbines near -20 to -30 C


def demand_hours(year: int) -> tuple[pd.DataFrame, float]:
    wy = ml.weather_years()
    wy = wy[(wy.winter >= FIRST_WINTER) & (wy.winter <= LAST_WINTER)].copy()
    shift = fleet.PEAK_FORECAST_MW[year] - wy.groupby("winter").demand_mw.max().median()
    wy["demand_mw"] += shift
    return wy, float(shift)


def forecast_errors() -> tuple[np.ndarray, np.ndarray, dict]:
    """Forward (one-season-ahead) residuals in high-demand hours, minus each season's mean bias (the level is
    anchored separately to Hydro's forecast)."""
    if not (ROOT / "results" / "e1_oos_residuals.csv").exists():
        ml.oos_residuals()
    r = pd.read_csv(ROOT / "results" / "e1_oos_residuals.csv")
    r = r[(r.scheme == "forward") & (r.pred >= 1350)].copy()
    r["dm"] = r.resid - r.groupby("season").resid.transform("mean")
    return *error_distribution(r.dm.values), dict(n=int(len(r)), sd=float(r.dm.std()),
                                                   q01=float(r.dm.quantile(0.01)), q99=float(r.dm.quantile(0.99)))


def wind_cf() -> pd.Series:
    fw = data.load_weather("fermeuse")
    cf = wind_power_curve(fw.wind_speed_100m.values)
    cf[fw.temperature_2m.values < WIND_COLD_CUTOUT_C] = 0.0
    return pd.Series(cf, index=fw.index)


def frazil_p() -> float:
    return FRAZIL_EVENT_HOURS / (FRAZIL_RETURN_YEARS * 268.0)


def with_holyrood_rate(p, rate):
    return replace(p, units=[replace(u, efor=rate) if u.name.startswith("Holyrood U") else u for u in p.units])


def wind_needed(p, dh, err, cf, target=fleet.LOLH_CRITERION_H, hi=1500.0, **kw) -> float | None:
    """Smallest added wind (MW, on top of the existing 54 MW) that brings LOLH to the target (bisection);
    None if even `hi` MW is not enough."""
    def f(w):
        return lole(replace(p, wind_mw=54.0 + w), dh, err, wind_cf=cf, **kw)["lole_h_per_yr"]
    if f(0.0) <= target:
        return 0.0
    if f(hi) > target:
        return None
    lo = 0.0
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > target else (lo, mid)
    return hi


def set_lil(p, eqfor):
    return replace(p, units=[fleet.lil(fleet.LIL_MW, eqfor) if u.name.startswith("Labrador") else u for u in p.units])


def run():
    vals, w, err_info = forecast_errors()
    err = (vals, w)
    cf = wind_cf()
    out = dict(forecast_error=err_info, frazil_p_per_cold_hour=frazil_p())

    # ---------------------------------------------------------------- 1. validation against NL Hydro (2026)
    dh26, shift26 = demand_hours(2026)
    val = {}
    for rate in (0.20, 0.34):
        r = lole(with_holyrood_rate(fleet.today(), rate), dh26, err, wind_cf=cf)
        val[f"holyrood_{int(rate * 100)}pct"] = r["lole_h_per_yr"]
    r_cm = lole(fleet.today(), dh26, err, wind_cf=cf, common_mode=[("Bay d'Espoir", frazil_p())])
    out["validation_2026"] = dict(ours_h=val, hydro_published_h={"holyrood_20pct": 1.8, "holyrood_34pct": 4.1},
                                  with_frazil_common_mode_h=r_cm["lole_h_per_yr"], level_shift_mw=shift26)
    print("2026-27 LOLH: ours", {k: round(v, 2) for k, v in val.items()}, " NL Hydro: 1.8 / 4.1 h;  with frazil risk",
          round(r_cm["lole_h_per_yr"], 2))

    # ---------------------------------------------------------------- 2. wind in the coldest hours
    j = dh26.join(cf.rename("cf"), how="inner")
    out["wind_at_peak"] = {f"top_{q}": float(j.cf[j.demand_mw >= j.demand_mw.quantile(1 - q)].mean())
                           for q in (1.0, 0.1, 0.01, 0.001)}
    print("Fermeuse wind capacity factor: all hours", round(out["wind_at_peak"]["top_1.0"], 3),
          " top 1% demand hours", round(out["wind_at_peak"]["top_0.01"], 3),
          " top 0.1%", round(out["wind_at_peak"]["top_0.001"], 3))

    # ---------------------------------------------------------------- 3. heat pumps at the worst hours
    lin = ml.LinearDegreeHour().fit(ml.dataset(), ml.dataset().demand_mw)
    slope = float(lin.m.coef_[0] + lin.m.coef_[1])           # MW per island heating degree (hdd + hdd24)
    ua = scenarios.ua_per_home_kw(slope)
    T = dh26.T_island.values
    hp = {}
    for label, kw in [("40k oil homes -> heat pump", dict(oil_to_hp=40_000)),
                      ("40k oil homes -> heat pump, oil furnace kept below -10 C", dict(oil_to_hp=40_000, oil_backup_below_c=-10)),
                      ("50k baseboard homes -> heat pump", dict(baseboard_to_hp=50_000))]:
        d = scenarios.demand_delta_mw(T, ua, **kw)
        dd = dh26.copy(); dd["demand_mw"] += d
        hp[label] = dict(delta_at_coldest_hour_mw=float(d[np.argmin(T)]),
                         delta_median_winter_peak_mw=float(dd.groupby("winter").demand_mw.max().median()
                                                          - dh26.groupby("winter").demand_mw.max().median()),
                         lolh=lole(fleet.today(), dd, err, wind_cf=cf)["lole_h_per_yr"])
    out["heat_pumps_2026"] = dict(slope_mw_per_degree=slope, ua_kw_per_degree_per_home=ua, scenarios=hp,
                                  oil_co2_kt_per_40k_homes=scenarios.oil_co2_t_per_year(40_000) / 1000)
    for k, v in hp.items():
        print(f"  {k:58s} coldest hour {v['delta_at_coldest_hour_mw']:+6.0f} MW  median peak {v['delta_median_winter_peak_mw']:+6.0f} MW"
              f"  LOLH {v['lolh']:.2f}")

    # ---------------------------------------------------------------- 4. the 2030 decision
    # First winter after Holyrood's steam units retire (peak forecast 1,860 MW). Each portfolio is evaluated
    # under Hydro's planning LIL outage rate (5% EqFOR) and the LIL's measured rate (1.1%, 12 months to Sep 2026).
    dh30, shift30 = demand_hours(2030)
    T30 = dh30.T_island.values
    holy_co2 = float(np.mean(list(fleet.HOLYROOD_CO2_KT.values())))
    oil_co2 = scenarios.oil_co2_t_per_year(40_000) / 1000
    dd_bb = dh30.copy(); dd_bb["demand_mw"] += scenarios.demand_delta_mw(T30, ua, baseboard_to_hp=50_000)
    dd_both = dh30.copy(); dd_both["demand_mw"] += scenarios.demand_delta_mw(T30, ua, baseboard_to_hp=50_000,
                                                                              oil_to_hp=40_000, oil_backup_below_c=-10)
    base = fleet.no_holyrood_2030()
    u8 = base.plus(Unit("Bay d'Espoir U8 (new)", 150, fleet.HYDRO_DAFOR, "hydro", cold_sensitive=True))
    cases = [  # label, portfolio, demand, added wind MW, new fossil MW, CO2 avoided (kt/yr)
        ("A. Holyrood retired, nothing added", base, dh30, 0, 0, holy_co2),
        ("B. Hydro's plan: new Avalon diesel CT + Bay d'Espoir U8", fleet.hydro_plan_2030(), dh30, 0, 150, holy_co2),
        ("C. No new diesel: Bay d'Espoir U8 + 300 MW wind", u8, dh30, 300, 0, holy_co2),
        ("D. C + 50k baseboard homes -> heat pumps", u8, dd_bb, 300, 0, holy_co2),
        ("E. D + 40k oil homes -> heat pumps (oil backup < -10 C)", u8, dd_both, 300, 0, holy_co2 + oil_co2),
        ("F. E with 600 MW wind and no Bay d'Espoir U8", base, dd_both, 600, 0, holy_co2 + oil_co2),
    ]
    portfolios = {}
    print("2030 portfolios (first winter without Holyrood oil; peak forecast 1,860 MW)   LOLH at LIL EqFOR 5% / 1.1%")
    for label, p, dd, wadd, fossil, co2 in cases:
        res = {}
        for tag, eq in [("lil_5pct", fleet.LIL_EQFOR), ("lil_1pct", 0.011)]:
            q = replace(set_lil(p, eq), wind_mw=54.0 + wadd)
            r = lole(q, dd, err, wind_cf=cf)
            res[tag] = dict(lolh=r["lole_h_per_yr"], worst_winter=r["worst_winter"], worst_winter_lolh=r["worst_winter_lole"],
                            lolh_with_frazil=lole(q, dd, err, wind_cf=cf, common_mode=[("Bay d'Espoir", frazil_p())])["lole_h_per_yr"],
                            lolh_by_winter=r["lole_by_winter"].round(4).to_dict())
        portfolios[label] = dict(**res, firm_mw=p.firm_mw, wind_mw=54.0 + wadd, new_fossil_mw=fossil,
                                 co2_avoided_kt_per_yr=co2,
                                 meets_criterion_planning=res["lil_5pct"]["lolh"] <= fleet.LOLH_CRITERION_H,
                                 meets_criterion_measured=res["lil_1pct"]["lolh"] <= fleet.LOLH_CRITERION_H)
        print(f"  {label:58s} {res['lil_5pct']['lolh']:7.2f} h / {res['lil_1pct']['lolh']:5.2f} h   "
              f"CO2 avoided {co2:5.0f} kt/yr   new fossil {fossil} MW")
    need = {lbl: wind_needed(p, dd, err, cf) for lbl, p, dd in
            [("U8 + wind", u8, dh30), ("U8 + wind + 50k baseboard HP", u8, dd_bb),
             ("wind + all heat pumps, no U8", base, dd_both)]}
    print("  added wind needed to meet 2.8 h at 5% EqFOR (None = more than 1,500 MW):",
          {k: (None if v is None else round(v)) for k, v in need.items()})
    out["decision_2030"] = dict(level_shift_mw=shift30, portfolios=portfolios, wind_needed_mw_at_5pct=need,
                                holyrood_co2_kt=holy_co2, oil_homes_co2_kt=oil_co2)
    (ROOT / "results" / "e2_reliability.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    run()
