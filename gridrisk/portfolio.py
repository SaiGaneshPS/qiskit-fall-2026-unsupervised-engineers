"""Every good 2030 plan: which investments keep the lights on, cut CO2 and lower bills?

Twelve yes/no options for the first winter after Holyrood's oil units retire (one qubit each, 4,096 plans).
Every plan is scored in one currency, $ per year for Newfoundland and Labrador:
    + annualised capital cost and operating cost of what is built
    - household energy savings (heat pumps lower bills; oil homes stop buying oil)
    - value of the wind energy produced
    + CO2 at a carbon value per tonne
    + blackout cost: a cost per expected loss-of-load hour, plus a steep penalty per hour above the 2.8 h/yr
      criterion (so plans that break the standard are effectively ruled out)
Loss-of-load hours come from the validated reliability engine (46 weather years, AI demand, AI errors,
weather-driven wind at each site). Heat-pump loads use the same data-calibrated home model as E2.

Cited inputs: Avalon CT $995.9M and Bay d'Espoir Unit 8 $1.08B (NL Hydro filings); battery cost range
$159M-$437M for 20-50 MW (NL Hydro battery feasibility study); battery capacity value 37% for the first 50 MW
and 23% at 100 MW (NL Hydro ELCC study). Everything else in ASSUMPTIONS is a stated assumption.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import fleet, scenarios
from .reliability import Portfolio, Unit, lole

ASSUMPTIONS = dict(
    electricity_rate=0.14,        # $/kWh, approximate NL residential rate
    oil_price=1.60,               # $/L heating oil, approximate
    kwh_per_litre=10.0,           # heating value of furnace oil
    furnace_efficiency=0.80,
    wind_capex_per_mw=2.3,        # $M per MW, typical recent onshore wind
    wind_om_per_mw=0.04,          # $M per MW-year
    wind_energy_value=50.0,       # $/MWh, export or reservoir-storage value
    carbon_price=170.0,           # $/t CO2, Canada's 2030 benchmark price for large emitters
    blackout_hour_cost=10.0,      # $M per expected loss-of-load hour
    criterion_penalty=100.0,      # $M per hour above the 2.8 h criterion
    crf_long=0.07,                # capital recovery factor, plants and wind (about 5% over 30 years)
    crf_short=0.09,               # batteries, heat pumps, demand response (about 5% over 15-18 years)
    hp_cost_baseboard=5000.0,     # $ per home, ductless heat pump installed
    hp_cost_oil=15000.0,          # $ per home, heat pump plus electrical work
    dr_mw=40.0, dr_unavailability=0.10, dr_capex=20.0,
    money_unit=10.0,              # energy unit for the sampler: $10M per year
)

# key, label, kind, size
OPTIONS = [
    ("ct", "Avalon combustion turbine, 150 MW diesel", "ct", 150),
    ("u8", "Bay d'Espoir Unit 8, 150 MW hydro", "hydro", 150),
    ("wind_avalon", "Wind, Avalon (Fermeuse area), 100 MW", "wind", 100),
    ("wind_burin", "Wind, Burin (St. Lawrence area), 100 MW", "wind", 100),
    ("wind_west", "Wind, west coast (Port au Port), 100 MW", "wind", 100),
    ("bat_small", "Battery, 25 MW for 4 h", "battery", 25),
    ("bat_large", "Battery, 50 MW for 4 h", "battery", 50),
    ("bb_small", "15,000 baseboard homes to heat pumps", "bb", 15000),
    ("bb_large", "35,000 baseboard homes to heat pumps", "bb", 35000),
    ("oil_small", "10,000 oil homes to heat pumps", "oil", 10000),
    ("oil_large", "30,000 oil homes to heat pumps", "oil", 30000),
    ("dr", "Smart-thermostat demand response, 40 MW", "dr", 40),
]
WIND_SITE = {"wind_avalon": "fermeuse", "wind_burin": "stlawrence", "wind_west": "portauport"}
CT_CAPEX, U8_CAPEX = 995.9, 1080.0


def battery_firm_mw(total_mw: float) -> float:
    """Firm (capacity-equivalent) MW of 4-hour batteries, interpolating NL Hydro's ELCC results."""
    return float(np.interp(total_mw, [0, 25, 50, 75, 100], [0.0, 10.0, 18.5, 21.5, 23.0]))


def battery_capex(mw: float) -> float:
    """$M, linear between 20 MW and 50 MW points of NL Hydro's 4-hour estimates (inside its $159M-$437M range)."""
    return float(np.interp(mw, [0, 25, 50, 75], [0.0, 140.0, 250.0, 375.0]))


class PlanModel:
    """Scores all 4,096 plans. dh: 2030 weather-year demand hours; err: AI error distribution;
    wind_cf: {site: hourly capacity-factor series}; ua: kW per degree per home."""

    def __init__(self, dh: pd.DataFrame, err, wind_cf: dict, ua: float, lil_eqfor: float, A=ASSUMPTIONS):
        self.dh, self.err, self.wind_cf, self.ua, self.lil_eqfor, self.A = dh, err, wind_cf, ua, lil_eqfor, A
        T = dh.T_island.values
        n_years = dh.winter.nunique()
        heat = scenarios.heat_kw(T, ua)                 # kW per home, every hour
        cop = scenarios.cop(T)
        warm = T >= -10.0
        self.per_home = dict(
            heat_kwh=float(heat.sum() / n_years),
            bb_kwh_saved=float((heat * (1 - 1 / cop)).sum() / n_years),
            oil_kwh_added=float((heat / cop * warm).sum() / n_years),
            oil_share_replaced=float((heat * warm).sum() / heat.sum()),
        )
        self.per_home["oil_litres_saved"] = self.per_home["oil_share_replaced"] * self.per_home["heat_kwh"] / (
            A["kwh_per_litre"] * A["furnace_efficiency"])
        self.per_home["bb_bill_saving"] = self.per_home["bb_kwh_saved"] * A["electricity_rate"]
        self.per_home["oil_bill_saving"] = (self.per_home["oil_litres_saved"] * A["oil_price"]
                                            - self.per_home["oil_kwh_added"] * A["electricity_rate"])
        self.per_home["oil_co2_t"] = self.per_home["oil_litres_saved"] * scenarios.KG_CO2_PER_LITRE_OIL / 1000
        self.wind_mwh = {k: float(wind_cf[s].reindex(dh.index).fillna(0).mean() * 8760 * 100) for k, s in WIND_SITE.items()}
        self._dd = {}
        self._lolh = {}

    def demand(self, n_bb: int, n_oil: int) -> pd.DataFrame:
        key = (n_bb, n_oil)
        if key not in self._dd:
            d = self.dh.copy()
            d["demand_mw"] = d.demand_mw + scenarios.demand_delta_mw(d.T_island.values, self.ua, baseboard_to_hp=n_bb,
                                                                      oil_to_hp=n_oil, oil_backup_below_c=-10)
            self._dd[key] = d
        return self._dd[key]

    def choices(self, x) -> dict:
        on = {k: bool(v) for (k, *_), v in zip(OPTIONS, x)}
        size = {k: s for k, _, _, s in OPTIONS}
        return dict(on=on, n_bb=sum(size[k] for k in ("bb_small", "bb_large") if on[k]),
                    n_oil=sum(size[k] for k in ("oil_small", "oil_large") if on[k]),
                    bat_mw=sum(size[k] for k in ("bat_small", "bat_large") if on[k]),
                    winds=tuple(k for k in WIND_SITE if on[k]))

    def lolh(self, x) -> float:
        c = self.choices(x)
        key = (c["on"]["ct"], c["on"]["u8"], c["winds"], c["bat_mw"], c["n_bb"], c["n_oil"], c["on"]["dr"])
        if key in self._lolh:
            return self._lolh[key]
        base = fleet.no_holyrood_2030()
        units = [fleet.lil(fleet.LIL_MW, self.lil_eqfor) if u.name.startswith("Labrador") else u for u in base.units]
        if c["on"]["ct"]:
            units.append(Unit("Avalon CT (new, diesel)", 150, fleet.CT_DAUFOP, "ct", "diesel"))
        if c["on"]["u8"]:
            units.append(Unit("Bay d'Espoir U8 (new)", 150, fleet.HYDRO_DAFOR, "hydro", cold_sensitive=True))
        if c["bat_mw"]:
            units.append(Unit("Batteries (firm part)", battery_firm_mw(c["bat_mw"]), 0.03, "battery"))
        if c["on"]["dr"]:
            units.append(Unit("Demand response", self.A["dr_mw"], self.A["dr_unavailability"], "demand_response"))
        # wind: existing 54 MW at Fermeuse/St. Lawrence plus each new 100 MW site at its own weather
        cf = 27 * self.wind_cf["fermeuse"] + 27 * self.wind_cf["stlawrence"]
        for k in c["winds"]:
            cf = cf + 100 * self.wind_cf[WIND_SITE[k]]
        p = Portfolio("plan", units, wind_mw=1.0)
        val = lole(p, self.demand(c["n_bb"], c["n_oil"]), self.err, wind_cf=cf)["lole_h_per_yr"]
        self._lolh[key] = val
        return val

    def breakdown(self, x) -> dict:
        A, c = self.A, self.choices(x)
        on = c["on"]
        capex = opex = co2 = 0.0
        if on["ct"]:
            capex += CT_CAPEX * A["crf_long"]; opex += 3.0; co2 += 10.0
        if on["u8"]:
            capex += U8_CAPEX * A["crf_long"]; opex += 2.0
        for k in c["winds"]:
            capex += 100 * A["wind_capex_per_mw"] * A["crf_long"]; opex += 100 * A["wind_om_per_mw"]
        capex += battery_capex(c["bat_mw"]) * A["crf_short"]; opex += 0.02 * battery_capex(c["bat_mw"])
        capex += (c["n_bb"] * A["hp_cost_baseboard"] + c["n_oil"] * A["hp_cost_oil"]) / 1e6 * A["crf_short"]
        if on["dr"]:
            capex += A["dr_capex"] * A["crf_short"]; opex += 1.0
        wind_value = sum(self.wind_mwh[k] for k in c["winds"]) * A["wind_energy_value"] / 1e6
        bills = (c["n_bb"] * self.per_home["bb_bill_saving"] + c["n_oil"] * self.per_home["oil_bill_saving"]) / 1e6
        co2 -= c["n_oil"] * self.per_home["oil_co2_t"] / 1000
        lolh = self.lolh(x)
        blackout = A["blackout_hour_cost"] * lolh + A["criterion_penalty"] * max(0.0, lolh - fleet.LOLH_CRITERION_H)
        total = capex + opex - wind_value - bills + co2 * A["carbon_price"] / 1000 + blackout
        return dict(capex=capex, opex=opex, wind_value=wind_value, household_savings=bills, co2_kt=co2,
                    carbon_cost=co2 * A["carbon_price"] / 1000, lolh=lolh, blackout_cost=blackout, net_cost=total,
                    built=[lab for (k, lab, *_), v in zip(OPTIONS, x) if v])


class PlanProblem:
    """Sampler interface (n, energy, energy_vector, all_configs) over the free options; energy in $10M/yr."""

    def __init__(self, table: np.ndarray, fixed: dict | None = None, unit: float = ASSUMPTIONS["money_unit"]):
        self.full = np.asarray(table) / unit            # net cost of all 4,096 plans, indexed little-endian
        self.fixed = fixed or {}
        self.free = [i for i in range(len(OPTIONS)) if i not in self.fixed]
        self.unit_names = [OPTIONS[i][1] for i in self.free]
        full_n = len(OPTIONS)
        k = np.arange(2 ** self.n, dtype=np.int64)
        bits = (k[:, None] >> np.arange(self.n)) & 1
        idx = np.zeros(len(k), dtype=np.int64)
        for j, i in enumerate(self.free):
            idx |= bits[:, j].astype(np.int64) << i
        for i, v in self.fixed.items():
            if v:
                idx |= np.int64(1) << i
        self._E = self.full[idx]
        assert full_n == 12

    @property
    def n(self) -> int:
        return len(self.free)

    def all_configs(self) -> np.ndarray:
        k = np.arange(2 ** self.n, dtype=np.int64)
        return ((k[:, None] >> np.arange(self.n)) & 1).astype(np.int8)

    def energy_vector(self) -> np.ndarray:
        return self._E

    def energy(self, x) -> np.ndarray:
        x = np.asarray(x)
        return self._E[(x.astype(np.int64) << np.arange(self.n)).sum(axis=-1)]
