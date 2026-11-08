"""Demand-side scenarios: switching homes to heat pumps, and what it does to the grid's worst hours.

Heat pump efficiency (coefficient of performance, COP) falls as it gets colder:
    +10 C: 3.3 and -8.3 C: 2.3 (NRCan, Heating and Cooling With a Heat Pump, typical air-source unit)
    +4.9 C: 3.0 and -21.1 C: 1.5 (NRC Canadian Centre for Housing Technology field test, cold-climate unit, 2013)
    -15 C: 1.75 (NEEP cold-climate ASHP specification v3.1, minimum COP at maximum capacity)
    below -25 C we assume electric-resistance backup (COP 1).

Per-home heating load is calibrated from the data, not assumed: the island's demand rises by S MW per
heating degree (from the linear degree-hour model). We attribute a share of S to electrically heated homes
(StatCan 38-10-0286-01: 64% of NL households heat mainly with electricity in 2023) to get kW per degree per home.

    oil -> heat pump:        +H(T) / COP(T)            (new electric load; zero if an oil furnace backs up)
    baseboard -> heat pump:  -H(T) * (1 - 1 / COP(T))  (the same heat for less electricity)
"""
from __future__ import annotations

import numpy as np

COP_T = np.array([-25.0, -21.1, -15.0, -8.3, 4.9, 10.0])
COP_V = np.array([1.3, 1.5, 1.75, 2.3, 3.0, 3.3])
T_BALANCE = 15.5
NL_HOUSEHOLDS_ISLAND = 215_000        # assumption: about 94% of NL's ~229,000 households are on the island
ELECTRIC_HEAT_SHARE = 0.64            # StatCan 38-10-0286-01, 2023
RESIDENTIAL_SHARE_OF_SLOPE = 0.70     # assumption: rest is commercial / institutional electric heat
OIL_LITRES_PER_HOME = 2400            # gov.nl.ca 2023 Oil to Electric example household
KG_CO2_PER_LITRE_OIL = 2.753          # ECCC emission factor, residential light fuel oil


def cop(T) -> np.ndarray:
    T = np.asarray(T, float)
    c = np.interp(T, COP_T, COP_V)
    return np.where(T < COP_T[0], 1.0, c)


def ua_per_home_kw(slope_mw_per_degree: float) -> float:
    homes = NL_HOUSEHOLDS_ISLAND * ELECTRIC_HEAT_SHARE
    return RESIDENTIAL_SHARE_OF_SLOPE * slope_mw_per_degree * 1000 / homes


def heat_kw(T, ua_kw: float) -> np.ndarray:
    return ua_kw * np.maximum(T_BALANCE - np.asarray(T, float), 0.0)


def demand_delta_mw(T, ua_kw: float, oil_to_hp: int = 0, baseboard_to_hp: int = 0,
                    oil_backup_below_c: float | None = None) -> np.ndarray:
    """Change in island demand (MW) for each hour's temperature T."""
    T = np.asarray(T, float)
    h = heat_kw(T, ua_kw)
    c = cop(T)
    add = h / c
    if oil_backup_below_c is not None:
        add = np.where(T < oil_backup_below_c, 0.0, add)       # hybrid: the old furnace takes the coldest hours
    return (oil_to_hp * add - baseboard_to_hp * h * (1 - 1 / c)) / 1000.0


def oil_co2_t_per_year(homes: int) -> float:
    return homes * OIL_LITRES_PER_HOME * KG_CO2_PER_LITRE_OIL / 1000.0
