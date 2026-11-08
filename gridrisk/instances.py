"""From the AI + reliability pipeline to a quantum-ready LOLP instance.

The question the quantum circuit answers: "in a random winter hour (December to March, any of 46 weather
years), is available supply below demand?"  Expected loss-of-load hours per winter = hours x that probability.

Everything except the K biggest units is folded, exactly, into one "net load" register:
    Y = demand (AI, weather year) + forecast error (AI out-of-sample) - wind (same weather)
        - available capacity of all small units (exact outage convolution) - capacity of the K big units
Shortfall  <=>  sum_k c_k f_k + Y > 0,   with f_k = 1 if big unit k is forced out.
Y is rounded up to `quantum_mw` (conservative) and clipped to [Y_min, 1]: Y >= 1 is a certain shortfall, so
clipping above is exact; clipping below only matters if every big unit fails at once (negligible).
The frazil-ice common-mode outage of Bay d'Espoir units 1-6 is a unit whose probability depends on the
net-load level through P(hour is colder than -10 C | level), learned from the same weather years.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .quantum import LolpInstance
from .reliability import Portfolio, copt

SEASON_MONTHS = (12, 1, 2, 3)


def season_hours(dh: pd.DataFrame) -> pd.DataFrame:
    loc = dh.index.tz_convert("America/St_Johns")
    return dh[np.isin(loc.month, SEASON_MONTHS)]


def build_instance(portfolio: Portfolio, dh: pd.DataFrame, err: tuple[np.ndarray, np.ndarray], wind_cf: pd.Series,
                   big: list[str], frazil: tuple[float, float] | None = None, quantum_mw: float = 50.0,
                   n_d: int = 5, cold_c: float = -10.0) -> LolpInstance:
    """big: names of units that get their own qubit (rounded down to whole quanta; the remainder stays firm).
    frazil: (capacity MW, probability per cold hour) of the common-mode Bay d'Espoir U1-6 outage, or None."""
    hrs = season_hours(dh)
    wind = portfolio.wind_mw * wind_cf.reindex(hrs.index).fillna(0).values
    x = hrs.demand_mw.values - wind
    cold = hrs.T_island.values < cold_c

    big_units = [u for u in portfolio.units if u.name in big]
    small = [u for u in portfolio.units if u.name not in big]
    caps = [max(1, int(round(u.mw / quantum_mw))) for u in big_units]
    remainder = sum(u.mw - c * quantum_mw for u, c in zip(big_units, caps))      # stays firm (can be < 0)
    names = [u.name for u in big_units]
    pf = [u.efor for u in big_units]
    cap_total_mw = quantum_mw * sum(caps)          # real capacity behind the big-unit qubits
    if frazil is not None:
        # Units 1-6 already sit in the small-unit convolution; this qubit only *removes* their capacity
        # when the plant-wide frazil shutdown happens (it adds no capacity of its own).
        caps.append(int(round(frazil[0] / quantum_mw)))
        names.append("Bay d'Espoir U1-6 frazil shutdown")

    # distribution of Z = error - small-unit available capacity - firm remainder (1 MW grid, exact)
    s_dist = copt(small, step=1.0)                                   # P(small available = i MW)
    ev, ew = err
    lo = int(np.floor(ev.min())) - len(s_dist) - int(remainder) - 2
    z = np.zeros(int(np.ceil(ev.max())) - lo + 2)
    for v, wgt in zip(ev, ew):
        base = int(round(v)) - int(round(remainder)) - lo
        idx = base - np.arange(len(s_dist))
        np.add.at(z, idx, wgt * s_dist)
    # Y_h = x_h - cap_total + Z, in quanta (ceil), clipped to [Y_min, 1]
    L = 2 ** n_d
    y_min = 2 - L
    zq = np.arange(len(z)) + lo                                      # MW value of each z bin
    p_level = np.zeros(L)
    p_cold = np.zeros(L)
    xq = np.round(x - cap_total_mw).astype(int)
    vals, counts = np.unique(np.column_stack([xq, cold]), axis=0, return_counts=True)
    nz = z > 0
    for (xv, cv), cnt in zip(vals, counts):
        yq = np.ceil((xv + zq[nz]) / quantum_mw).astype(int)
        b = np.clip(yq, y_min, 1) - y_min
        contrib = np.bincount(b, weights=z[nz], minlength=L) * cnt
        p_level += contrib
        if cv:
            p_cold += contrib
    tot = p_level.sum()
    p_cold_given = np.divide(p_cold, p_level, out=np.zeros(L), where=p_level > 0)
    p_level /= tot
    K = len(caps)
    p_fail = np.zeros((K, L, 2))
    for k in range(len(big_units)):
        p_fail[k] = pf[k]
    if frazil is not None:
        p_fail[-1] = (frazil[1] * p_cold_given)[:, None]
    base_q = y_min + sum(caps)
    inst = LolpInstance(names=names, cap=caps, p_fail=p_fail, p_level=p_level, base=base_q, step=1,
                        quantum_mw=quantum_mw)
    inst.labels = dict(season_hours_per_winter=float(len(hrs) / hrs.winter.nunique()), y_min_quanta=y_min,
                       big_units=names, remainder_mw=remainder, p_cold_given_level=p_cold_given.tolist())
    return inst


def lolh_from_instance(inst: LolpInstance, p: float | None = None) -> float:
    """Expected loss-of-load hours per winter season implied by the instance's probability."""
    return (inst.exact() if p is None else p) * inst.labels["season_hours_per_winter"]
