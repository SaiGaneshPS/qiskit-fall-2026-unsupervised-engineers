"""Classical reliability engine for the island grid: will available supply cover demand, hour by hour?

Loss-of-load expectation (LOLE, hours/year) = sum over hours of P(available capacity < demand).
Available capacity is random (forced outages of each unit, independent given the weather, with outage
rates optionally raised in severe cold); demand comes from the AI model under each historical winter's
weather, plus the model's own error. Wind output is computed from the same weather, so the
cold/calm or cold/windy correlation is captured automatically.

Ground truth: the capacity outage probability table (COPT, exact convolution). Baseline estimator:
Monte Carlo sampling of outages and demand errors. The quantum estimator targets the same quantity.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class Unit:
    name: str
    mw: float
    efor: float                 # equivalent forced outage rate (probability unavailable at a random hour)
    kind: str                   # hydro, thermal_oil, ct_diesel, hvdc, wind, battery, import, ...
    fuel: str = "none"
    cold_sensitive: bool = False


@dataclass
class Portfolio:
    name: str
    units: list[Unit]
    wind_mw: float = 0.0                     # wind fleet, output from the weather-driven capacity factor
    wind_cf_scale: float = 1.0
    demand_delta: str | None = None         # name of a demand modifier (heat-pump scenario) or None
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def without(self, *prefixes: str) -> "Portfolio":
        keep = [u for u in self.units if not any(u.name.startswith(p) for p in prefixes)]
        return replace(self, units=keep)

    def plus(self, *units: Unit, name: str | None = None, wind_mw: float | None = None) -> "Portfolio":
        return replace(self, units=self.units + list(units), name=name or self.name,
                       wind_mw=self.wind_mw if wind_mw is None else wind_mw)

    @property
    def firm_mw(self) -> float:
        return float(sum(u.mw for u in self.units))


def copt(units: list[Unit], step: float = 1.0, cold_mult: float = 1.0) -> np.ndarray:
    """Distribution of available capacity (MW grid of size `step`): out[i] = P(available = i*step)."""
    total = sum(int(round(u.mw / step)) for u in units)
    dist = np.zeros(total + 1)
    dist[0] = 1.0
    cur = 0
    for u in units:
        c = int(round(u.mw / step))
        q = min(u.efor * (cold_mult if u.cold_sensitive else 1.0), 0.999)
        new = np.zeros_like(dist)
        new[: cur + 1] += dist[: cur + 1] * q                  # unit out
        new[c: cur + c + 1] += dist[: cur + 1] * (1 - q)       # unit available
        dist = new
        cur += c
    return dist


def error_distribution(resid: np.ndarray, bins: int = 41) -> tuple[np.ndarray, np.ndarray]:
    """Discretise demand-forecast errors (MW) into (values, weights) for the LOLP mixture."""
    h, e = np.histogram(resid, bins=bins)
    return 0.5 * (e[1:] + e[:-1]), h / h.sum()


def lolp_hours(demand: np.ndarray, err: tuple[np.ndarray, np.ndarray] | float, avail_dist: np.ndarray,
               step: float = 1.0, extra_supply: np.ndarray | None = None) -> np.ndarray:
    """P(shortfall) for each hour: P(available + extra < demand + e).

    err is either (values, weights) of an empirical forecast-error distribution, or a Gaussian sigma.
    """
    net = demand - (0 if extra_supply is None else extra_supply)
    if np.isscalar(err):
        cap = np.arange(len(avail_dist)) * step
        nz = avail_dist > 1e-15
        cap, pr = cap[nz], avail_dist[nz]
        out = np.empty(len(net))
        for i0 in range(0, len(net), 4096):
            z = (net[i0:i0 + 4096, None] - cap[None, :]) / max(float(err), 1e-9)
            out[i0:i0 + 4096] = norm.cdf(z) @ pr
        return out
    vals, w = err
    below = np.concatenate([[0.0], np.cumsum(avail_dist)])      # below[i] = P(available < i * step)
    x = net[:, None] + vals[None, :]
    idx = np.clip(np.ceil(x / step - 1e-9).astype(int), 0, len(avail_dist))
    return below[idx] @ w


def wind_power_curve(ws_100m_kmh: np.ndarray) -> np.ndarray:
    """Generic 3 MW-class turbine curve (cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s), with 10% losses."""
    v = np.asarray(ws_100m_kmh) / 3.6
    cf = np.clip((v ** 3 - 3 ** 3) / (12 ** 3 - 3 ** 3), 0, 1)
    cf[(v < 3) | (v > 25)] = 0.0
    return 0.9 * cf


def lole(portfolio: Portfolio, demand_hours: pd.DataFrame, err, wind_cf: pd.Series | None = None,
         cold_mult: float = 1.0, cold_threshold_c: float = -10.0, step: float = 2.0,
         common_mode: list[tuple[str, float]] | None = None) -> dict:
    """LOLH (expected loss-of-load hours per year) and per-winter results over all weather years.

    demand_hours: columns demand_mw, T_island, winter (from ml.weather_years, plus any scenario delta).
    cold_mult multiplies the outage rate of cold-sensitive units in hours colder than cold_threshold_c.
    common_mode: list of (unit-name prefix, probability per cold hour) for whole-plant cold-weather outages,
    e.g. ("Bay d'Espoir", p) for the January 2026 frazil-ice shutdown of all seven units.
    """
    dh = demand_hours
    extra = None
    if portfolio.wind_mw > 0 and wind_cf is not None:
        extra = portfolio.wind_mw * portfolio.wind_cf_scale * wind_cf.reindex(dh.index).fillna(0).values
    cold = dh.T_island.values < cold_threshold_c
    p = np.zeros(len(dh))
    relevant = dh.demand_mw.values > 0.45 * portfolio.firm_mw          # other hours cannot be short
    for is_cold, mult in [(False, 1.0), (True, cold_mult)]:
        m = relevant & (cold == is_cold)
        if not m.any():
            continue
        ex = None if extra is None else extra[m]
        base = lolp_hours(dh.demand_mw.values[m], err, copt(portfolio.units, step, mult), step, ex)
        if is_cold and common_mode:
            p_none = 1.0
            mix = np.zeros(m.sum())
            for prefix, pc in common_mode:
                units = [u for u in portfolio.units if not u.name.startswith(prefix)]
                mix += pc * lolp_hours(dh.demand_mw.values[m], err, copt(units, step, mult), step, ex)
                p_none -= pc
            base = p_none * base + mix
        p[m] = base
    s = pd.Series(p, index=dh.index)
    per_winter = s.groupby(dh.winter).sum()
    return dict(lole_h_per_yr=float(per_winter.mean()), lole_by_winter=per_winter, lolp=s,
                worst_winter=int(per_winter.idxmax()), worst_winter_lole=float(per_winter.max()))
