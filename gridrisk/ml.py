"""AI layer: hourly island electricity demand from weather, and 46 winters of "what today's grid would face".

Model (gradient boosting with a physics cold tail):
    inside the weather range seen in training: gradient-boosted trees on weather, lags and calendar;
    colder than anything seen: the trees' prediction at the training edge plus a linear heating slope
    (MW per degree, fitted on the coldest training hours), because trees cannot extrapolate.
Baselines: linear degree-hour regression (standard utility practice) and gradient boosting without the tail.

Inputs: NL Hydro island demand (HFED) and ERA5 weather at St. John's, Gander and Corner Brook.
Train: Sep 2022 - Aug 2025 (three winters). Test: Dec 2025 - Oct 2026 (the 2025-26 winter, unseen).
Extrapolation test: train only on hours milder than -8 C (island-weighted), predict the colder hours.
Stress test: hold out the 2022-23 winter, which contains the island record (1,980 MW, 4 Feb 2023).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .data import load_demand, load_weather, nl_holidays

ROOT = Path(__file__).resolve().parents[1]
TZ = "America/St_Johns"
T_BASE = 15.5                         # heating balance temperature (C)
EPOCH = pd.Timestamp("2022-09-01", tz="UTC")
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
TEST_START = pd.Timestamp("2025-12-01", tz="UTC")
SEED = 7


# ---------------------------------------------------------------------------------------------- features
def weather_frame() -> pd.DataFrame:
    sj = load_weather("stjohns")
    out = pd.DataFrame(index=sj.index)
    out["T"] = sj.temperature_2m
    out["Td"] = sj.dew_point_2m
    out["wind"] = sj.wind_speed_10m
    out["solar"] = sj.shortwave_radiation
    out["cloud"] = sj.cloud_cover
    out["snow"] = sj.snowfall
    for site, key in [("gander", "T_gander"), ("cornerbrook", "T_cb")]:
        out[key] = load_weather(site).temperature_2m.reindex(out.index)
    return out


def features(w: pd.DataFrame, years_since_epoch: float | None = None) -> pd.DataFrame:
    """Feature table for every hour in w. If years_since_epoch is given, the trend is frozen at that value."""
    f = pd.DataFrame(index=w.index)
    f["T"] = w["T"]
    for k in (1, 3, 6, 12, 24):
        f[f"T_lag{k}"] = w["T"].shift(k)
    f["T_mean24"] = w["T"].rolling(24, min_periods=1).mean()
    f["T_mean72"] = w["T"].rolling(72, min_periods=1).mean()
    f["Td"], f["wind"], f["solar"], f["cloud"], f["snow"] = w.Td, w.wind, w.solar, w.cloud, w.snow
    f["T_gander"], f["T_cb"] = w.T_gander, w.T_cb
    f["T_island"] = 0.55 * w["T"] + 0.25 * w.T_gander + 0.20 * w.T_cb       # rough load-weighted mix
    f["hdd"] = np.maximum(T_BASE - f.T_island, 0)
    f["hdd24"] = np.maximum(T_BASE - (0.55 * f.T_mean24 + 0.45 * f.T_island), 0)
    f["hdd_wind"] = f.hdd * w.wind / 30.0                                    # infiltration / wind chill
    loc = w.index.tz_convert(TZ)
    f["hour"] = loc.hour
    f["dow"] = loc.dayofweek
    hol = nl_holidays(range(loc.year.min(), loc.year.max() + 1))
    f["holiday"] = pd.Index(loc.tz_localize(None).normalize()).isin(hol).astype(int)
    f["offday"] = ((f.dow >= 5) | (f.holiday == 1)).astype(int)
    doy = loc.dayofyear
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    yrs = (w.index - EPOCH).total_seconds() / (365.25 * 86400)
    f["trend"] = years_since_epoch if years_since_epoch is not None else yrs
    return f


BASE_COLS = ["hdd", "hdd24", "hdd_wind", "trend", "offday", "doy_sin", "doy_cos"]
GBM_COLS = ["T", "T_lag1", "T_lag3", "T_lag6", "T_lag12", "T_lag24", "T_mean24", "T_mean72", "Td", "wind", "solar",
            "cloud", "snow", "T_gander", "T_cb", "hour", "dow", "holiday", "doy_sin", "doy_cos"]


def _base_matrix(f: pd.DataFrame) -> np.ndarray:
    hours = np.eye(24)[f.hour.values]                    # hour-of-day profile (local time)
    off_hours = hours * f.offday.values[:, None]
    hdd_hours = hours[:, [6, 7, 8, 9, 16, 17, 18, 19]] * f.hdd.values[:, None]   # morning/evening heating peaks
    return np.column_stack([f[BASE_COLS].values, hours, off_hours, hdd_hours])


class LinearDegreeHour:
    """Baseline: ridge regression on heating degree-hours, wind interaction, trend and hourly profile."""

    def __init__(self):
        self.m = Ridge(alpha=1.0)

    def fit(self, f, y):
        self.m.fit(_base_matrix(f), y.values)
        self.resid_ = y.values - self.predict(f)
        return self

    def predict(self, f):
        return self.m.predict(_base_matrix(f))


T_COLS = ["T", "T_lag1", "T_lag3", "T_lag6", "T_lag12", "T_lag24", "T_mean24", "T_mean72", "Td", "T_gander", "T_cb"]


class DemandModel:
    """Gradient boosting on weather + calendar, with a linear physics tail below the training temperatures."""

    def __init__(self, seed: int = SEED, tail: bool = True):
        self.tail, self.seed = tail, seed
        self.cols = GBM_COLS + ["trend"]
        self.gbm = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                                 min_samples_leaf=40, l2_regularization=1.0, random_state=seed)

    def fit(self, f: pd.DataFrame, y: pd.Series):
        self.gbm.fit(f[self.cols].values, y.values)
        self.t_min_ = {c: float(f[c].min()) for c in T_COLS}
        self.ti_min_ = float(f.T_island.min())
        cold = f.T_island <= f.T_island.quantile(0.05)          # slope from the coldest 5% of training hours
        x = np.column_stack([f.T_island[cold], f.wind[cold], np.ones(cold.sum())])
        coef, *_ = np.linalg.lstsq(x, y[cold].values, rcond=None)
        self.slope_ = float(min(coef[0], 0.0))                  # MW per degree C (negative: colder -> more)
        self.resid_ = y.values - self.predict(f)
        return self

    def predict(self, f: pd.DataFrame) -> np.ndarray:
        g = f[self.cols].copy()
        if self.tail:
            for c in T_COLS:
                g[c] = np.maximum(g[c], self.t_min_[c])
        out = self.gbm.predict(g.values)
        if self.tail:
            out += self.slope_ * np.minimum(f.T_island.values - self.ti_min_, 0.0)
        return out


MODELS = {"linear degree-hour (baseline)": lambda: LinearDegreeHour(),
          "gradient boosting, no tail (baseline)": lambda: DemandModel(tail=False),
          "gradient boosting + physics tail (ours)": lambda: DemandModel(tail=True)}
OURS = "gradient boosting + physics tail (ours)"


def dataset() -> pd.DataFrame:
    d = load_demand()
    w = weather_frame()
    f = features(w)
    df = f.join(d[["demand_mw", "n_obs"]], how="inner").dropna()
    return df[df.n_obs >= 6]                              # at least half of the 5-minute readings present


# ---------------------------------------------------------------------------------------------- evaluation
def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    top = y >= np.quantile(y, 0.99)
    return dict(mae=float(np.mean(np.abs(y - p))), mape=float(np.mean(np.abs(y - p) / y) * 100),
                rmse=float(np.sqrt(np.mean((y - p) ** 2))),
                mae_top1pct=float(np.mean(np.abs(y[top] - p[top]))), bias_top1pct=float(np.mean(p[top] - y[top])),
                peak_actual=float(y.max()), peak_pred_at_peak_hour=float(p[np.argmax(y)]))


def daily_peak_metrics(idx, y, p) -> dict:
    s = pd.DataFrame({"y": y, "p": p}, index=idx.tz_convert(TZ))
    dp = s.resample("1D").max().dropna()
    return dict(daily_peak_mape=float(np.mean(np.abs(dp.y - dp.p) / dp.y) * 100))


COLD_SPLIT_C = -8.0


def train_and_evaluate() -> dict:
    df = dataset()
    tr = df[df.index < TRAIN_END]
    te = df[df.index >= TEST_START]
    y_tr, y_te = tr.demand_mw, te.demand_mw
    res, preds = {}, {}
    for name, make in MODELS.items():
        m = make().fit(tr, y_tr)
        preds[name] = p = m.predict(te)
        res[name] = dict(**metrics(y_te.values, p), **daily_peak_metrics(te.index, y_te.values, p))
    # extrapolation test: never see an hour colder than COLD_SPLIT_C, predict all colder hours
    pool = df[df.index < TRAIN_END]
    mild, cold = pool[pool.T_island > COLD_SPLIT_C], pool[pool.T_island <= COLD_SPLIT_C]
    extrap = {}
    for name, make in MODELS.items():
        m = make().fit(mild, mild.demand_mw)
        p = m.predict(cold)
        y = cold.demand_mw.values
        extrap[name] = dict(n_cold_hours=int(len(cold)), mae=float(np.mean(np.abs(y - p))), bias=float(np.mean(p - y)),
                            mape=float(np.mean(np.abs(y - p) / y) * 100))
    # stress test: hold out the 2022-23 winter (record peak), train on the other winters
    hold = (df.index >= pd.Timestamp("2022-11-01", tz="UTC")) & (df.index < pd.Timestamp("2023-05-01", tz="UTC"))
    stress = {}
    for name, make in MODELS.items():
        m = make().fit(df[~hold], df[~hold].demand_mw)
        p = m.predict(df[hold])
        y = df[hold].demand_mw.values
        k = int(np.argmax(y))
        stress[name] = dict(record_actual=float(y[k]), record_pred=float(p[k]), record_time=str(df[hold].index[k]),
                            top10_mae=float(np.mean(np.abs(y - p)[np.argsort(-y)[:10]])), **metrics(y, p))
    final = DemandModel().fit(df, df.demand_mw)
    joblib.dump(final, ROOT / "results" / "demand_model.joblib")
    te_pred = preds[OURS]
    top = y_te.values >= np.quantile(y_te.values, 0.95)
    out = dict(n_train_hours=int(len(tr)), n_test_hours=int(len(te)), test=res, extrapolation_below_minus8C=extrap,
               stress_holdout_2022_23=stress, cold_slope_mw_per_c=final.slope_, coldest_training_T_island=final.ti_min_,
               resid_sd_train=float(np.std(final.resid_)), test_resid_sd=float(np.std(y_te.values - te_pred)),
               test_peak_hour_resid_sd=float(np.std((y_te.values - te_pred)[top])),
               test_peak_hour_bias=float(np.mean((te_pred - y_te.values)[top])))
    (ROOT / "results" / "e1_demand_model.json").write_text(json.dumps(out, indent=1))
    pd.DataFrame({"actual": y_te, "ours": te_pred, "linear": preds["linear degree-hour (baseline)"],
                  "gbm_no_tail": preds["gradient boosting, no tail (baseline)"]}).round(1).to_csv(
        ROOT / "results" / "e1_test_predictions.csv")
    return out


def oos_residuals() -> pd.DataFrame:
    """Out-of-sample residuals (actual - predicted) for every hour, two ways:
    forward: train on all earlier seasons, predict the next season (one-year-ahead, as used in planning);
    lowo: leave one season out (includes the 2022-23 flash-freeze record; used as a stress case)."""
    df = dataset()
    loc = df.index.tz_convert(TZ)
    season = np.where(loc.month >= 7, loc.year, loc.year - 1)
    out = []
    for s in np.unique(season):
        test = season == s
        if test.sum() < 1000:
            continue
        for scheme, train in [("lowo", ~test), ("forward", season < s)]:
            if train.sum() < 4000:
                continue
            m = DemandModel().fit(df[train], df[train].demand_mw)
            out.append(pd.DataFrame({"actual": df.demand_mw[test], "pred": m.predict(df[test]), "season": s,
                                     "scheme": scheme}, index=df.index[test]))
    r = pd.concat(out)
    r["resid"] = r.actual - r.pred
    r.round(2).to_csv(ROOT / "results" / "e1_oos_residuals.csv")
    return r


def peak_residuals(scheme: str = "forward", min_pred_mw: float = 1350.0) -> np.ndarray:
    """Out-of-sample residuals in high-demand hours (predicted above min_pred_mw): the error that matters."""
    f = ROOT / "results" / "e1_oos_residuals.csv"
    r = pd.read_csv(f) if f.exists() else oos_residuals().reset_index()
    r = r[r.scheme == scheme]
    return r.resid[r.pred >= min_pred_mw].values


def load_model() -> DemandModel:
    return joblib.load(ROOT / "results" / "demand_model.joblib")


def weather_years(model: DemandModel | None = None, at_trend: float | None = None) -> pd.DataFrame:
    """Hourly demand the *current* island load would draw under every winter's weather since 1980."""
    model = model or load_model()
    w = weather_frame()
    if at_trend is None:
        at_trend = float((TEST_START + pd.Timedelta(days=60) - EPOCH).total_seconds() / (365.25 * 86400))
    f = features(w, years_since_epoch=at_trend).dropna()
    out = pd.DataFrame({"demand_mw": model.predict(f), "T": f["T"], "wind": f.wind, "T_island": f.T_island},
                       index=f.index)
    loc = out.index.tz_convert(TZ)
    out["winter"] = np.where(loc.month >= 7, loc.year, loc.year - 1)       # winter labelled by its start year
    return out
