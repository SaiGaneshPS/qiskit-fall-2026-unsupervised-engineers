"""Public inputs: island electricity demand (NL Hydro via Statistics Canada) and hourly weather (ERA5 via Open-Meteo).

Demand: the Canadian Centre for Energy Information's High-Frequency Electricity Data (HFED) archives the
"Current Island Demand" that Newfoundland and Labrador Hydro publishes for the Island Interconnected System
(about every 5 minutes, since 13 Sep 2022). We average it to hourly MW and drop obvious scrape errors.

Weather: Open-Meteo historical archive (ERA5 reanalysis), hourly, 1980 onward, for St. John's (the largest
load centre), Gander and Corner Brook (central and western island), and the Fermeuse wind farm site.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
UA = {"User-Agent": "qiskit-fallfest-2026-lightson/1.0 (research)"}

HFED_URL = "https://api.statcan.gc.ca/hfed-dehf/sdmx/rest/data/CCEI,DF_HFED_NL,1.0/all"
DEMAND_FILE = RAW / "nl_island_demand_hourly.csv"

SITES = {
    "stjohns": (47.5615, -52.7126),       # downtown St. John's
    "gander": (48.9569, -54.6089),
    "cornerbrook": (48.9490, -57.9503),
    "fermeuse": (46.9780, -52.9550),      # Fermeuse wind farm, Southern Shore of the Avalon
    "stlawrence": (46.9200, -55.3900),    # St. Lawrence wind farm, Burin Peninsula
    "portauport": (48.5500, -58.7500),    # Port au Port / Stephenville area, west coast wind projects
}
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "wind_speed_10m", "wind_direction_10m",
                "shortwave_radiation", "cloud_cover", "precipitation", "snowfall", "wind_speed_100m"]


# ---------------------------------------------------------------------------------------------- demand
def download_demand(start: str = "2022-01-01") -> pd.DataFrame:
    """Download the HFED Newfoundland demand series and save hourly means (UTC) to data/raw."""
    hdr = dict(UA, Accept="application/vnd.sdmx.data+csv;version=1.0.0")
    r = requests.get(HFED_URL, params={"startPeriod": start}, headers=hdr, timeout=600)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), usecols=["TIME_PERIOD", "OBS_VALUE"])
    df["t"] = pd.to_datetime(df.TIME_PERIOD)                 # TIME_PERIOD is UTC (DATETIME_LOCAL is NT)
    s = df.drop_duplicates("t").set_index("t").OBS_VALUE.sort_index().astype(float)
    s = clean_demand(s)
    h = s.resample("1h").agg(["mean", "count"]).rename(columns={"mean": "demand_mw", "count": "n_obs"})
    h = h[h.n_obs > 0]
    h.index.name = "time_utc"
    RAW.mkdir(parents=True, exist_ok=True)
    h.round(2).to_csv(DEMAND_FILE)
    return h


def clean_demand(s: pd.Series) -> pd.Series:
    """Remove scrape errors: non-positive values and isolated spikes far from the local median."""
    s = s[s > 100]
    med = s.rolling("3h", center=True, min_periods=3).median()
    return s[(s - med).abs() < 250]


def load_demand() -> pd.DataFrame:
    if not DEMAND_FILE.exists():
        download_demand()
    d = pd.read_csv(DEMAND_FILE, parse_dates=["time_utc"]).set_index("time_utc")
    d.index = d.index.tz_localize("UTC") if d.index.tz is None else d.index
    return d


# ---------------------------------------------------------------------------------------------- weather
def _weather_file(site: str) -> Path:
    return RAW / f"era5_{site}_hourly.csv.gz"


def download_weather(site: str, start: str = "1980-01-01", end: str | None = None,
                     variables: list[str] | None = None) -> pd.DataFrame:
    lat, lon = SITES[site]
    end = end or (pd.Timestamp.utcnow() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    frames = []
    # chunk by decade to keep each request small
    for a in pd.date_range(start, end, freq="10YS").union([pd.Timestamp(start)]):
        b = min(pd.Timestamp(end), a + pd.DateOffset(years=10) - pd.Timedelta(days=1))
        for attempt in range(5):
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", headers=UA, timeout=300, params=dict(
                latitude=lat, longitude=lon, start_date=a.strftime("%Y-%m-%d"), end_date=b.strftime("%Y-%m-%d"),
                hourly=",".join(variables or WEATHER_VARS), timezone="UTC", wind_speed_unit="kmh"))
            if r.status_code == 200:
                break
            time.sleep(20 * (attempt + 1))
        r.raise_for_status()
        h = r.json()["hourly"]
        frames.append(pd.DataFrame(h))
    w = pd.concat(frames).drop_duplicates("time")
    w["time"] = pd.to_datetime(w["time"])
    w = w.set_index("time").sort_index()
    w.index.name = "time_utc"
    RAW.mkdir(parents=True, exist_ok=True)
    w.to_csv(_weather_file(site), float_format="%.2f")
    return w


def load_weather(site: str = "stjohns") -> pd.DataFrame:
    f = _weather_file(site)
    if not f.exists():
        download_weather(site)
    w = pd.read_csv(f, parse_dates=["time_utc"]).set_index("time_utc")
    w.index = w.index.tz_localize("UTC")
    return w.dropna(how="all")


# ---------------------------------------------------------------------------------------------- calendar
def nl_holidays(years) -> set:
    """Province-wide statutory holidays observed by most employers in Newfoundland and Labrador."""
    out = set()
    for y in years:
        easter = _easter(y)
        out |= {pd.Timestamp(y, 1, 1), easter - pd.Timedelta(days=2),        # New Year, Good Friday
                _last_weekday_before(pd.Timestamp(y, 5, 25), 0),               # Victoria Day (Mon before 25 May)
                pd.Timestamp(y, 7, 1),                                         # Canada Day / Memorial Day
                _nth_weekday(y, 9, 0, 1),                                      # Labour Day
                _nth_weekday(y, 10, 0, 2),                                     # Thanksgiving
                pd.Timestamp(y, 11, 11), pd.Timestamp(y, 12, 25), pd.Timestamp(y, 12, 26)}
    return {d.normalize() for d in out}


def _easter(y: int) -> pd.Timestamp:
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = (h + l_ - 7 * m + 114) % 31 + 1
    return pd.Timestamp(y, month, day)


def _nth_weekday(y, month, weekday, n):
    d = pd.Timestamp(y, month, 1)
    d += pd.Timedelta(days=(weekday - d.weekday()) % 7)
    return d + pd.Timedelta(weeks=n - 1)


def _last_weekday_before(d, weekday):
    d -= pd.Timedelta(days=1)
    return d - pd.Timedelta(days=(d.weekday() - weekday) % 7)


if __name__ == "__main__":
    h = download_demand()
    print("demand hours:", len(h), h.index.min(), h.index.max(), "max", h.demand_mw.max())
    for s in SITES:
        w = download_weather(s)
        print(s, w.shape, w.index.min(), w.index.max())
