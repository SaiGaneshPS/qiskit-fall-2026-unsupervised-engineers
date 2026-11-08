"""Fair rotating outages for St. John's as an Ising problem on the real neighbourhood map (E8).

When supply runs short (DarkNL, January 2014: rolling 30-60 minute outages), neighbourhoods are split into two
groups that take turns in the dark. A "checkerboard" split keeps every dark neighbourhood next to a lit one
(warmth, traffic lights and phone service within reach) and spreads the cold-load surge when power returns.

    s_i = +1: neighbourhood i is in group A (dark first),  s_i = -1: group B
    E(s) = sum_{i<j} w_ij s_i s_j          neighbours prefer opposite groups (w = shared border length)
         + lam * (sum_i l_i s_i)^2         both groups shed about the same load (l = load share)
         + sum_i f_i s_i                   fairness: areas that have spent longer in the dark lean to group B

Data: OpenStreetMap neighbourhood boundaries (admin_level 9) and building footprints for St. John's
(data/raw/osm_stjohns_*.json, fetched with the Overpass API). Load is approximated by heated buildings
(houses count 1, apartment buildings 8, commercial 4; sheds and garages 0); OSM building coverage is uneven.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
BUILDING_WEIGHT = {"apartments": 8.0, "commercial": 4.0, "retail": 4.0, "office": 4.0, "hospital": 10.0,
                   "university": 6.0, "school": 3.0, "shed": 0.0, "garage": 0.0, "garages": 0.0, "greenhouse": 0.0,
                   "carport": 0.0, "roof": 0.0, "static_caravan": 0.0, "hut": 0.0, "boathouse": 0.0}


def _haversine_km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    x = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(2 * 6371.0 * np.arcsin(np.sqrt(x)))


def _rings(ways):
    """Join way geometries end to end into closed rings (lists of (lat, lon))."""
    segs = [[(p["lat"], p["lon"]) for p in w] for w in ways if len(w) > 1]
    rings = []
    while segs:
        ring = segs.pop()
        changed = True
        while changed and ring[0] != ring[-1]:
            changed = False
            for k, s in enumerate(segs):
                if s[0] == ring[-1]:
                    ring += s[1:]
                elif s[-1] == ring[-1]:
                    ring += s[::-1][1:]
                elif s[-1] == ring[0]:
                    ring = s[:-1] + ring
                elif s[0] == ring[0]:
                    ring = s[::-1][:-1] + ring
                else:
                    continue
                segs.pop(k)
                changed = True
                break
        rings.append(ring)
    return rings


@dataclass
class CityMap:
    names: list[str]
    centre: np.ndarray        # (n, 2) lat, lon
    border_km: np.ndarray     # (n, n) shared border length
    load: np.ndarray          # (n,) heated-building weight
    rings: list               # outer rings per neighbourhood, for maps

    def subregion(self, seed: str, k: int) -> list[int]:
        """k contiguous neighbourhoods grown breadth-first from `seed`, strongest borders first."""
        start = self.names.index(seed)
        chosen, queue = [start], deque([start])
        while queue and len(chosen) < k:
            i = queue.popleft()
            for j in np.argsort(-self.border_km[i]):
                if self.border_km[i, j] > 0 and j not in chosen and len(chosen) < k:
                    chosen.append(int(j)); queue.append(int(j))
        return chosen


def load_city() -> CityMap:
    d = json.loads((RAW / "osm_stjohns_districts.json").read_text())
    rels = [e for e in d["elements"] if e["tags"].get("admin_level") == "9"]
    names = [e["tags"]["name"] for e in rels]
    n = len(rels)
    way_len, owners = {}, defaultdict(set)
    rings = []
    for i, e in enumerate(rels):
        outer = []
        for m in e["members"]:
            if m["type"] == "way" and "geometry" in m:
                g = m["geometry"]
                way_len[m["ref"]] = sum(_haversine_km((g[k]["lat"], g[k]["lon"]), (g[k + 1]["lat"], g[k + 1]["lon"]))
                                        for k in range(len(g) - 1))
                owners[m["ref"]].add(i)
                if m.get("role", "outer") != "inner":
                    outer.append(g)
        rings.append(_rings(outer))
    B = np.zeros((n, n))
    for w, s in owners.items():
        s = sorted(s)
        for a in range(len(s)):
            for b in range(a + 1, len(s)):
                B[s[a], s[b]] += way_len[w]; B[s[b], s[a]] += way_len[w]
    # building load by point-in-polygon on the outer rings
    from matplotlib.path import Path as MPath
    pts = json.loads((RAW / "osm_stjohns_buildings.json").read_text())
    P = np.array([(p[1], p[0]) for p in pts])
    wts = np.array([BUILDING_WEIGHT.get(p[2], 1.0) for p in pts])
    load = np.zeros(n)
    centre = np.zeros((n, 2))
    for i, rs in enumerate(rings):
        inside = np.zeros(len(P), dtype=bool)
        allv = []
        for r in rs:
            if len(r) < 4:
                continue
            v = np.array([(q[1], q[0]) for q in r])
            inside |= MPath(v).contains_points(P)
            allv.append(v)
        load[i] = wts[inside].sum()
        v = np.vstack(allv) if allv else np.zeros((1, 2))
        centre[i] = (v[:, 1].mean(), v[:, 0].mean())
    return CityMap(names, centre, B, load, rings)


class IsingProblem:
    """Energy of 0/1 designs (bit 1 = group B, i.e. z = 1 - 2 bit), for the chain runners in gridrisk.qmcmc."""

    def __init__(self, h, J):
        self.h, self.J = np.asarray(h, float), np.triu(np.asarray(J, float), 1)
        self.n = len(h)

    def energy(self, X):
        Z = 1.0 - 2.0 * np.asarray(X, float)
        return Z @ self.h + np.einsum("...i,ij,...j->...", Z, self.J, Z)


def ising(city: CityMap, idx: list[int], lam: float = 0.15, fairness: np.ndarray | None = None):
    """(h, J) for the neighbourhoods `idx`. Borders are scaled to mean 1 over the region's links; the balance term
    uses load shares scaled to mean 1, so lam plays the same role as in the E8 'planar_bal' screen."""
    idx = list(idx)
    B = city.border_km[np.ix_(idx, idx)]
    links = B[np.triu_indices(len(idx), 1)]
    W = B / links[links > 0].mean()
    L = city.load[idx] / max(city.load[idx].mean(), 1e-9)
    J = np.triu(W, 1) + lam * np.triu(np.outer(L, L), 1) * 2
    h = np.zeros(len(idx)) if fairness is None else np.asarray(fairness, float)
    return h, J
