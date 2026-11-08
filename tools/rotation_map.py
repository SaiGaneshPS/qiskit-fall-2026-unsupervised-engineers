"""Map for E8: a good checkerboard rotation of St. John's neighbourhoods, with the hardware cluster outlined.

The whole-city plan is a low-energy state of the E8 energy (no fairness history), found by classical simulated
annealing; it is an illustration, not a quantum result. Writes figures/e8_rotation_map.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridrisk.rotation import IsingProblem, ising, load_city  # noqa: E402

HW = ["The Battery", "Fort William", "Bannerman", "Downtown", "Pennywell", "Rabbittown", "Ecclesiastical District", "Georgestown"]


def anneal(p, rng, sweeps=4000):
    x = rng.integers(0, 2, p.n).astype(np.int8)
    e = p.energy(x)
    for T in np.geomspace(3.0, 0.01, sweeps):
        i = rng.integers(p.n)
        y = x.copy(); y[i] ^= 1
        ey = p.energy(y)
        if ey <= e or rng.random() < np.exp(-(ey - e) / T):
            x, e = y, ey
    return x, e


def main():
    city = load_city()
    idx = [i for i in range(len(city.names)) if city.border_km[i].sum() > 0]
    h, J = ising(city, idx)
    p = IsingProblem(h, J)
    rng = np.random.default_rng(1)
    best = min((anneal(p, rng) for _ in range(12)), key=lambda r: r[1])[0]
    group = {city.names[i]: int(best[k]) for k, i in enumerate(idx)}
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.4), dpi=160, gridspec_kw=dict(width_ratios=[1.15, 1]))
    colours = {0: "#2a78d6", 1: "#f2b134"}
    hw_v = []
    for ax, zoom in zip(axs, (False, True)):
        for i, rings in enumerate(city.rings):
            name = city.names[i]
            for r in rings:
                if len(r) < 4:
                    continue
                v = np.array([(q[1], q[0]) for q in r])
                fc = colours[group[name]] if name in group else "#d9dde6"
                ax.add_patch(Polygon(v, closed=True, facecolor=fc, edgecolor="white", linewidth=0.8, alpha=0.9))
                if name in HW:
                    ax.add_patch(Polygon(v, closed=True, facecolor="none", edgecolor="#141a2e", linewidth=2.0))
                    if not zoom:
                        hw_v.append(v)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_aspect(1 / np.cos(np.radians(47.57)))
    axs[0].set_xlim(-52.80, -52.64); axs[0].set_ylim(47.505, 47.635)
    V = np.vstack(hw_v)
    pad = 0.002
    axs[1].set_xlim(V[:, 0].min() - pad, V[:, 0].max() + pad); axs[1].set_ylim(V[:, 1].min() - pad, V[:, 1].max() + pad)
    def label_point(i):
        """Area centroid of the largest outer ring (shoelace formula)."""
        rings = [np.array([(q[1], q[0]) for q in r]) for r in city.rings[i] if len(r) >= 4]
        v = max(rings, key=lambda a: abs(np.dot(a[:-1, 0], a[1:, 1]) - np.dot(a[1:, 0], a[:-1, 1])))
        x, y = v[:-1, 0], v[:-1, 1]
        x1, y1 = v[1:, 0], v[1:, 1]
        c = x * y1 - x1 * y
        A = c.sum() / 2
        return (y + y1) @ c / (6 * A), (x + x1) @ c / (6 * A)

    for name in HW:
        i = city.names.index(name)
        cy, cx = label_point(i)
        axs[1].text(cx, cy, name.replace(" District", "\nDistrict"), fontsize=8,
                    ha="center", va="center", color="#141a2e", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    axs[0].set_title("St. John's: one good checkerboard rotation", fontsize=11)
    axs[1].set_title("The 8 neighbourhoods run on ibm_quebec", fontsize=11)
    from matplotlib.patches import Patch
    axs[0].legend(handles=[Patch(color=colours[0], label="group A: dark in the first hour"),
                           Patch(color=colours[1], label="group B: dark in the second hour"),
                           Patch(facecolor="none", edgecolor="#141a2e", linewidth=2, label="run on ibm_quebec")],
                  loc="lower right", fontsize=8.5, frameon=True)
    fig.tight_layout()
    out = ROOT / "figures" / "e8_rotation_map.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    same = sum(city.border_km[a, b] for a in idx for b in idx if a < b and group[city.names[a]] == group[city.names[b]])
    tot = sum(city.border_km[a, b] for a in idx for b in idx if a < b)
    loadA = sum(city.load[i] for i in idx if group[city.names[i]] == 0) / city.load[idx].sum()
    print(f"wrote {out}; shared border between same-group neighbours {same / tot:.1%}; group A load share {loadA:.1%}")


if __name__ == "__main__":
    main()
