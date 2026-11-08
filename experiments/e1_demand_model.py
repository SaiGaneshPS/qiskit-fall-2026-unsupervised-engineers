"""E1: the AI demand model (gradient boosting + physics cold tail) vs a linear degree-hour model and boosting
without the tail.

Test set: the 2025-26 winter and following months (Dec 2025 - Oct 2026), never seen in training.
Extrapolation test: train only on hours milder than -8 C (island-weighted), predict the colder hours.
Stress test: train without the 2022-23 winter and predict the island's record hour (4 Feb 2023).
Also writes the out-of-sample residuals (forward and leave-one-winter-out) used by the reliability engine.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridrisk import ml  # noqa: E402


def run():
    out = ml.train_and_evaluate()
    ml.oos_residuals()
    print(f"train hours {out['n_train_hours']:,}  test hours {out['n_test_hours']:,}")
    for name, r in out["test"].items():
        print(f"TEST  {name:36s} MAPE {r['mape']:5.2f}%  MAE {r['mae']:6.1f}  daily-peak MAPE {r['daily_peak_mape']:5.2f}%"
              f"  top-1% MAE {r['mae_top1pct']:6.1f} (bias {r['bias_top1pct']:+6.1f})")
    for name, r in out["extrapolation_below_minus8C"].items():
        print(f"EXTRAP {name:35s} {r['n_cold_hours']} hours colder than -8 C: MAE {r['mae']:6.1f}  bias {r['bias']:+6.1f}"
              f"  MAPE {r['mape']:5.2f}%")
    print(f"cold slope {out['cold_slope_mw_per_c']:.1f} MW per C below {out['coldest_training_T_island']:.1f} C")
    for name, r in out["stress_holdout_2022_23"].items():
        print(f"STRESS {name:35s} record {r['record_actual']:.0f} MW predicted {r['record_pred']:.0f}  "
              f"top-10 MAE {r['top10_mae']:.0f}  MAPE {r['mape']:.2f}%")
    print("residual sd: train", round(out["resid_sd_train"], 1), " test", round(out["test_resid_sd"], 1),
          " test top-5% hours", round(out["test_peak_hour_resid_sd"], 1))
    return out


if __name__ == "__main__":
    run()
