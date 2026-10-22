import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from train import conformal_q, cost_threshold, split  # noqa: E402


def test_split_is_time_ordered():
    df = pd.DataFrame({"arrival": pd.date_range("2020-06-01", "2024-12-31", freq="D")})
    s = split(df)
    assert s["train"].arrival.max() < s["val"].arrival.min()
    assert s["val"].arrival.max() < s["cal"].arrival.min()
    assert s["cal"].arrival.max() < s["test"].arrival.min()
    assert s["train"].arrival.min() >= pd.Timestamp("2020-03-01")  # Jan-Feb 2020 only warm up 30d history


def test_conformal_hits_target_coverage():
    rng = np.random.default_rng(0)
    y_cal, y_new = rng.normal(size=20000), rng.normal(size=20000)
    lo, hi = -0.5, 0.5  # deliberately too narrow
    q = conformal_q(np.full_like(y_cal, lo), np.full_like(y_cal, hi), y_cal)
    cov = np.mean((y_new >= lo - q) & (y_new <= hi + q))
    assert abs(cov - 0.80) < 0.03


def test_cost_threshold_prefers_recall():
    y = np.array([1] * 50 + [0] * 50)
    p = np.r_[np.linspace(0.3, 0.9, 50), np.linspace(0.1, 0.7, 50)]
    assert cost_threshold(p, y) < 0.5  # FN costs 3x -> flag earlier
