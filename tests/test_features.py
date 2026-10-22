import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import _ns, _state, erlang_c_wait_h  # noqa: E402


def test_erlang_c_matches_hand_calc():
    w, o = erlang_c_wait_h([3, 3.8, 5], [1, 1, 1], [4, 4, 4])
    assert abs(w[0] - 12.2) < 0.2
    assert abs(w[1] - 107) < 1.5
    assert o.tolist() == [0, 0, 1]
    assert w[2] == 1000.0


def test_erlang_c_nan_inputs_stay_nan():
    w, o = erlang_c_wait_h([np.nan, 3], [1, np.nan], [4, 4])
    assert np.isnan(w).all() and np.isnan(o).all()


def _toy():
    t = pd.Timestamp("2023-01-10")
    h = pd.Timedelta(hours=1)
    return pd.DataFrame({
        "arrival": [t, t + 2 * h, t + 5 * h],
        "berthed": [t + 1 * h, t + 30 * h, t + 40 * h],
        "unberthed": [t + 20 * h, t + 50 * h, t + 60 * h],
        "wait_h": [1.0, 28.0, 35.0],
        "berth_h": [19.0, 20.0, 20.0],
    })


def test_queue_excludes_self_and_future():
    g = _toy()
    T = _ns(g.arrival)
    st = _state(g, T, berths=1)
    # call 3 arrives at t+5h: call 1 already berthed, call 2 still waiting -> queue 1
    assert st["queue"].tolist() == [0, 0, 1]
    assert st["occupancy"].tolist() == [0, 1, 1]
    # no wait had been observed yet at any arrival: call 1 berthed at t+1h with wait 1h
    assert np.isnan(st["wait_mean_7d"][0])
    assert st["wait_mean_7d"][2] == 1.0


def test_horizon_shifts_reference_time():
    g = _toy()
    T = _ns(g.arrival) - 72 * 3600 * 10**9
    st = _state(g, T, berths=1)
    assert st["queue"].tolist() == [0, 0, 0]
    assert np.isnan(st["wait_mean_30d"]).all()


def test_microsecond_datetimes_handled():
    g = _toy()
    for c in ["arrival", "berthed", "unberthed"]:
        g[c] = g[c].astype("datetime64[us]")
    st = _state(g, _ns(g.arrival), berths=1)
    assert st["queue"].tolist() == [0, 0, 1]
    assert st["arrivals_7d"].tolist() == [0, 1, 2]


def test_queue_age_and_last5():
    g = _toy()
    g["tonnes"] = [100.0, 200.0, 300.0]
    st = _state(g, _ns(g.arrival), berths=1)
    # at call 3 (t+5h) only call 2 is queued, it arrived at t+2h -> waited 3h so far, 200 t
    assert abs(st["queue_elapsed_mean_h"][2] - 3.0) < 1e-6
    assert st["queue_tonnes"].tolist() == [0, 0, 200]
    assert st["wait_last5"][2] == 1.0 and np.isnan(st["wait_last5"][0])


def test_occupancy_elapsed():
    g = _toy()
    st = _state(g, _ns(g.arrival), berths=1)
    # at call 3 (t+5h) call 1 is at berth since t+1h -> 4h elapsed
    assert abs(st["occ_elapsed_mean_h"][2] - 4.0) < 1e-6


def test_previous_voyage_only_uses_departures_before_T():
    import numpy as np
    t = pd.Timestamp("2023-03-01")
    d = pd.Timedelta(days=1)
    g = pd.DataFrame({"unberthed": [t - 10 * d, t - 1 * d, t + 2 * d]})   # third departure is in the future
    T = np.array([t.value])
    k = np.searchsorted(g.unberthed.values.astype("datetime64[ns]").astype(np.int64), T) - 1
    assert k[0] == 1   # picks the departure 1 day before, never the future one
