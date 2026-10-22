import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from explain import actions, reason_text  # noqa: E402


def test_reason_text_mentions_queue_level():
    assert reason_text("term_queue", 12.0) == "12 ships already waiting at this terminal"
    assert "95%" in reason_text("port_rho", 0.95) and "queue maths" in reason_text("port_rho", 0.95)
    assert reason_text("port_wait_mean_7d", 120) == "Ships at the port waited 5.0 days on average (last 7 days)"


def test_reason_text_handles_missing_and_unknown():
    assert reason_text("term_queue", float("nan")) == "term_queue = nan"
    assert reason_text("mystery", 3) == "mystery = 3"


def test_actions_depend_on_horizon_and_risk():
    early = actions(0.9, 0.25, 20, 40, 80, horizon_h=72, nearby_green=True)
    late = actions(0.9, 0.25, 20, 40, 80, horizon_h=0)
    assert any("Slow steam" in a for a in early) and any("diverting" in a for a in early)
    assert any("Warn the customer" in a for a in late) and not any("Slow steam" in a for a in late)
    assert actions(0.1, 0.25, 0, 2, 10, 0) == ["Low risk: proceed as planned."]
    assert actions(0.4, 0.25, 0, 20, 60, 0)[0].startswith("Medium risk")


def test_every_feature_has_one_plain_cause():
    from explain import CAUSES, cause_of
    from features import FEATURES
    assert cause_of("term_queue") == CAUSES["line"]
    assert cause_of("term_occupancy") == CAUSES["berths"]
    assert cause_of("pc_wait_last5") == CAUSES["slow"]
    assert cause_of("port_queue") == CAUSES["port"]
    assert cause_of("ship_wait_mean_1y") == CAUSES["ship"]
    assert cause_of("log_tonnes") == CAUSES["cargo"]
    assert all(cause_of(f) in CAUSES.values() for f in FEATURES)


def test_scenario_zero_queue_clears_related_clues():
    import pandas as pd
    from explain import apply_scenario
    row = pd.DataFrame({"term_queue": [4.0], "term_queue_tonnes": [200000.0], "term_work_clear_h": [100.0],
                        "term_queue_trend": [2.0], "term_queue_elapsed_mean_h": [40.0], "term_berths": [2.0], "term_occupancy": [2.0],
                        "term_berth_h_mean_30d": [48.0], "term_arrivals_7d": [7.0], "log_tonnes": [10.0],
                        "term_clear_time_h": [96.0], "term_rho": [0.5]})
    z = apply_scenario(row, queue=0, berths=2).iloc[0]
    assert z.term_queue_tonnes == 0 and z.term_work_clear_h == 0 and z.term_queue_elapsed_mean_h == 0
    assert z.term_clear_time_h == 0 and z.term_queue_trend == -2
    d = apply_scenario(row, queue=8, berths=2).iloc[0]
    assert d.term_queue_tonnes == 400000 and d.term_work_clear_h == 200 and d.term_clear_time_h == 192


def test_situation_flags_full_berths_and_bad_history():
    import pandas as pd
    from explain import FACTS, situation
    cols = [c for _, c, _, _ in FACTS]
    norms = pd.DataFrame([{c: 10.0 for c in cols}], index=["T1"])
    row = pd.DataFrame([{**{c: 10.0 for c in cols}, "terminal": "T1", "term_berths": 2.0,
                         "term_occupancy": 2.0, "term_wait_mean_7d": 70.0, "term_queue": 0.0}])
    facts = {f["fact"]: f for f in situation(row, norms)}
    assert facts["Ships at berth (of working berths)"]["flag"] == "worse"
    assert facts["Ships at berth (of working berths)"]["now"] == "2 of 2"
    assert facts["Avg wait at this terminal, last 7 days"]["flag"] == "worse"
    assert facts["Ships in line at this terminal"]["flag"] == "better"
    assert facts["Rain, last 3 days"]["flag"] == "usual"


def test_closing_berths_moves_no_ship_into_a_closed_berth():
    import pandas as pd
    from explain import apply_scenario
    row = pd.DataFrame({"term_queue": [2.0], "term_queue_tonnes": [1.0], "term_work_clear_h": [1.0],
                        "term_queue_trend": [0.0], "term_queue_elapsed_mean_h": [5.0], "term_berths": [4.0],
                        "term_occupancy": [3.0], "term_berth_h_mean_30d": [48.0], "term_arrivals_7d": [7.0],
                        "log_tonnes": [10.0], "term_clear_time_h": [24.0], "term_rho": [0.5]})
    r = apply_scenario(row, queue=2, berths=1).iloc[0]
    assert r.term_berths == 1 and r.term_occupancy == 1
    assert r.term_queue == 4


def test_queue_caps_per_terminal():
    import pandas as pd
    from explain import queue_caps
    f = pd.DataFrame({"arrival": pd.to_datetime(["2021-01-01"] * 4 + ["2024-01-01"]),
                      "terminal": ["A", "A", "B", "B", "A"], "term_queue": [2, 10, 0, 1, 99]})
    assert queue_caps(pd.DataFrame({"arrival": pd.to_datetime(["2021-01-01"] * 2), "terminal": ["Z", "Z"],
                                    "term_queue": [500, 600]}))["Z"] == 25   # hard ceiling
    c = queue_caps(f)
    assert c["B"] == 15                     # floor of 15
    assert 15 <= c["A"] <= 25               # ~1.2x its busy level, floor 15, capped, ignores the 2024 row


def test_scenario_moves_same_cargo_and_port_lines_too():
    import pandas as pd
    from explain import apply_scenario
    row = pd.DataFrame({"term_queue": [15.0], "term_queue_tonnes": [9e5], "term_work_clear_h": [300.0], "term_queue_trend": [0.0],
                        "term_queue_elapsed_mean_h": [400.0], "term_berths": [3.0], "term_occupancy": [0.0],
                        "term_berth_h_mean_30d": [60.0], "term_arrivals_7d": [7.0], "log_tonnes": [11.0],
                        "tc_queue": [15.0], "tc_clear_time_h": [300.0], "tc_queue_elapsed_mean_h": [400.0],
                        "pc_queue": [54.0], "pc_clear_time_h": [500.0], "port_queue": [63.0],
                        "term_clear_time_h": [300.0], "term_rho": [0.9]})
    z = apply_scenario(row, queue=0, berths=3).iloc[0]
    assert z.tc_queue == 0 and z.tc_clear_time_h == 0 and z.tc_queue_elapsed_mean_h == 0
    assert z.pc_queue == 39 and z.port_queue == 48   # the 15 terminal ships leave the port-wide lines too


def test_untouched_sliders_return_the_real_row():
    import pandas as pd
    from explain import apply_scenario
    row = pd.DataFrame({"term_queue": [3.0], "term_berths": [4.0], "term_occupancy": [2.0], "term_rho": [0.71],
                        "term_clear_time_h": [55.5], "term_erlang_wait_h": [9.9]})
    pd.testing.assert_frame_equal(apply_scenario(row, 3, 4), row)


def test_monotone_guard_keeps_anchor_and_direction():
    import numpy as np
    from explain import _monotone
    g = np.array([[0.5, 0.4, 0.6, 0.55], [0.45, 0.7, 0.5, 0.9]])   # row 0 = no berths closed, anchor at queue index 1
    m = _monotone(g, 1)
    assert m[0, 1] == 0.4                                        # the real situation is untouched
    assert (np.diff(m, axis=1) >= 0).all() and (np.diff(m, axis=0) >= 0).all()
