"""Turn one prediction into plain-language reasons (SHAP) and suggested actions."""
import calendar
import pickle

import numpy as np
import pandas as pd

from train import MODELS, STATUS_CUTS

LEVEL = {"port": "the port", "term": "this terminal", "pc": "this port (same cargo type)",
         "tc": "this terminal (same cargo type)"}
STATUS = ["normal", "slightly disrupted", "heavily disrupted"]

TEMPLATES = {
    "queue": "{n} ships already waiting at {lvl}",
    "occupancy": "{n} ships currently at berth at {lvl}",
    "queue_7d_ago": "{n} ships were waiting at {lvl} a week earlier",
    "queue_trend": "queue at {lvl} changed by {s} ships over 7 days",
    "arrivals_7d": "{n} arrivals at {lvl} in the last 7 days",
    "berths": "{lvl} has {n} active berths",
    "wait_mean_7d": "ships at {lvl} waited {h} on average (last 7 days)",
    "wait_mean_30d": "ships at {lvl} waited {h} on average (last 30 days)",
    "delayed_share_30d": "{pct} of ships at {lvl} were delayed in the last 30 days",
    "berth_h_mean_30d": "a ship stays {h} at berth at {lvl} on average",
    "wait_trend": "waits at {lvl} are {s}h vs the 30-day average",
    "queue_elapsed_mean_h": "ships queued at {lvl} have already waited {h} on average",
    "queue_tonnes": "{t} of cargo queued ahead at {lvl}",
    "wait_last5": "the last 5 ships to berth at {lvl} waited {h} on average",
    "occ_elapsed_mean_h": "ships at berth at {lvl} have been there {h} on average",
    "occ_remaining_h": "next berth at {lvl} expected to free up in about {h}",
    "rho": "{lvl} is running at {pct} of berth capacity (queue maths)",
    "erlang_wait_h": "queue maths predicts about {h} wait at {lvl}",
    "overloaded": "{lvl} is overloaded: arrivals exceed berth capacity (queue maths)",
    "clear_time_h": "clearing the queue ahead at {lvl} takes about {h} (queue maths)",
    "work_clear_h": "the cargo queued ahead at {lvl} needs about {h} to handle",
    "near_wait_mean_7d": "nearby ports are averaging {h} waits",
    "ship_wait_mean_1y": "this ship waited {h} on average on its calls in the past year",
    "ship_visits_1y": "this ship made {n} calls in the past year",
    "prev_days_since_departure": "this ship left its previous Brazilian port {n} days ago",
    "prev_wait_h": "at its previous port this ship waited {h}",
    "prev_same_port": "{same}",
    "prev_sail_days": "its last leg took {n} days",
    "prev_route_typical_days": "this route usually takes {n} days",
    "prev_early_days": "{early}",
    "prev3_wait_mean": "over its last 3 voyages this ship waited {h} on average",
    "prev_port_code": "the port it is coming from",
    "log_tonnes": "cargo of {t}",
    "teu": "{n} TEU of containers",
    "owner_foreign": "{owner} operator",
    "month": "arrival in {month}",
    "dow": "arrival on a {day}",
    "hour": "arrival at {n}:00",
    "precip_3d": "{n} mm of rain over the last 3 days",
    "wind_max_3d": "max wind {n} km/h over the last 3 days",
    "gust_max_3d": "gusts up to {n} km/h over the last 3 days",
    "port": "port: {v}",
    "terminal": "terminal: {v}",
    "nav_type": "{v} voyage",
    "cargo_nature": "cargo type: {v}",
}


def _hours(x):
    return f"{x / 24:.1f} days" if x >= 48 else f"{x:.0f}h"


def reason_text(feature, value):
    lvl, base = "", feature
    for p, name in LEVEL.items():
        if feature.startswith(p + "_"):
            lvl, base = name, feature[len(p) + 1:]
    tpl = TEMPLATES.get(base)
    if tpl is None or (not isinstance(value, str) and pd.isna(value)):
        return f"{feature} = {value}"
    v = value
    num = float(v) if not isinstance(v, str) else np.nan
    text = tpl.format(
        lvl=lvl, v=v, n=f"{num:.0f}" if np.isfinite(num) else v, s=f"{num:+.0f}" if np.isfinite(num) else v,
        h=_hours(num) if np.isfinite(num) else v, pct=f"{num:.0%}" if np.isfinite(num) else v,
        t=f"{(np.expm1(num) if base == 'log_tonnes' else num):,.0f} t" if np.isfinite(num) else v,
        owner="foreign" if num == 1 else "Brazilian",
        same="its previous call was at this same port" if num == 1 else "it is coming from a different port",
        early=(f"it arrives {num:.1f} days earlier than usual for this route" if num > 0.5 else
               f"it arrives {-num:.1f} days later than usual for this route" if num < -0.5 else "it arrives on a normal schedule"),
        month=calendar.month_name[int(num)] if base == "month" else "",
        day=calendar.day_name[int(num)] if base == "dow" else "")
    return text[0].upper() + text[1:]


def actions(prob, thr, lo, mid, hi, horizon_h, nearby_green=False):
    rng = f"{_hours(lo)}–{_hours(hi)}"
    if prob >= 0.7:
        if horizon_h >= 24:
            out = [f"Slow steam: the ship will likely wait {rng} anyway, so arriving later saves fuel at no time cost.",
                   f"Warn the customer now: expected delay about {_hours(mid)}."]
            if nearby_green:
                out.append("A nearby port is running normally: compare the cost of diverting.")
            return out
        return [f"Warn the customer: delivery likely late by about {_hours(mid)} ({rng}).",
                f"Re-book trucks/rail and the warehouse slot for about {_hours(mid)} from now.",
                f"Budget for demurrage of about {_hours(mid)}."]
    if prob >= thr:
        return [f"Medium risk: monitor the queue; tell the customer a delay of up to {_hours(hi)} is possible.",
                "Keep landside bookings flexible."]
    return ["Low risk: proceed as planned."]


def load_monthly(h, month):
    """Production bundle for ships arriving in `month` ("YYYY-MM"): retrained on data known on the 1st of that month."""
    import json
    d = MODELS / "monthly"
    with open(d / f"h{h}_{month}.pkl", "rb") as f:
        b = pickle.load(f)
    b.update(json.loads((d / f"h{h}_meta.json").read_text(encoding="utf-8")))
    b["horizon_h"] = h
    return b


def prepare(b, X):
    X = X[b["features"]].copy()
    for c, cats in b["categories"].items():
        X[c] = pd.Categorical(X[c].astype(object), categories=cats)
    return X


def shap_values(b, X):
    """Exact TreeSHAP values (log-odds) and the base value, from LightGBM itself: identical to shap.TreeExplainer,"""
    c = b["clf"].predict(X, pred_contrib=True)
    return c[:, :-1], c[:, -1]


def explain_row(b, X_row, top=4):
    X = prepare(b, X_row)
    raw = b["clf"].predict_proba(X)[:, 1]
    cal = (lambda v: b["iso"].predict(v)) if b.get("iso") is not None else (lambda v: np.asarray(v, dtype=float))
    prob = float(cal(raw)[0])
    q = b["q"]
    lo = float(np.expm1(b["qm"][0.1].predict(X)[0] - q).clip(0))
    mid = float(np.expm1(b["qm"][0.5].predict(X)[0]))
    hi = float(np.expm1(b["qm"][0.9].predict(X)[0] + q))
    lo, mid, hi = sorted(max(0.0, v) for v in (lo, mid, hi))   # waits are never negative

    sv, base = shap_values(b, X)
    sv = sv.reshape(-1)
    # spread the gap between the base rate and this prediction over features, in proportion to SHAP
    base_p = float(cal([1 / (1 + np.exp(-base[0]))])[0])
    # (approximate: SHAP is additive in log-odds, so this keeps signs and relative sizes, and sums to the gap)
    total = sv.sum()
    pp = sv / total * (prob - base_p) if abs(total) > 1e-6 else np.zeros_like(sv)
    order = np.argsort(-np.abs(sv))
    reasons = [{"text": reason_text(b["features"][i], X.iloc[0, i]), "effect_pp": float(pp[i] * 100),
                "cause": cause_of(b["features"][i])} for i in order[:top]]
    by_cause = pd.Series(pp * 100, index=[cause_of(f) for f in b["features"]]).groupby(level=0).sum()
    causes = [{"cause": c, "effect_pp": float(v)} for c, v in by_cause.sort_values(key=abs, ascending=False).items()]

    wait7 = X_row["port_wait_mean_7d"].iloc[0]
    status = STATUS[int(np.digitize(0 if pd.isna(wait7) else wait7, STATUS_CUTS))]
    near = X_row["near_wait_mean_7d"].iloc[0]
    return {"prob": prob, "delayed": prob >= b["thr"], "range_h": (lo, mid, hi), "status": status,
            "base_rate": base_p, "reasons": reasons, "causes": causes,
            "actions": actions(prob, b["thr"], lo, mid, hi, b["horizon_h"],
                               nearby_green=bool(pd.notna(near) and near < STATUS_CUTS[0]))}


# plain-language causes: every feature belongs to exactly one
CAUSES = {
    "line": "Line of ships at the terminal",
    "berths": "Busy berths / tight capacity",
    "slow": "Terminal has been slow lately",
    "port": "Port-wide congestion",
    "ship": "This ship's track record",
    "cargo": "Cargo type and size",
    "weather": "Weather",
    "calendar": "Time of year / week",
    "identity": "This terminal's usual pattern",
}
_LOCAL = {  # terminal / same-cargo level features
    "line": ("queue", "queue_tonnes", "queue_elapsed_mean_h", "clear_time_h", "work_clear_h"),   # the line RIGHT NOW only
    "berths": ("occupancy", "occ_elapsed_mean_h", "occ_remaining_h", "berths", "rho", "erlang_wait_h", "overloaded"),
    "slow": ("wait_mean_7d", "wait_mean_30d", "delayed_share_30d", "wait_last5", "wait_trend", "berth_h_mean_30d", "arrivals_7d",
             "queue_7d_ago", "queue_trend"),                                                       # recent history
}
_OTHER = {"ship_wait_mean_1y": "ship", "ship_visits_1y": "ship", "prev_days_since_departure": "ship",
          "prev_wait_h": "ship", "prev_same_port": "ship", "prev_sail_days": "ship", "prev_route_typical_days": "ship",
          "prev_early_days": "ship", "prev3_wait_mean": "ship", "prev_port_code": "ship", "near_wait_mean_7d": "port",
          "precip_3d": "weather", "wind_max_3d": "weather", "gust_max_3d": "weather",
          "month": "calendar", "dow": "calendar", "hour": "calendar", "port": "identity", "terminal": "identity"}


def cause_of(f):
    level, _, base = f.partition("_")
    if level == "port" and base:
        return CAUSES["port"]                       # whole-port state = port-wide congestion
    if level in ("term", "pc", "tc"):
        for k, bases in _LOCAL.items():
            if base in bases:
                return CAUSES[k]
    return CAUSES[_OTHER.get(f, "cargo")]


# situation check: facts now vs what is normal at this terminal (training years only)
FACTS = [  # (label, column, unit, higher_is_worse)
    ("Ships in line at this terminal", "term_queue", "ships", True),
    ("Ships at berth (of working berths)", "term_occupancy", "ships", True),
    ("Next berth expected free in", "term_occ_remaining_h", "h", True),
    ("Avg wait at this terminal, last 7 days", "term_wait_mean_7d", "h", True),
    ("Last 5 ships at this terminal waited", "term_wait_last5", "h", True),
    ("Ships in line, whole port", "port_queue", "ships", True),
    ("Avg wait at nearby ports, last 7 days", "near_wait_mean_7d", "h", True),
    ("This ship's avg wait, past year", "ship_wait_mean_1y", "h", True),
    ("Rain, last 3 days", "precip_3d", "mm", True),
]


def terminal_norms(features, end="2022-12-31"):
    """Median of each fact per terminal over the training years (no 2023-24 info)."""
    tr = features[features.arrival <= end + " 23:59"]
    cols = [c for _, c, _, _ in FACTS]
    return tr.groupby(tr.terminal.astype(str), observed=True)[cols].median()


def _fmt(v, unit):
    if pd.isna(v):
        return "—"
    if unit == "h":
        return _hours(max(float(v), 0))
    return f"{v:.0f} {unit}" if unit != "ships" else f"{v:.0f}"


def situation(row, norms):
    """List of dicts: label, now, normal, flag (worse/better/usual)."""
    r = row.iloc[0]
    term = str(r["terminal"])
    out = []
    for label, col, unit, worse_up in FACTS:
        now = r[col]
        normal = norms.loc[term, col] if term in norms.index else np.nan
        flag = "usual"
        if pd.notna(now) and pd.notna(normal):
            gap = max(1.0, 0.1 * abs(normal)) if unit != "ships" else 1.0
            if now > normal * 1.5 and now - normal >= gap:
                flag = "worse" if worse_up else "better"
            elif now < normal / 1.5 and normal - now >= gap:
                flag = "better" if worse_up else "worse"
        now_s = _fmt(now, unit)
        if col == "term_occupancy" and pd.notna(r["term_berths"]):
            now_s = f"{now:.0f} of {r['term_berths']:.0f}"
            if now >= r["term_berths"]:
                flag = "worse"
        out.append({"fact": label, "now": now_s, "normal": _fmt(normal, unit), "flag": flag})
    return out


def free_berth_norms(features, end="2022-12-31"):
    """Per terminal: median recorded wait when ships arrived to an EMPTY line with a FREE berth (training years)."""
    tr = features[(features.arrival <= end + " 23:59") & (features.term_queue == 0)
                  & (features.term_occupancy < features.term_berths)]
    return tr.groupby(tr.terminal.astype(str), observed=True).wait_h.median()


def queue_caps(features, end="2022-12-31", ceiling=25):
    """Per terminal: slider maximum for "ships waiting" = 1.2 x the terminal's busy level (95th percentile of the"""
    tr = features[features.arrival <= end + " 23:59"]
    q = tr.groupby(tr.terminal.astype(str), observed=True).term_queue.quantile(0.95)
    return (q * 1.2).round().clip(lower=15, upper=ceiling).astype(int)   # at least 15 so "what if it gets busy" can be explored


def displaced(row, berths):
    """Ships sitting at berths that get closed: free berths close first; beyond that, ships must leave and wait."""
    occ0 = float(np.nan_to_num(row.term_occupancy.iloc[0]))
    c0 = float(np.nan_to_num(row.term_berths.iloc[0], nan=berths))
    # only ships at the usual berths count (an extra ship squeezed in is not displaced)
    return int(max(0.0, min(occ0, c0) - berths))


def apply_scenario(row, queue, berths):
    """What-if: set the terminal's line length and number of working berths; keep every related clue consistent."""
    from features import erlang_c_wait_h
    q0 = float(row.term_queue.iloc[0])
    c0 = float(np.nan_to_num(row.term_berths.iloc[0], nan=berths))
    if queue == q0 and berths == c0:
        return row.copy()
    row = row.copy()
    moved = displaced(row, berths)
    q = queue + moved
    bh = row.term_berth_h_mean_30d.values
    with np.errstate(invalid="ignore", divide="ignore"):
        if q0 > 0:
            r = q / q0
            row["term_queue_tonnes"] = row.term_queue_tonnes.values * r
            row["term_work_clear_h"] = row.term_work_clear_h.values * r * (c0 / berths)
            row["term_clear_time_h"] = row.term_clear_time_h.values * r * (c0 / berths)
        else:
            row["term_queue_tonnes"] = q * np.expm1(row.log_tonnes.values)
            row["term_clear_time_h"] = q * bh / berths
            row["term_work_clear_h"] = row["term_clear_time_h"].values
            row["term_queue_elapsed_mean_h"] = 0.0
        if q == 0:
            row["term_queue_elapsed_mean_h"] = 0.0
        row["term_queue_trend"] = row.term_queue_trend.values + (q - q0)
        row["term_queue"] = q
        if berths != c0:
            lam = row.term_arrivals_7d.values / 7
            row["term_rho"] = row.term_rho.values * (c0 / berths)
            row["term_erlang_wait_h"], row["term_overloaded"] = erlang_c_wait_h(lam, 24 / bh, [berths])
            row["term_berths"] = berths
            row["term_occupancy"] = np.minimum(row.term_occupancy.values, berths)

    # lines that contain this terminal's ships move with it
    delta = q - q0
    tc0 = float(np.nan_to_num(row.get("tc_queue", pd.Series([0.0])).iloc[0]))
    tc_new = float(np.clip(tc0 + delta, 0, q))
    for level in ("tc", "pc", "port"):
        col = f"{level}_queue"
        if col not in row:
            continue
        old = float(np.nan_to_num(row[col].iloc[0]))
        new = tc_new if level == "tc" else max(0.0, old + (tc_new - tc0 if level == "pc" else delta))
        if new == 0:
            row[col] = 0.0
            for c in (f"{level}_queue_tonnes", f"{level}_work_clear_h", f"{level}_clear_time_h", f"{level}_queue_elapsed_mean_h"):
                if c in row:
                    row[c] = 0.0
            if f"{level}_queue_trend" in row:
                row[f"{level}_queue_trend"] = row[f"{level}_queue_trend"].values + (new - old)
            continue
        if new == old:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            for c in (f"{level}_queue_tonnes", f"{level}_work_clear_h", f"{level}_clear_time_h"):
                if c in row:
                    row[c] = row[c].values * (new / old) if old > 0 else (row["term_" + c.split("_", 1)[1]].values if level == "tc" else row[c].values)
        row[col] = new
        if f"{level}_queue_trend" in row:
            row[f"{level}_queue_trend"] = row[f"{level}_queue_trend"].values + (new - old)
        if new == 0 and f"{level}_queue_elapsed_mean_h" in row:
            row[f"{level}_queue_elapsed_mean_h"] = 0.0
    return row


# what-if guard
def _raw_grid(b, row, qs, berths_list):
    """Risk and P10/P50/P90 hours for every (queue, berths) combination, in one batched call per model."""
    rows = pd.concat([apply_scenario(row, q, c) for c in berths_list for q in qs], ignore_index=True)
    X = prepare(b, rows)
    raw = b["clf"].predict_proba(X)[:, 1]
    p = b["iso"].predict(raw) if b.get("iso") is not None else raw
    qq = b["q"]
    lo = np.expm1(b["qm"][0.1].predict(X) - qq)
    mid = np.expm1(b["qm"][0.5].predict(X))
    hi = np.expm1(b["qm"][0.9].predict(X) + qq)
    shape = (len(berths_list), len(qs))
    return [np.maximum(a, 0).reshape(shape) for a in (p, lo, mid, hi)]


def _monotone(grid, i0):
    """Anchored at the real queue (column i0, row 0 = no berths closed): never decreasing in queue, never"""
    g = grid.copy()
    for k in range(g.shape[0]):                       # along the queue axis, anchored at i0
        g[k, i0:] = np.maximum.accumulate(g[k, i0:])
        g[k, :i0 + 1] = np.minimum.accumulate(g[k, :i0 + 1][::-1])[::-1]
    return np.maximum.accumulate(g, axis=0)           # along berths closed (row 0 = none closed)


def whatif_predict(b, row, queue, closed, qmax):
    """Prediction for a what-if, guarded so it can never contradict the slider direction."""
    q0 = int(row.term_queue.iloc[0])
    c0 = int(np.nan_to_num(row.term_berths.iloc[0], nan=1))
    qs = sorted(set(range(0, max(qmax, q0, queue) + 1)))
    berths_list = [c0 - k for k in range(0, closed + 1)]
    p, lo, mid, hi = (_monotone(g, qs.index(q0)) for g in _raw_grid(b, row, qs, berths_list))
    i, k = qs.index(queue), closed
    scen = apply_scenario(row, queue, c0 - closed)
    r = explain_row(b, scen)
    vals = sorted(float(v[k, i]) for v in (lo, mid, hi))
    r["prob"], r["range_h"] = float(p[k, i]), tuple(vals)
    r["delayed"] = r["prob"] >= b["thr"]
    near = scen["near_wait_mean_7d"].iloc[0]
    r["actions"] = actions(r["prob"], b["thr"], *vals, b["horizon_h"],
                           nearby_green=bool(pd.notna(near) and near < STATUS_CUTS[0]))
    # an empty line can never be shown as a reason for MORE delay
    if float(scen.term_queue.iloc[0]) == 0:
        r["causes"] = [dict(c, effect_pp=min(c["effect_pp"], 0.0)) if c["cause"] == CAUSES["line"] else c for c in r["causes"]]
        r["causes"].sort(key=lambda c: -abs(c["effect_pp"]))
        r["reasons"] = [x for x in r["reasons"] if not (x["cause"] == CAUSES["line"] and x["effect_pp"] > 0)]
    return r
