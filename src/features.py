"""Leak-free features per call, computed as of T = arrival - horizon."""
import numpy as np
import pandas as pd

from load import PROC
from prev_voyage import PREV_FEATURES, PREVE_FEATURES

DAY = np.timedelta64(1, "D").astype("timedelta64[ns]").astype(np.int64)
HOUR = DAY // 24


def _ns(s):
    """Datetime series -> int64 nanoseconds (pandas 3 may store datetimes as us)."""
    return s.values.astype("datetime64[ns]").astype(np.int64)


OVERLOAD_CAP_H = 1000.0

LEVELS = {"port": "port", "term": "terminal", "pc": "port_cargo", "tc": "term_cargo"}  # pc/tc = same cargo at port / terminal
QUEUE_MATHS = [f"{p}_{f}" for p in LEVELS for f in ["rho", "erlang_wait_h", "overloaded", "clear_time_h", "work_clear_h"]]
STATE = [f"{p}_{f}" for p in LEVELS for f in
         ["queue", "occupancy", "queue_7d_ago", "queue_trend", "arrivals_7d", "berths",
          "wait_mean_7d", "wait_mean_30d", "delayed_share_30d", "berth_h_mean_30d", "wait_trend",
          "queue_elapsed_mean_h", "queue_tonnes", "wait_last5", "occ_elapsed_mean_h", "occ_remaining_h"]]
OTHER = ["near_wait_mean_7d", "ship_wait_mean_1y", "ship_visits_1y", "log_tonnes", "teu", "owner_foreign", "month", "dow", "hour",
         "precip_3d", "wind_max_3d", "gust_max_3d"]
CAT_FEATURES = ["port", "terminal", "nav_type", "cargo_nature"]
VOYAGE = PREV_FEATURES + PREVE_FEATURES + ["prev_port_code"]   # the ship's previous voyages
FEATURES = STATE + QUEUE_MATHS + OTHER + VOYAGE + CAT_FEATURES


def erlang_c_wait_h(lam_per_day, mu_per_day, c):
    """M/M/c expected wait in queue (hours). Returns (wait_h, overloaded)."""
    lam, mu, c = (np.asarray(x, dtype=float) for x in (lam_per_day, mu_per_day, c))
    a = lam / mu
    overloaded = ~(a < c)  # also True when inputs are NaN... fixed below
    b = np.ones_like(a)
    for k in range(1, int(np.nanmax(np.where(np.isfinite(c), c, 1))) + 1):
        b = np.where(k <= c, a * b / (k + a * b), b)
    with np.errstate(divide="ignore", invalid="ignore"):
        pc = c * b / (c - a * (1 - b))
        wait_h = pc / (c * mu - lam) * 24
    wait_h = np.where(overloaded, OVERLOAD_CAP_H, np.minimum(wait_h, OVERLOAD_CAP_H))
    nan = ~(np.isfinite(a) & np.isfinite(c) & (c > 0))
    wait_h = np.where(nan, np.nan, wait_h)
    return wait_h, np.where(nan, np.nan, overloaded.astype(float))


def _count_before(sorted_t, t):
    return np.searchsorted(sorted_t, t, side="left")


def _window(sorted_t, cumsum, t, width):
    """Sum and count of events with time in [t - width, t)."""
    i1, i0 = _count_before(sorted_t, t), _count_before(sorted_t, t - width)
    return cumsum[i1] - cumsum[i0], i1 - i0


def _mean(sorted_t, vals, t, width):
    cs = np.concatenate([[0.0], np.cumsum(vals)])
    s, n = _window(sorted_t, cs, t, width)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _state(g, T, berths):
    """Port-state features for target times T (int ns) from one group's calls g."""
    arr = np.sort(_ns(g.arrival))
    ber_order = np.argsort(g.berthed.values)
    ber = _ns(g.berthed)[ber_order]
    unb_order = np.argsort(g.unberthed.values)
    unb = _ns(g.unberthed)[unb_order]
    wait_by_ber = g.wait_h.values[ber_order]
    berth_h_by_unb = g.berth_h.values[unb_order]

    def queue_at(t):
        return _count_before(arr, t) - _count_before(ber, t)

    q = queue_at(T)
    q7 = queue_at(T - 7 * DAY)
    out = {
        "queue": q,
        "occupancy": _count_before(ber, T) - _count_before(unb, T),
        "queue_7d_ago": q7,
        "queue_trend": q - q7,
        "arrivals_7d": _count_before(arr, T) - _count_before(arr, T - 7 * DAY),
        "berths": np.full(len(T), berths, dtype=float),
        "wait_mean_7d": _mean(ber, wait_by_ber, T, 7 * DAY),
        "wait_mean_30d": _mean(ber, wait_by_ber, T, 30 * DAY),
        "delayed_share_30d": _mean(ber, (wait_by_ber > 24).astype(float), T, 30 * DAY),
        "berth_h_mean_30d": _mean(unb, berth_h_by_unb, T, 30 * DAY),
    }
    out["wait_trend"] = out["wait_mean_7d"] - out["wait_mean_30d"]

    # who is in the queue right now: arrived before T and not yet berthed at T
    arr_order = np.argsort(g.arrival.values)
    hours = _ns(g.arrival) / HOUR
    tonnes = np.nan_to_num(g["tonnes"].values if "tonnes" in g else np.zeros(len(g)))

    def queue_sum(v):
        a = np.concatenate([[0.0], np.cumsum(v[arr_order])])
        b = np.concatenate([[0.0], np.cumsum(v[ber_order])])
        x = a[_count_before(arr, T)] - b[_count_before(ber, T)]
        return np.where(x < 1e-6, 0.0, x)   # an empty line has exactly zero (no float round-off)

    with np.errstate(invalid="ignore", divide="ignore"):
        out["queue_elapsed_mean_h"] = np.where(q > 0, (q * (T / HOUR) - queue_sum(hours)) / np.maximum(q, 1), 0.0)
        out["queue_tonnes"] = queue_sum(tonnes)
        tput = _mean(unb, tonnes[unb_order], T, 30 * DAY) * (_count_before(unb, T) - _count_before(unb, T - 30 * DAY)) / (30 * 24)
        out["work_clear_h"] = out["queue_tonnes"] / tput  # tonnes waiting / tonnes handled per hour
    # ships at berth now: how long they've been there vs a normal stay -> when a berth frees up
    occ = out["occupancy"]
    bh = _ns(g.berthed) / HOUR
    a = np.concatenate([[0.0], np.cumsum(bh[ber_order])])
    b = np.concatenate([[0.0], np.cumsum(bh[unb_order])])
    with np.errstate(invalid="ignore", divide="ignore"):
        occ_el = np.where(occ > 0, (occ * (T / HOUR) - (a[_count_before(ber, T)] - b[_count_before(unb, T)]))
                          / np.maximum(occ, 1), 0.0)
    out["occ_elapsed_mean_h"] = occ_el
    out["occ_remaining_h"] = out["berth_h_mean_30d"] - occ_el

    i1 = _count_before(ber, T)
    cs = np.concatenate([[0.0], np.cumsum(wait_by_ber)])
    n5 = np.minimum(i1, 5)
    out["wait_last5"] = np.where(n5 > 0, (cs[i1] - cs[i1 - n5]) / np.maximum(n5, 1), np.nan)

    # queue maths
    lam = out["arrivals_7d"] / 7
    mu = 24 / out["berth_h_mean_30d"]
    c = out["berths"]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["rho"] = lam / (c * mu)
        out["clear_time_h"] = q * out["berth_h_mean_30d"] / c  # time for c berths to clear the queue ahead
    out["erlang_wait_h"], out["overloaded"] = erlang_c_wait_h(lam, mu, c)
    return out


def static_berths(calls, end="2022-12-31", min_calls=20):
    """Berths per group = berths with >= min_calls in the training era (no test info)."""
    tr = calls[calls.arrival <= end]
    res = {}
    for name, key in LEVELS.items():
        cnt = tr.groupby([key, "berth_id"]).size()
        res[name] = cnt[cnt >= min_calls].groupby(level=0).size()
    return res


def _nearest_ports(calls, k=3):
    pos = calls.groupby("port")[["lat", "lon"]].median()
    lat, lon = np.radians(pos.lat.values), np.radians(pos.lon.values)
    d = np.arccos(np.clip(np.sin(lat[:, None]) * np.sin(lat) +
                          np.cos(lat[:, None]) * np.cos(lat) * np.cos(lon[:, None] - lon), -1, 1))
    np.fill_diagonal(d, np.inf)
    return {p: list(pos.index[np.argsort(d[i])[:k]]) for i, p in enumerate(pos.index)}


def build_features(calls, weather, horizon_h):
    calls = calls.sort_values("arrival").reset_index(drop=True)
    calls["port_cargo"] = calls.port.astype(str) + "|" + calls.cargo_nature.astype(str)
    calls["term_cargo"] = calls.terminal.astype(str) + "|" + calls.cargo_nature.astype(str)
    T_all = _ns(calls.arrival) - horizon_h * HOUR
    feats = pd.DataFrame(index=calls.index)
    berths = static_berths(calls)

    for name, key in LEVELS.items():
        cols = {}
        for gkey, g in calls.groupby(key):
            idx = g.index.values
            st = _state(g, T_all[idx], berths[name].get(gkey, np.nan))
            for f, v in st.items():
                cols.setdefault(f, np.full(len(calls), np.nan))[idx] = v
        for f, v in cols.items():
            feats[f"{name}_{f}"] = v

    # congestion at the 3 nearest ports, as of T
    near = _nearest_ports(calls)
    port_groups = {p: g for p, g in calls.groupby("port")}
    feats["near_wait_mean_7d"] = np.nan
    for p, g in port_groups.items():
        idx = g.index.values
        vals = []
        for q in near[p]:
            h = port_groups[q].sort_values("berthed")
            vals.append(_mean(_ns(h.berthed), h.wait_h.values, T_all[idx], 7 * DAY))
        feats.loc[idx, "near_wait_mean_7d"] = np.nanmean(vals, axis=0)

    # this vessel's own history (any port), waits known by T
    feats["ship_wait_mean_1y"], feats["ship_visits_1y"] = np.nan, np.nan
    for imo, g in calls.dropna(subset=["imo"]).groupby("imo"):
        idx = g.index.values
        ber = np.sort(_ns(g.berthed))
        wait = g.wait_h.values[np.argsort(g.berthed.values)]
        feats.loc[idx, "ship_wait_mean_1y"] = _mean(ber, wait, T_all[idx], 365 * DAY)
        feats.loc[idx, "ship_visits_1y"] = _count_before(ber, T_all[idx]) - _count_before(ber, T_all[idx] - 365 * DAY)


    # this ship's previous voyage (built by src/prev_voyage.py from calls at ALL Brazilian ports)
    prev_path = PROC / f"prev_h{horizon_h}.parquet"
    prev = pd.read_parquet(prev_path).set_index("call_id") if prev_path.exists() else pd.DataFrame(columns=PREV_FEATURES)
    for c in PREV_FEATURES:
        feats[c] = calls.call_id.map(prev[c]).values if c in prev else np.nan
    pe_path = PROC / f"prevE_h{horizon_h}.parquet"
    pe = pd.read_parquet(pe_path).set_index("call_id") if pe_path.exists() else pd.DataFrame(columns=PREVE_FEATURES + ["prev_port"])
    if "prev_port" in pe:
        pe["prev_port_code"] = pe.prev_port.astype("category").cat.codes.replace(-1, np.nan)
    for c in PREVE_FEATURES + ["prev_port_code"]:
        feats[c] = calls.call_id.map(pe[c]).values if c in pe else np.nan

    # weather: arrival day + 2 days before (actual weather as a stand-in for the forecast)
    w = weather.sort_values(["port", "date"]).copy()
    grp = w.groupby("port")
    w["precip_3d"] = grp.precipitation_sum.transform(lambda s: s.rolling(3, min_periods=1).sum())
    w["wind_max_3d"] = grp.wind_speed_10m_max.transform(lambda s: s.rolling(3, min_periods=1).max())
    w["gust_max_3d"] = grp.wind_gusts_10m_max.transform(lambda s: s.rolling(3, min_periods=1).max())
    key = pd.DataFrame({"port": calls.port, "date": calls.arrival.dt.normalize()})
    feats[["precip_3d", "wind_max_3d", "gust_max_3d"]] = key.merge(
        w[["port", "date", "precip_3d", "wind_max_3d", "gust_max_3d"]], on=["port", "date"], how="left"
    )[["precip_3d", "wind_max_3d", "gust_max_3d"]].values

    feats["log_tonnes"] = np.log1p(calls.tonnes)
    feats["teu"] = calls.teu
    feats["owner_foreign"] = calls.owner_foreign.astype(float)
    feats["month"] = calls.arrival.dt.month
    feats["dow"] = calls.arrival.dt.dayofweek
    feats["hour"] = calls.arrival.dt.hour
    for c in CAT_FEATURES:
        feats[c] = calls[c].astype("category")

    meta = calls[["call_id", "arrival", "wait_h"]].copy()
    meta["delayed"] = (calls.wait_h > 24).astype(int)
    return pd.concat([meta, feats[FEATURES]], axis=1)


if __name__ == "__main__":
    calls = pd.read_parquet(PROC / "calls.parquet")
    weather = pd.read_parquet(PROC / "weather.parquet")
    for h in [0, 72]:
        f = build_features(calls, weather, h)
        f.to_parquet(PROC / f"features_h{h}.parquet", index=False)
        print(f"h={h}", f.shape)
        print(f[["port_queue", "port_rho", "port_erlang_wait_h", "port_clear_time_h", "term_queue",
                 "near_wait_mean_7d", "precip_3d"]].describe().round(2).to_string())
        num = f[[c for c in FEATURES if c not in CAT_FEATURES]]
        print("corr with wait_h (top):")
        print(num.corrwith(f.wait_h, method="spearman").sort_values(key=abs, ascending=False).head(10).round(3).to_string())
