"""Idea A: this ship's previous voyage, from ANTAQ calls at ALL Brazilian ports."""
import numpy as np
import pandas as pd

from load import PROC, load_calls

PREV_FEATURES = ["prev_days_since_departure", "prev_wait_h", "prev_same_port"]


def build(horizons=(0, 72)):
    allc = load_calls()[["call_id", "imo", "port", "arrival", "unberthed", "wait_h"]].dropna(subset=["imo"])
    ours = pd.read_parquet(PROC / "calls.parquet")[["call_id", "imo", "port", "arrival"]]
    allc = allc.sort_values("unberthed")
    by_imo = {imo: g for imo, g in allc.groupby("imo")}
    ns = lambda s: s.values.astype("datetime64[ns]").astype(np.int64)  # noqa: E731
    for h in horizons:
        out = pd.DataFrame(np.nan, index=ours.index, columns=PREV_FEATURES)
        T_all = ns(ours.arrival) - h * 3600 * 10**9
        for imo, idx in ours.dropna(subset=["imo"]).groupby("imo").groups.items():
            g = by_imo.get(imo)
            if g is None:
                continue
            idx = np.asarray(idx)
            k = np.searchsorted(ns(g.unberthed), T_all[idx]) - 1          # last departure strictly before T
            ok = k >= 0
            kk = np.maximum(k, 0)
            out.loc[idx, "prev_days_since_departure"] = np.where(ok, (T_all[idx] - ns(g.unberthed)[kk]) / (86400 * 10**9), np.nan)
            out.loc[idx, "prev_wait_h"] = np.where(ok, g.wait_h.values[kk], np.nan)
            out.loc[idx, "prev_same_port"] = np.where(ok, (g.port.values[kk] == ours.port.values[idx]).astype(float), np.nan)
        out.insert(0, "call_id", ours.call_id.values)
        out.to_parquet(PROC / f"prev_h{h}.parquet", index=False)
        print(f"h={h}: coverage", out.prev_wait_h.notna().mean().round(3), "| median days since departure",
              out.prev_days_since_departure.median().round(1))


# richer voyage history: days sailed vs usual route time, previous port, last 3 waits
PREVE_FEATURES = ["prev_sail_days", "prev_route_typical_days", "prev_early_days", "prev3_wait_mean"]
PREVE_CATS = ["prev_port"]


def build_e(horizons=(0, 72), typical_until="2022-12-31"):
    """Sailing time of the last leg vs the typical time on that route (arriving early on purpose often means"""
    allc = load_calls()[["call_id", "imo", "port", "arrival", "unberthed", "wait_h"]].dropna(subset=["imo"])
    allc = allc.sort_values(["imo", "arrival"])
    nxt = allc.groupby("imo").shift(-1)
    legs = pd.DataFrame({"from": allc.port, "to": nxt.port,
                         "days": (nxt.arrival - allc.unberthed).dt.total_seconds() / 86400, "arr": nxt.arrival})
    legs = legs[(legs.days > 0) & (legs.days < 60) & (legs.arr <= typical_until)]
    typical = legs.groupby(["from", "to"]).days.median()

    ours = pd.read_parquet(PROC / "calls.parquet")[["call_id", "imo", "port", "arrival"]]
    allc = allc.sort_values("unberthed")
    by_imo = {imo: g for imo, g in allc.groupby("imo")}
    ns = lambda s: s.values.astype("datetime64[ns]").astype(np.int64)  # noqa: E731
    DAYNS = 86400 * 10**9
    for h in horizons:
        out = pd.DataFrame(np.nan, index=ours.index, columns=PREVE_FEATURES)
        out["prev_port"] = None
        T_all = ns(ours.arrival) - h * 3600 * 10**9
        for imo, idx in ours.dropna(subset=["imo"]).groupby("imo").groups.items():
            g = by_imo.get(imo)
            if g is None:
                continue
            idx = np.asarray(idx)
            unb = ns(g.unberthed)
            k = np.searchsorted(unb, T_all[idx]) - 1
            ok = k >= 0
            kk = np.maximum(k, 0)
            sail = (ns(ours.arrival)[idx] - unb[kk]) / DAYNS          # ETA assumed known (as for calendar hints)
            prev_port = g.port.values[kk]
            typ = typical.reindex(list(zip(prev_port, ours.port.values[idx]))).values
            w = g.wait_h.values
            cs = np.concatenate([[0.0], np.cumsum(w)])
            n3 = np.minimum(kk + 1, 3)
            out.loc[idx, "prev_sail_days"] = np.where(ok, sail, np.nan)
            out.loc[idx, "prev_route_typical_days"] = np.where(ok, typ, np.nan)
            out.loc[idx, "prev_early_days"] = np.where(ok, typ - sail, np.nan)
            out.loc[idx, "prev3_wait_mean"] = np.where(ok, (cs[kk + 1] - cs[kk + 1 - n3]) / n3, np.nan)
            out.loc[idx, "prev_port"] = np.where(ok, prev_port, None)
        out.insert(0, "call_id", ours.call_id.values)
        out.to_parquet(PROC / f"prevE_h{h}.parquet", index=False)
        print(f"h={h}: early-days coverage", out.prev_early_days.notna().mean().round(3),
              "| median early days", out.prev_early_days.median().round(2))


if __name__ == "__main__":
    build()
    build_e()
