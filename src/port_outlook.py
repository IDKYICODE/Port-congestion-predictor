"""Port-level outlook and confidence checks, built from the monthly production models' predictions for 2024."""
import json

import numpy as np
import pandas as pd

from explain import load_monthly, prepare
from load import PROC, ROOT


def build_preds(h):
    f = pd.read_parquet(PROC / f"features_h{h}.parquet")
    f = f[f.arrival >= "2024-01-01"]
    parts = []
    for m in range(1, 13):
        g = f[f.arrival.dt.month == m]
        b = load_monthly(h, f"2024-{m:02d}")
        X = prepare(b, g)
        parts.append(pd.DataFrame({"call_id": g.call_id.values, "arrival": g.arrival.values, "port": g.port.astype(str).values,
                                   "terminal": g.terminal.astype(str).values, "p": b["clf"].predict_proba(X)[:, 1],
                                   "mid_h": np.maximum(np.expm1(b["qm"][0.5].predict(X)), 0),
                                   "delayed": g.delayed.values, "wait_h": g.wait_h.values}))
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(PROC / f"preds_2024_h{h}.parquet", index=False)
    return out


def check_confidence(p):
    conf = np.maximum(p.p, 1 - p.p)
    right = (p.p >= 0.5) == (p.delayed == 1)
    bins = pd.cut(conf, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True)
    t = pd.DataFrame({"conf": conf, "right": right}).groupby(bins, observed=True).agg(
        ships=("right", "size"), avg_confidence=("conf", "mean"), actually_right=("right", "mean"))
    return t.round(3)


def check_ports(p):
    w = p.assign(week=p.arrival.dt.to_period("W").dt.start_time).groupby(["port", "week"]).agg(
        ships=("p", "size"), pred_share=("p", "mean"), real_share=("delayed", "mean"),
        pred_wait=("mid_h", "median"), real_wait=("wait_h", "median"))
    w = w[w.ships >= 5]
    err = (w.pred_share - w.real_share).abs()
    return {"port_weeks": int(len(w)), "share_mae_pts": float(err.mean() * 100),
            "share_within_10pts": float((err <= 0.10).mean()),
            "share_corr": float(w.pred_share.corr(w.real_share)),
            "wait_median_abs_err_h": float((w.pred_wait - w.real_wait).abs().median())}


def outlook(preds, port, start, days=7):
    start = pd.Timestamp(start).normalize()
    g = preds[(preds.port == port) & (preds.arrival >= start) & (preds.arrival < start + pd.Timedelta(days=days))]
    if g.empty:
        return None
    t = g.groupby("terminal").agg(n=("p", "size"), risk=("p", "mean"))
    t = t[t.n >= 2] if (t.n >= 2).any() else t
    return {"ships": int(len(g)), "exp_delayed": float(g.p.sum()), "share": float(g.p.mean()),
            "typical_wait_h": float(g.mid_h.median()), "busiest_terminal": str(t.risk.idxmax()),
            "busiest_risk": float(t.risk.max()), "real_delayed": int(g.delayed.sum()),
            "start": start, "end": start + pd.Timedelta(days=days - 1)}


if __name__ == "__main__":
    res = {}
    for h in (0, 72):
        pr = build_preds(h)
        res[f"h{h}"] = {"confidence": check_confidence(pr).reset_index(names="band").astype({"band": str}).to_dict("records"),
                        "port_week": check_ports(pr)}
        print(f"\nhorizon {h}h - confidence vs reality:\n{check_confidence(pr).to_string()}")
        print("port-week outlook:", {k: round(v, 3) for k, v in check_ports(pr).items()})
    (ROOT / "report" / "confidence_port_check.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
