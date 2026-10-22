"""What-if audit: sweep the demo's sliders over many real ships and check that predictions and explanations"""
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from explain import CAUSES, apply_scenario, displaced, explain_row, load_monthly, queue_caps, whatif_predict  # noqa: E402
from load import PROC, ROOT  # noqa: E402
from scene import scene_data  # noqa: E402

TOL_P, TOL_H = 0.02, 0.05
LINE = CAUSES["line"]


def predict(b, row):
    r = explain_row(b, row)
    return r, r["prob"], r["range_h"]


def check_consistency(s, berths_changed=True):
    v = []
    g = lambda c: float(np.nan_to_num(s.get(c, 0.0)))  # noqa: E731
    if g("tc_queue") > g("term_queue") + 1e-9: v.append("tc_queue > term_queue")
    if g("term_queue") > g("port_queue") + 1e-9: v.append("term_queue > port_queue")
    if g("tc_queue") > g("pc_queue") + 1e-9: v.append("tc_queue > pc_queue")
    if g("pc_queue") > g("port_queue") + 1e-9: v.append("pc_queue > port_queue")
    if berths_changed and g("term_occupancy") > g("term_berths") + 1e-9: v.append("occupancy > berths after closing")
    for lvl in ("term", "tc"):
        if g(f"{lvl}_queue") == 0:
            for c in ("queue_tonnes", "queue_elapsed_mean_h", "clear_time_h", "work_clear_h"):
                if g(f"{lvl}_{c}") > 1e-9:
                    v.append(f"{lvl}_queue=0 but {lvl}_{c}={g(f'{lvl}_{c}'):.1f}")
    for c in [c for c in s.index if c.endswith(("_queue", "_queue_tonnes", "_clear_time_h", "_work_clear_h", "_occupancy"))]:
        if g(c) < -1e-9: v.append(f"{c} negative")
    return v


def audit(n_per=1, seed=0):
    rng = np.random.default_rng(seed)
    out = defaultdict(list)
    counts = defaultdict(int)
    bundles = {}
    for h in (0, 72):
        f = pd.read_parquet(PROC / f"features_h{h}.parquet")
        f = f[f.arrival >= "2024-01-01"]
        caps = queue_caps(pd.read_parquet(PROC / f"features_h{h}.parquet"))
        for port, gp in f.groupby(f.port.astype(str), observed=True):
            for month in (1, 4, 7, 10):
                gm = gp[gp.arrival.dt.month == month]
                if gm.empty:
                    continue
                for i in rng.choice(len(gm), min(n_per, len(gm)), replace=False):
                    row = gm.iloc[[i]]
                    key = (h, f"{row.arrival.iloc[0]:%Y-%m}")
                    if key not in bundles:
                        bundles[key] = load_monthly(*key)
                    b = bundles[key]
                    ship = f"h{h} {port} {row.arrival.iloc[0]:%Y-%m-%d %H:%M}"
                    q0 = int(row.term_queue.iloc[0])
                    c0 = int(np.nan_to_num(row.term_berths.iloc[0], nan=1))
                    qmax = max(q0, int(caps.get(str(row.terminal.iloc[0]), 15)))
                    counts["ships"] += 1

                    # identity
                    r0, p0, h0 = predict(b, row)
                    r1 = whatif_predict(b, row, q0, 0, qmax)
                    p1, h1 = r1["prob"], r1["range_h"]
                    counts["identity"] += 1
                    if abs(p1 - p0) > 1e-9 or abs(h1[1] - h0[1]) > 1e-6:
                        out["identity"].append({"ship": ship, "real": [p0, h0[1]], "untouched": [p1, h1[1]]})

                    def scen(q, closed):
                        rr = apply_scenario(row, q, c0 - closed)
                        r = whatif_predict(b, row, q, closed, qmax)       # exactly what the app shows
                        p, hh = r["prob"], r["range_h"]
                        counts["predictions"] += 1
                        tag = f"{ship} q={q} closed={closed}"
                        # range
                        lo, mid, hi = hh
                        if not (0 <= p <= 1 and np.isfinite([lo, mid, hi]).all() and 0 <= lo <= mid <= hi and mid < 1440):
                            out["range"].append({"case": tag, "p": p, "range": hh})
                        # consistency
                        for v in check_consistency(rr.iloc[0], berths_changed=closed > 0):
                            out["consistency"].append({"case": tag, "issue": v})
                        # explanation
                        if q + displaced(row, c0 - closed) == 0:
                            line_eff = sum(c["effect_pp"] for c in r["causes"] if c["cause"] == LINE)
                            if line_eff > 0.5:
                                out["explanation"].append({"case": tag, "issue": f"line cause +{line_eff:.1f} pts with empty line"})
                            for x in r["reasons"]:
                                if x["cause"] == LINE and x["effect_pp"] > 0.5:
                                    out["explanation"].append({"case": tag, "issue": f"reason: {x['text']}"})
                        # scene
                        sd = scene_data(rr, r, total_berths=c0, closed=closed, thr=b["thr"])
                        if sd["queue"] != q + displaced(row, c0 - closed) or sd["queueDrawn"] > sd["queue"] or sd["occupied"] > sd["berths"] - sd["closed"]:
                            out["scene"].append({"case": tag, "scene": {k: sd[k] for k in ("queue", "queueDrawn", "occupied", "berths", "closed")}})
                        return p, mid

                    qs = sorted({0, 1, 2, 5, 10, q0, qmax} & set(range(qmax + 1)))
                    seq = [scen(q, 0) for q in qs]
                    counts["mono_queue"] += 1
                    for (qa, (pa, ma)), (qb, (pb, mb)) in zip(zip(qs, seq), zip(qs[1:], seq[1:])):
                        if pb < pa - TOL_P or mb < ma * (1 - TOL_H) - 1:
                            out["mono_queue"].append({"ship": ship, "from_q": qa, "to_q": qb, "risk": [pa, pb], "wait_h": [ma, mb]})
                    if c0 > 1:
                        seq = [scen(q0, k) for k in range(c0)]
                        counts["mono_closed"] += 1
                        for k, ((pa, ma), (pb, mb)) in enumerate(zip(seq, seq[1:])):
                            if pb < pa - TOL_P or mb < ma * (1 - TOL_H) - 1:
                                out["mono_closed"].append({"ship": ship, "from_closed": k, "to_closed": k + 1,
                                                           "risk": [pa, pb], "wait_h": [ma, mb]})
    summary = {r: len(out[r]) for r in ("identity", "mono_queue", "mono_closed", "range", "consistency", "explanation", "scene")}
    report = {"counts": dict(counts), "violations": summary, "examples": {k: v[:15] for k, v in out.items()}}
    (ROOT / "report" / "audit_whatif.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = audit(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    print("checked:", rep["counts"])
    print("violations:", rep["violations"])
    for k, v in rep["examples"].items():
        for e in v[:3]:
            print(f"  [{k}]", json.dumps(e, default=float)[:220])
