"""Build the production setup: one model bundle per month, retrained on everything known at the month's start."""
import json
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from features import CAT_FEATURES, FEATURES
from load import ROOT
from train import PARAMS, conformal_q, cost_threshold
from walkforward import N_EST, known_rows, load

OUT = ROOT / "models" / "monthly"
FAST = dict(PARAMS, learning_rate=0.03, n_estimators=430)   # hour models: 3x faster, same quality in tests


def fit_bundle(tr, h):
    clf = lgb.LGBMClassifier(**dict(PARAMS, n_estimators=N_EST[h])).fit(
        tr[FEATURES], tr.delayed, categorical_feature=CAT_FEATURES)
    qm = {a: lgb.LGBMRegressor(**FAST, objective="quantile", alpha=a).fit(
        tr[FEATURES], np.log1p(tr.wait_h), categorical_feature=CAT_FEATURES) for a in (0.1, 0.5, 0.9)}
    return clf, qm


def run(h):
    df = load(h)
    rows_out = []
    for m in pd.date_range("2023-07-01", "2024-12-01", freq="MS"):
        rows = df[(df.arrival >= m) & (df.arrival < m + pd.offsets.MonthBegin(1))]
        clf, qm = fit_bundle(known_rows(df, m), h)
        o = pd.DataFrame({"arrival": rows.arrival, "delayed": rows.delayed, "wait_h": rows.wait_h,
                          "p": clf.predict_proba(rows[FEATURES])[:, 1],
                          **{f"q{int(a * 100)}": qm[a].predict(rows[FEATURES]) for a in qm}})
        rows_out.append(o)
        if m.year == 2024:
            with open(OUT / f"h{h}_{m:%Y-%m}.pkl", "wb") as f:
                pickle.dump({"clf": clf, "qm": qm, "features": FEATURES, "iso": None, "trained_until": f"{m:%Y-%m-%d}",
                             "categories": {c: list(df[c].cat.categories) for c in CAT_FEATURES}}, f)
        print(f"h={h} {m:%Y-%m} done", flush=True)

    o = pd.concat(rows_out)
    sel, te = o[o.arrival < "2024-01-01"], o[o.arrival >= "2024-01-01"]
    thr = cost_threshold(sel.p.values, sel.delayed.values)
    q = conformal_q(sel.q10.values, sel.q90.values, np.log1p(sel.wait_h.values))
    meta = {"thr": thr, "q": q}
    (OUT / f"h{h}_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    y, p, w = te.delayed.values, te.p.values, te.wait_h.values
    lo, mid, hi = np.expm1(te.q10 - q).clip(0).values, np.expm1(te.q50).values, np.expm1(te.q90 + q).values
    err = np.abs(mid - w)
    metrics = {"horizon_h": h, "n_test": len(te), "threshold": thr, "conformal_q": q,
               "accuracy_at_0.5": accuracy_score(y, p >= 0.5), "auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p),
               "recall_at_thr": float(((p >= thr) & (y == 1)).sum() / y.sum()),
               "precision_at_thr": float(((p >= thr) & (y == 1)).sum() / (p >= thr).sum()),
               "accuracy_at_thr": accuracy_score(y, p >= thr),
               "hours_median_err": float(np.median(err)), "hours_mean_err": float(err.mean()),
               "hours_within_12h": float(np.mean(err <= 12)), "hours_within_24h": float(np.mean(err <= 24)),
               "range_coverage_80": float(np.mean((w >= lo) & (w <= hi))), "range_median_width": float(np.median(hi - lo))}
    (ROOT / "report" / f"metrics_monthly_h{h}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}), flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for h in (0, 72):
        run(h)
