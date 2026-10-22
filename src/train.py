"""Train baselines + calibrated LightGBM + conformal hour ranges; evaluate on 2024 (never touched before)."""
import json
import pickle

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

from features import CAT_FEATURES, FEATURES, QUEUE_MATHS
from load import PROC, ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS, REPORT = ROOT / "models", ROOT / "report"
SPLITS = {"train": ("2020-03-01", "2022-12-31"), "val": ("2023-01-01", "2023-06-30"),
          "cal": ("2023-07-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")}
COST_FN, COST_FP = 3.0, 1.0   # assumption: missing a delay costs 3x a false alarm
COVERAGE = 0.80
STATUS_CUTS = [48, 96]       # 7-day mean wait (h): green / yellow / red

# chosen by random search on 2023 H2
PARAMS = dict(n_estimators=4000, learning_rate=0.01, num_leaves=63, min_child_samples=400, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, cat_smooth=50.0, verbose=-1)


UP = ("_queue", "_queue_tonnes", "_clear_time_h", "_work_clear_h", "_occupancy", "_rho", "_erlang_wait_h",
      "_overloaded", "_queue_elapsed_mean_h")


def mono_vector(feats):
    """Hard rules for LightGBM: more ships waiting / fuller berths / longer clearing time can only RAISE risk and"""
    return [1 if f.endswith(UP) else -1 if f.endswith("_berths") else 0 for f in feats]


def split(df):
    end = pd.Timedelta(days=1)
    return {k: df[(df.arrival >= a) & (df.arrival < pd.Timestamp(b) + end)] for k, (a, b) in SPLITS.items()}


def conformal_q(lo, hi, y, coverage=COVERAGE):
    """Conformalized quantile regression: widen [lo, hi] by q so held-out coverage ~= target."""
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * coverage) / n)))


def cost_threshold(p, y):
    grid = np.linspace(0.05, 0.95, 91)
    costs = [COST_FN * ((p < t) & (y == 1)).sum() + COST_FP * ((p >= t) & (y == 0)).sum() for t in grid]
    return float(grid[int(np.argmin(costs))])


def status(wait7):
    return np.digitize(np.nan_to_num(wait7, nan=0), STATUS_CUTS)  # 0 green, 1 yellow, 2 red


def _fit(cls, s, feats, target, **kw):
    """Early-stop on val to pick the number of rounds, then refit on train+val with that many (+10%)."""
    tgt = target if callable(target) else (lambda d: d[target])
    cats = [c for c in CAT_FEATURES if c in feats]
    m = cls(**PARAMS, **kw).fit(s["train"][feats], tgt(s["train"]), eval_set=[(s["val"][feats], tgt(s["val"]))],
                                categorical_feature=cats, callbacks=[lgb.early_stopping(150, verbose=False)])
    full = pd.concat([s["train"], s["val"]])
    return cls(**dict(PARAMS, n_estimators=int(m.best_iteration_ * 1.1) + 1), **kw).fit(
        full[feats], tgt(full), categorical_feature=cats)


def _fit_clf(s, feats):
    return _fit(lgb.LGBMClassifier, s, feats, "delayed")


def _scores(y, p, thr):
    return {"auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "f1": f1_score(y, p >= thr),
            "recall": float(((p >= thr) & (y == 1)).sum() / max(1, y.sum())),
            "precision": float(((p >= thr) & (y == 1)).sum() / max(1, (p >= thr).sum()))}


def run(h):
    df = pd.read_parquet(PROC / f"features_h{h}.parquet")
    for c in CAT_FEATURES:
        df[c] = df[c].astype("category")
    s = split(df)
    yte, ycal = s["test"].delayed.values, s["cal"].delayed.values
    res = {}

    # B1: port's recent delay share
    p = s["test"].port_delayed_share_30d.fillna(s["train"].delayed.mean()).values
    res["B1 port 30d delay rate"] = _scores(yte, p, cost_threshold(s["cal"].port_delayed_share_30d.fillna(.5).values, ycal))

    # B2: traffic light -> delay rate of that light in train
    rate = pd.Series(s["train"].delayed.values).groupby(status(s["train"].port_wait_mean_7d.values)).mean()
    p = rate.reindex(status(s["test"].port_wait_mean_7d.values)).values
    res["B2 traffic-light rule"] = _scores(yte, p, cost_threshold(rate.reindex(status(s["cal"].port_wait_mean_7d.values)).values, ycal))

    # B3: queue maths alone (logistic on the 8 queue-maths features)
    med = s["train"][QUEUE_MATHS].median()
    prep = lambda d: np.log1p(d[QUEUE_MATHS].fillna(med).clip(lower=0))  # noqa: E731
    lr = LogisticRegression(max_iter=2000).fit(prep(s["train"]), s["train"].delayed)
    res["B3 queue maths only"] = _scores(yte, lr.predict_proba(prep(s["test"]))[:, 1],
                                         cost_threshold(lr.predict_proba(prep(s["cal"]))[:, 1], ycal))

    # B5: LightGBM without queue maths (ablation)
    no_qm = [f for f in FEATURES if f not in QUEUE_MATHS]
    m5 = _fit_clf(s, no_qm)
    res["B5 LightGBM w/o queue maths"] = _scores(yte, m5.predict_proba(s["test"][no_qm])[:, 1],
                                                 cost_threshold(m5.predict_proba(s["cal"][no_qm])[:, 1], ycal))

    # Ours: all features + isotonic calibration on cal
    clf = _fit_clf(s, FEATURES)
    raw_cal, raw_te = clf.predict_proba(s["cal"][FEATURES])[:, 1], clf.predict_proba(s["test"][FEATURES])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, ycal)
    p_cal, p_te = iso.predict(raw_cal), iso.predict(raw_te)
    thr = cost_threshold(p_cal, ycal)
    res["Ours uncalibrated"] = _scores(yte, raw_te, cost_threshold(raw_cal, ycal))
    res["Ours (calibrated)"] = _scores(yte, p_te, thr)

    # hours: quantile models on log1p(wait), conformalized on cal
    qm = {}
    for a in (0.1, 0.5, 0.9):
        qm[a] = _fit(lgb.LGBMRegressor, s, FEATURES, lambda d: np.log1p(d.wait_h), objective="quantile", alpha=a)
    pred = lambda d, a: qm[a].predict(d[FEATURES])  # noqa: E731
    q = conformal_q(pred(s["cal"], .1), pred(s["cal"], .9), np.log1p(s["cal"].wait_h.values))
    lo, mid, hi = (np.expm1(pred(s["test"], .1) - q).clip(0), np.expm1(pred(s["test"], .5)),
                   np.expm1(pred(s["test"], .9) + q))
    yh = s["test"].wait_h.values
    hours = {"mae_p50_h": float(np.mean(np.abs(mid - yh))), "rmse_p50_h": float(np.sqrt(np.mean((mid - yh) ** 2))),
             "median_abs_err_h": float(np.median(np.abs(mid - yh))),
             "coverage_80": float(np.mean((yh >= lo) & (yh <= hi))), "median_width_h": float(np.median(hi - lo)),
             "naive_mae_h": float(np.mean(np.abs(s["train"].groupby("port", observed=True).wait_h.median()
                                                  .reindex(s["test"].port).values - yh)))}

    # plain L2 regressor on hours, only to quote an RMSE comparable in spirit to published work
    l2 = _fit(lgb.LGBMRegressor, s, FEATURES, "wait_h")
    hours["rmse_l2_h"] = float(np.sqrt(np.mean((l2.predict(s["test"][FEATURES]) - yh) ** 2)))

    # does queue maths help when training data is scarce? (same test set, subsampled train)
    rng = np.random.default_rng(0)
    scarce = {}
    for frac in (0.02, 0.05, 0.2, 1.0):
        idx = rng.choice(len(s["train"]), int(frac * len(s["train"])), replace=False)
        vidx = rng.choice(len(s["val"]), max(200, int(frac * len(s["val"]))), replace=False)
        sub = dict(s, train=s["train"].iloc[idx], val=s["val"].iloc[vidx])  # refit uses val too: shrink both
        scarce[frac] = {name: roc_auc_score(yte, _fit_clf(sub, f).predict_proba(s["test"][f])[:, 1])
                        for name, f in [("with_qm", FEATURES), ("without_qm", no_qm)]}

    per_port = {p: roc_auc_score(g.delayed, p_te[s["test"].index.get_indexer(g.index)])
                for p, g in s["test"].groupby("port", observed=True) if g.delayed.nunique() == 2}

    MODELS.mkdir(exist_ok=True)
    with open(MODELS / f"h{h}.pkl", "wb") as f:
        pickle.dump({"clf": clf, "iso": iso, "thr": thr, "qm": qm, "q": q, "features": FEATURES,
                     "categories": {c: list(df[c].cat.categories) for c in CAT_FEATURES}}, f)

    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for name, pp in [("uncalibrated", raw_te), ("calibrated", p_te)]:
        fy, fx = calibration_curve(yte, pp, n_bins=10)
        ax.plot(fx, fy, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="predicted probability", ylabel="observed delay rate", title=f"Reliability, horizon {h}h")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT / "figures" / f"reliability_h{h}.png", dpi=150)
    plt.close(fig)

    out = {"horizon_h": h, "n": {k: len(v) for k, v in s.items()}, "test_base_rate": float(yte.mean()),
           "threshold": thr, "classification": res, "hours": hours, "per_port_auc": per_port,
           "scarce_data_auc": scarce}
    (REPORT / f"metrics_h{h}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    for h in (0, 72):
        o = run(h)
        print(f"\n=== horizon {h}h | test n={o['n']['test']} base rate={o['test_base_rate']:.3f} thr={o['threshold']:.2f}")
        print(pd.DataFrame(o["classification"]).T.round(3).to_string())
        print({k: round(v, 3) for k, v in o["hours"].items()})
        print("scarce-data AUC:", {k: {a: round(b, 3) for a, b in v.items()} for k, v in o["scarce_data_auc"].items()})
        print("per-port AUC:", {k[:18]: round(v, 3) for k, v in o["per_port_auc"].items()})
