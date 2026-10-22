"""Recompute the 2024 test results from the saved predictions and print them (run after port_outlook.py)."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from load import PROC, ROOT


def results(h):
    p = pd.read_parquet(PROC / f"preds_2024_h{h}.parquet")
    thr = json.loads((ROOT / "models" / "monthly" / f"h{h}_meta.json").read_text(encoding="utf-8"))["thr"]
    y, prob = p.delayed.values, p.p.values
    return {
        "ships": len(p),
        "accuracy": ((prob >= 0.5) == y).mean(),
        "caught": ((prob >= thr) & (y == 1)).sum() / y.sum(),
        "auc": roc_auc_score(y, prob),
        "hours": np.median(np.abs(p.mid_h.values - p.wait_h.values)),
        "thr": thr,
    }


if __name__ == "__main__":
    r = {h: results(h) for h in (0, 72)}
    print("Test on all 2024 arrivals (never used to build the models)\n")
    print(f"{'':32}{'At arrival':>14}{'3 days before':>16}")
    print(f"{'Ships tested':32}{r[0]['ships']:>14,}{r[72]['ships']:>16,}")
    print(f"{'Accuracy (yes/no right)':32}{r[0]['accuracy']:>14.1%}{r[72]['accuracy']:>16.1%}")
    print(f"{'Delays caught (alarm line)':32}{r[0]['caught']:>14.1%}{r[72]['caught']:>16.1%}")
    print(f"{'ROC-AUC':32}{r[0]['auc']:>14.3f}{r[72]['auc']:>16.3f}")
    print(f"{'Typical hours error (median)':32}{r[0]['hours']:>13.1f}h{r[72]['hours']:>15.1f}h")
    print(f"\nAlarm line: {r[0]['thr']:.0%} at arrival, {r[72]['thr']:.0%} three days before")
