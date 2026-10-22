"""Monthly retraining helpers: which outcomes are known at a cutoff date, and how many trees to use."""
import pandas as pd

from features import CAT_FEATURES
from load import PROC

TRAIN_START = "2020-03-01"
N_EST = {0: 1269, 72: 773}  # trees per horizon, from early stopping on 2023 H1


def known_rows(df, cutoff, start=None):
    """Training rows whose outcome is known at the cutoff: already berthed, or already waiting > 24 h."""
    cutoff = pd.Timestamp(cutoff)
    berthed = df.arrival + pd.to_timedelta(df.wait_h, unit="h")
    tr = df[(df.arrival >= (start or TRAIN_START)) & (df.arrival < cutoff)]
    b = berthed.loc[tr.index]
    known = (b < cutoff) | (cutoff - tr.arrival > pd.Timedelta(hours=24))
    return tr[known]


def load(h):
    df = pd.read_parquet(PROC / f"features_h{h}.parquet")
    for c in CAT_FEATURES:
        df[c] = df[c].astype("category")
    return df
