"""Load + clean ANTAQ berthing records into one table of cargo port calls."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
# files are keyed by UNBERTH year: 2025 is needed so ships arriving late 2024 with long waits aren't lost
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
LAST_ARRIVAL = "2024-12-31 23:59:59"
DATE_FMT = "%d/%m/%Y %H:%M:%S"
NAV_KEEP = {"Longo Curso", "Cabotagem"}

ATR_COLS = {
    "IDAtracacao": "call_id",
    "IDBerco": "berth_id",
    "Porto Atracação": "port",
    "Coordenadas": "coords",
    "Tipo da Autoridade Portuária": "authority",
    "Data Chegada": "arrival",
    "Data Atracação": "berthed",
    "Data Desatracação": "unberthed",
    "Tipo de Operação": "op_type",
    "Tipo de Navegação da Atracação": "nav_type",
    "Nacionalidade do Armador": "owner_foreign",
    "Terminal": "terminal",
    "Nº do IMO": "imo",
}


def _read(name, **kw):
    return pd.read_csv(RAW / name, sep=";", encoding="utf-8-sig", decimal=",", low_memory=False, **kw)


def load_calls(years=YEARS):
    atr = pd.concat([_read(f"{y}Atracacao.txt", usecols=list(ATR_COLS)) for y in years]).rename(columns=ATR_COLS)
    tmp = pd.concat([_read(f"{y}TemposAtracacao.txt", usecols=["IDAtracacao", "TEsperaAtracacao", "TAtracado"]) for y in years])
    tmp = tmp.rename(columns={"IDAtracacao": "call_id", "TEsperaAtracacao": "wait_antaq_h", "TAtracado": "berth_h_antaq"})
    for c in ["wait_antaq_h", "berth_h_antaq"]:
        # text column (decimal comma); "Valor Discrepante" (ANTAQ outlier flag) -> NaN
        tmp[c] = pd.to_numeric(tmp[c].astype(str).str.replace(",", "."), errors="coerce")

    df = atr.drop_duplicates("call_id").merge(tmp.drop_duplicates("call_id"), on="call_id", how="left")
    for c in ["arrival", "berthed", "unberthed"]:
        df[c] = pd.to_datetime(df[c], format=DATE_FMT, errors="coerce")

    df = df[(df.op_type == "Movimentação da Carga") & df.nav_type.isin(NAV_KEEP)]
    df = df.dropna(subset=["arrival", "berthed", "unberthed"])
    df = df[df.arrival <= LAST_ARRIVAL]
    df["wait_h"] = (df.berthed - df.arrival).dt.total_seconds() / 3600
    df["berth_h"] = (df.unberthed - df.berthed).dt.total_seconds() / 3600
    # impossible timestamps: berthed before arrival, or unberthed before berthed, or > 60 days
    df = df[(df.wait_h >= 0) & (df.berth_h > 0) & (df.wait_h < 24 * 60) & (df.berth_h < 24 * 60)]

    lonlat = df.coords.str.split(",", expand=True)
    df["lon"] = pd.to_numeric(lonlat[0], errors="coerce")
    df["lat"] = pd.to_numeric(lonlat[1], errors="coerce")
    df["owner_foreign"] = (df.owner_foreign == 2) | (df.owner_foreign == "Estrangeira")
    return df.drop(columns=["coords", "op_type"]).sort_values("arrival").reset_index(drop=True)


def load_cargo(call_ids, years=YEARS):
    """Per-call cargo summary: total tonnes, TEU, dominant cargo nature, share loaded (exports), cargo mix."""
    cols = ["IDAtracacao", "Natureza da Carga", "TEU", "VLPesoCargaBruta", "Sentido", "CDMercadoria"]
    parts = []
    for y in years:
        for chunk in _read(f"{y}Carga.txt", usecols=cols, chunksize=1_000_000):
            parts.append(chunk[chunk.IDAtracacao.isin(call_ids)])
    c = pd.concat(parts).rename(columns={"IDAtracacao": "call_id", "Natureza da Carga": "cargo_nature",
                                         "VLPesoCargaBruta": "tonnes", "TEU": "teu"})
    c["tonnes"] = pd.to_numeric(c.tonnes, errors="coerce")
    c["teu"] = pd.to_numeric(c.teu, errors="coerce")
    c["loaded_t"] = c.tonnes.where(c.Sentido == "Embarcados", 0.0)
    g = c.groupby("call_id")
    tot = g[["tonnes", "teu", "loaded_t"]].sum()
    tot["load_share"] = (tot.loaded_t / tot.tonnes).where(tot.tonnes > 0)   # 1 = pure export loading, 0 = pure unloading
    tot["n_goods"] = g.CDMercadoria.nunique()                               # different goods (NCM codes) on board
    tot["n_natures"] = g.cargo_nature.nunique()                             # different cargo types on board
    nature = c.sort_values("tonnes").groupby("call_id").cargo_nature.last()  # heaviest cargo nature
    return tot.drop(columns="loaded_t").join(nature).reset_index()


def build(top_n=15):
    calls = load_calls()
    top = calls[calls.arrival >= "2021-01-01"].port.value_counts().head(top_n).index
    calls = calls[calls.port.isin(top)]
    calls = calls.merge(load_cargo(set(calls.call_id)), on="call_id", how="left")
    PROC.mkdir(parents=True, exist_ok=True)
    calls.to_parquet(PROC / "calls.parquet", index=False)
    return calls


if __name__ == "__main__":
    df = build()
    print(df.shape)
    print(df.port.value_counts())
    print(df.wait_h.describe(percentiles=[.5, .75, .9, .95, .99]))
    print("share wait > 24h:", (df.wait_h > 24).mean().round(3))
    print("timestamp vs ANTAQ wait abs diff median:", (df.wait_h - df.wait_antaq_h).abs().median())
