"""Daily historical weather per port from the Open-Meteo archive (free, no key)."""
import time

import pandas as pd
import requests

from load import PROC

URL = "https://archive-api.open-meteo.com/v1/archive"
VARS = ["precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max"]


def fetch_port(lat, lon, start="2020-01-01", end="2024-12-31"):
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "daily": ",".join(VARS), "timezone": "America/Sao_Paulo"}
    for attempt in range(4):
        r = requests.get(URL, params=params, timeout=60)
        if r.status_code == 200:
            d = r.json()["daily"]
            return pd.DataFrame({"date": pd.to_datetime(d["time"]), **{v: d[v] for v in VARS}})
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()


def build():
    calls = pd.read_parquet(PROC / "calls.parquet")
    ports = calls.groupby("port")[["lat", "lon"]].median()
    out = []
    for port, (lat, lon) in ports.iterrows():
        w = fetch_port(round(lat, 3), round(lon, 3))
        w["port"] = port
        out.append(w)
        print("weather ok:", port, len(w), flush=True)
        time.sleep(1)
    w = pd.concat(out)
    w.to_parquet(PROC / "weather.parquet", index=False)
    return w


if __name__ == "__main__":
    print(build().describe())
