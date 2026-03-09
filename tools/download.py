#!/usr/bin/env python3
"""Free data: 20 years of daily bars from the Yahoo chart API for the universe in data/universe.csv (no key).
   python tools/download.py        -> data/raw/yahoo/<symbol>.json, data/derived/prices.parquet
The parquet is committed, so the pipeline runs without this step; rerun it to refresh the window."""
import csv
import datetime as dt
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "yahoo")
DER = os.path.join(ROOT, "data", "derived")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"


def get(url, retries=3, timeout=60):
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if k == retries - 1:
                print("  failed", url[:80], e)
                return None
            time.sleep(2 * (k + 1))


def yahoo(sym):
    raw = get(f"https://query2.finance.yahoo.com/v8/finance/chart/{sym}?range=20y&interval=1d&events=div,splits", retries=2, timeout=30)
    if raw is None:
        return None
    try:
        return json.loads(raw)["chart"]["result"][0]
    except Exception:  # noqa: BLE001
        return None


def main():
    os.makedirs(RAW, exist_ok=True); os.makedirs(DER, exist_ok=True)
    with open(os.path.join(ROOT, "data", "universe.csv"), newline="") as f:
        uni = list(csv.DictReader(f))
    bars = []
    for u in uni:
        sym = u["symbol"]; p = os.path.join(RAW, f"{sym}.json")
        if os.path.exists(p):
            r = json.load(open(p))
        else:
            r = yahoo(sym)
            if r is None:
                print("  no data", sym); continue
            json.dump(r, open(p, "w")); time.sleep(0.25)
        ts = r.get("timestamp") or []; q = r["indicators"]["quote"][0]; adj = r["indicators"].get("adjclose", [{}])[0].get("adjclose", [None] * len(ts))
        n = 0
        for i, t in enumerate(ts):
            c = q["close"][i]
            if c is None:
                continue
            d = dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat()
            bars.append((sym, d, q["open"][i] if q["open"][i] is not None else c, q["high"][i] if q["high"][i] is not None else c, q["low"][i] if q["low"][i] is not None else c, c, adj[i] if adj[i] is not None else c, q["volume"][i] or 0)); n += 1
        print(f"  {sym}: {n} bars")
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = list(zip(*bars))
    table = pa.table({"symbol": cols[0], "date": cols[1], "open": pa.array(cols[2], pa.float64()), "high": pa.array(cols[3], pa.float64()), "low": pa.array(cols[4], pa.float64()), "close": pa.array(cols[5], pa.float64()), "adjclose": pa.array(cols[6], pa.float64()), "volume": pa.array(cols[7], pa.float64())})
    pq.write_table(table, os.path.join(DER, "prices.parquet"), compression="zstd")
    print(f"prices: {len(bars)} bars -> {DER}")


if __name__ == "__main__":
    main()
