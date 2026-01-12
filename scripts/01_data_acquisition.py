"""
01_data_acquisition.py
----------------------------------------
Lädt 1-Minuten-Bars für AAPL, MSFT, NVDA über Alpaca (Historical Data API)
und speichert pro Symbol eine CSV in ../data/<SYMBOL>_1min.csv

Wichtig:
- Läuft auf SERVER im Repo-Ordner, nicht in /tmp/pycharm_project_XX
- ALPACA_API_KEY und ALPACA_SECRET_KEY müssen als ENV gesetzt sein
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from alpaca.data import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# -------------------------------
# Config
# -------------------------------
SYMBOLS = ["AAPL", "MSFT", "NVDA"]
START_DATE = "2022-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# -------------------------------
# Keys
# -------------------------------
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise RuntimeError("ALPACA_API_KEY oder ALPACA_SECRET_KEY fehlen (ENV Variables).")

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# -------------------------------
# Paths
# -------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Projekt-Root:", PROJECT_ROOT)
print("Data-Dir     :", DATA_DIR)

# -------------------------------
# Request + Fetch
# -------------------------------
request_params = StockBarsRequest(
    symbol_or_symbols=SYMBOLS,
    timeframe=TimeFrame.Minute,
    adjustment=Adjustment.ALL,
    start=START_DATE,
    end=END_DATE,
)

try:
    bars = client.get_stock_bars(request_params)
except Exception as e:
    print("\n[ERROR] Alpaca Request fehlgeschlagen.")
    print("Grund:", e)
    print("\nTipp: Falls 'subscription does not permit querying recent SIP data':")
    print("- dann hast du keinen Zugriff auf SIP/Realtime/Recent für 1Min.")
    print("- Lösung: IEX Daten / älteren Zeitraum / anderen Provider (Yahoo) für recent.")
    raise

df = bars.df

print("\n=== HEAD (raw) ===")
print(df.head())
print("\n=== TAIL (raw) ===")
print(df.tail())

if not isinstance(df.index, pd.MultiIndex):
    raise RuntimeError(f"Unerwarteter Index-Typ: {type(df.index)}. Erwartet MultiIndex (symbol, timestamp).")

print("\nIndex names:", df.index.names)
available_symbols = df.index.get_level_values(0).unique().tolist()
print("Verfügbare Symbole in df:", available_symbols)

saved_paths = []

for symbol in SYMBOLS:
    if symbol not in available_symbols:
        print(f"[WARN] {symbol} wurde nicht geliefert. Überspringe.")
        continue

    df_sym = df.xs(symbol, level=0).copy()
    df_sym = df_sym.reset_index()  # timestamp wird Spalte

    # Sortieren
    df_sym["timestamp"] = pd.to_datetime(df_sym["timestamp"], errors="coerce")
    df_sym = df_sym.sort_values("timestamp").reset_index(drop=True)

    # Symbolspalte mitgeben (hilft später bei merges)
    df_sym.insert(0, "symbol", symbol)

    csv_path = DATA_DIR / f"{symbol}_1min.csv"
    df_sym.to_csv(csv_path, index=False)

    saved_paths.append(csv_path)
    print(f"✅ gespeichert: {csv_path} | rows={len(df_sym)} | "
          f"min_ts={df_sym['timestamp'].min()} | max_ts={df_sym['timestamp'].max()}")

print("\n✅ Fertig. Gespeicherte Dateien:")
for p in saved_paths:
    print(" -", p)

try:
    last_ts = df.index.get_level_values(-1).max()
    print("\nLetzter Timestamp (gesamt):", last_ts)
except Exception as e:
    print("\n[WARN] Konnte letzten Timestamp nicht bestimmen:", e)
