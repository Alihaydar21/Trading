"""
03_pre_split_prep.py
----------------------------------------
Pre-Split Data Preparation (Multi-Asset, Intraday korrekt!)

Fixes:
- Features werden pro Handelstag berechnet -> keine Overnight-Sprünge
- Target bekommt Schwelle -> weniger Label-Rauschen
- output enthält timestamp,symbol,close (für Backtesting-Merge)
"""

import os
import pandas as pd
import numpy as np

# -----------------------------
# Config
# -----------------------------
SYMBOLS = ["AAPL", "MSFT", "NVDA"]
HOLD_MINUTES = 60
TARGET_MIN_RETURN = 0.0005  # 0.05% Schwelle (anpassbar)

# -----------------------------
# Helpers
# -----------------------------
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

# -----------------------------
# Paths
# -----------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
DATA_DIR = os.path.join(ROOT, "data")

all_assets = []

for symbol in SYMBOLS:
    print("\n==============================")
    print(f"⚙️ Pre-Split Preparation: {symbol}")
    print("==============================")

    input_path = os.path.join(DATA_DIR, f"{symbol}_1min.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Fehlt: {input_path} (erst 01 laufen lassen)")

    df = pd.read_csv(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # symbol-spalte existiert evtl. schon; sicher setzen
    df["symbol"] = symbol

    # Handelstag definieren (UTC-based). Für absolute Korrektheit: US/Eastern.
    df["date"] = df["timestamp"].dt.date

    # -------------------------
    # Features (intraday / per day)
    # -------------------------
    features = pd.DataFrame(index=df.index)

    # Returns intraday: groupby(date)
    features["return_1min"] = df.groupby("date")["close"].pct_change(1)
    features["return_15min"] = df.groupby("date")["close"].pct_change(15)

    # EMA intraday: pro Tag neu starten
    def ema_grouped(x, span):
        return x.ewm(span=span, adjust=False).mean()

    ema_5  = df.groupby("date")["close"].apply(lambda s: ema_grouped(s, 5)).reset_index(level=0, drop=True)
    ema_15 = df.groupby("date")["close"].apply(lambda s: ema_grouped(s, 15)).reset_index(level=0, drop=True)
    ema_30 = df.groupby("date")["close"].apply(lambda s: ema_grouped(s, 30)).reset_index(level=0, drop=True)

    features["ema_5_30_diff"] = ema_5 - ema_30
    features["ema_15_30_diff"] = ema_15 - ema_30

    # RSI intraday: pro Tag
    rsi = df.groupby("date")["close"].apply(lambda s: compute_rsi(s, 14)).reset_index(level=0, drop=True)
    features["rsi_14"] = rsi

    # -------------------------
    # Target: 60m Future Return mit Schwelle
    # -------------------------
    future_close = df.groupby("date")["close"].shift(-HOLD_MINUTES)
    ret60 = future_close / df["close"] - 1.0

    df["target_trend_1h"] = (ret60 > TARGET_MIN_RETURN).astype(int)

    # -------------------------
    # Final
    # (close behalten -> Backtesting sauber mergen)
    # -------------------------
    df_final = pd.concat(
        [
            df[["timestamp", "symbol", "close", "target_trend_1h"]],
            features
        ],
        axis=1
    )

    df_final = df_final.dropna().reset_index(drop=True)
    all_assets.append(df_final)

    print(f"✓ {symbol}: {len(df_final)} Zeilen | target_rate={df_final['target_trend_1h'].mean():.4f}")

df_all = pd.concat(all_assets).reset_index(drop=True)

output_path = os.path.join(DATA_DIR, "MULTI_prepared_intraday.csv")
df_all.to_csv(output_path, index=False)

print("\n==============================")
print("✅ Pre-Split Preparation abgeschlossen!")
print("Gesamtzeilen:", len(df_all))
print("Gespeichert unter:", output_path)
print("==============================")
