"""
03_pre_split_prep.py
----------------------------------------
Pre-Split Data Preparation für Intraday-Daten (1-Minute Bars)
Multi-Asset-Version – (ohne absolute Preise)

Features:
- Returns
- EMA-Differenzen
- RSI

Target:
- Punktuelle 60-Minuten-Future-Rendite
"""

import os
import pandas as pd
import numpy as np

# --------------------------------------------------------
# Feature-Funktionen
# --------------------------------------------------------

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


# --------------------------------------------------------
# 1. Pfade
# --------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
DATA_DIR = os.path.join(ROOT, "data")

# --------------------------------------------------------
# 2. Assets
# --------------------------------------------------------

symbols = ["AAPL", "MSFT", "NVDA"]
all_assets = []

# --------------------------------------------------------
# 3. Loop über alle Assets
# --------------------------------------------------------

for symbol in symbols:

    print("\n==============================")
    print(f"⚙️ Pre-Split Preparation: {symbol}")
    print("==============================")

    input_path = os.path.join(DATA_DIR, f"{symbol}_1min.csv")
    df = pd.read_csv(input_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["symbol"] = symbol

    # ----------------------------------------------------
    # Feature Engineering (preislevel-frei!)
    # ----------------------------------------------------

    features = pd.DataFrame(index=df.index)

    # Returns
    features["return_1min"] = df["close"].pct_change(1)
    features["return_15min"] = df["close"].pct_change(15)

    # EMAs (nur zur Differenzbildung)
    ema_5 = df["close"].ewm(span=5, adjust=False).mean()
    ema_15 = df["close"].ewm(span=15, adjust=False).mean()
    ema_30 = df["close"].ewm(span=30, adjust=False).mean()

    features["ema_5_30_diff"] = ema_5 - ema_30
    features["ema_15_30_diff"] = ema_15 - ema_30

    # RSI
    features["rsi_14"] = compute_rsi(df["close"], 14)

    # ----------------------------------------------------
    # Target: punktueller Future Return (60 Minuten)
    # ----------------------------------------------------

    df["future_close_60m"] = df["close"].shift(-60)

    df["target_trend_1h"] = (
        df["future_close_60m"] > df["close"]
    ).astype(int)

    # ----------------------------------------------------
    # Zusammenführen
    # ----------------------------------------------------

    df_final = pd.concat(
        [
            df[["timestamp", "symbol", "target_trend_1h"]],
            features
        ],
        axis=1
    )

    df_final = df_final.dropna().reset_index(drop=True)
    all_assets.append(df_final)

    print(f"✓ {symbol}: {len(df_final)} Zeilen")

# --------------------------------------------------------
# 4. Speichern
# --------------------------------------------------------

df_all = pd.concat(all_assets).reset_index(drop=True)

output_path = os.path.join(DATA_DIR, "MULTI_prepared_intraday.csv")
df_all.to_csv(output_path, index=False)

print("\n==============================")
print("✅ Pre-Split Preparation abgeschlossen!")
print("Gesamtzeilen:", len(df_all))
print("Gespeichert unter:", output_path)
print("==============================")
