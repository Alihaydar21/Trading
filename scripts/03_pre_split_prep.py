"""
03_data_preparation.py
----------------------------------------
Pre-Split Data Preparation für Intraday-Daten (1-Minute Bars)

Dieses Skript:
- lädt Intraday-Rohdaten
- berechnet leak-free technische Features
- erzeugt das Target: Trend der nächsten Stunde
- entfernt ungültige Zeilen
- speichert das vorbereitete Dataset
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

input_path = os.path.join(DATA_DIR, "AAPL_1min.csv")
output_path = os.path.join(DATA_DIR, "AAPL_prepared_intraday.csv")

print("Lade Datei:", input_path)

# --------------------------------------------------------
# 2. Daten laden
# --------------------------------------------------------

df = pd.read_csv(input_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# --------------------------------------------------------
# 3. Feature Engineering (Pre-Split!)
# --------------------------------------------------------

features = pd.DataFrame(index=df.index)

# Returns
features["return_1min"] = df["close"].pct_change(1)
features["return_15min"] = df["close"].pct_change(15)
features["return_60min"] = df["close"].pct_change(60)

# EMAs (Minutenfenster)
features["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
features["ema_15"] = df["close"].ewm(span=15, adjust=False).mean()
features["ema_30"] = df["close"].ewm(span=30, adjust=False).mean()
features["ema_60"] = df["close"].ewm(span=60, adjust=False).mean()

features["ema_5_30_diff"] = features["ema_5"] - features["ema_30"]
features["ema_15_60_diff"] = features["ema_15"] - features["ema_60"]

# Volatilität (1h)
features["volatility_60min"] = df["close"].pct_change().rolling(60).std()

# RSI (Intraday kürzer)
features["rsi_14"] = compute_rsi(df["close"], 14)

# --------------------------------------------------------
# 4. Target: Trend der nächsten Stunde
# --------------------------------------------------------

df["future_mean_1h"] = (
    df["close"]
    .shift(-1)
    .rolling(window=60)
    .mean()
)

df["target_trend_1h"] = (df["future_mean_1h"] > df["close"]).astype(int)

# --------------------------------------------------------
# 5. Zusammenführen & Bereinigen
# --------------------------------------------------------

df_final = pd.concat([df, features], axis=1)
df_final = df_final.dropna().reset_index(drop=True)

# Zukunfts-Infos entfernen (nur fürs Modeling)
df_final = df_final.drop(columns=["future_mean_1h"])

# --------------------------------------------------------
# 6. Speichern
# --------------------------------------------------------

df_final.to_csv(output_path, index=False)
print("✓ Intraday-Datensatz gespeichert:", output_path)


