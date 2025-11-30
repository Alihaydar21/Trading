"""
03_data_preparation.py
----------------------------------------
Vollständige Datenaufbereitung in EINEM Skript.

Dieses Skript:
1. Lädt die Daily-Rohdaten
2. Berechnet technische Features:
    - Tagesrenditen
    - EMAs
    - EMA-Differenzen
    - Volatilität
    - RSI14
    - z-Normalisierung
3. Erzeugt die Target-Variable (steigt der Kurs morgen?)
4. Bereinigt NaN-Werte
5. Speichert das vollständig vorbereitete Dataset
6. Erzeugt Heatmap + Target-Verteilung und speichert sie im /images Ordner
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# --------------------------------------------------------
# Hilfsfunktionen (Feature Engineering)
# --------------------------------------------------------

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def z_norm(series, window=50):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / (std + 1e-12)


# --------------------------------------------------------
# 1. Pfade vorbereiten
# --------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))       # /Trade/scripts
ROOT = os.path.abspath(os.path.join(BASE, ".."))        # /Trade
DATA_DIR = os.path.join(ROOT, "data")
IMG_DIR = os.path.join(ROOT, "images")

os.makedirs(IMG_DIR, exist_ok=True)

input_path = os.path.join(DATA_DIR, "AAPL_daily.csv")
output_path = os.path.join(DATA_DIR, "AAPL_prepared.csv")

print("Lade Datei:", input_path)


# --------------------------------------------------------
# 2. Daten laden
# --------------------------------------------------------

df = pd.read_csv(input_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("\n=== Rohdaten ===")
print(df.head())


# --------------------------------------------------------
# 3. Feature Engineering
# --------------------------------------------------------

features = pd.DataFrame(index=df.index)

# Renditen
features["return_1d"] = df["close"].pct_change(1)
features["return_3d"] = df["close"].pct_change(3)
features["return_5d"] = df["close"].pct_change(5)
features["return_10d"] = df["close"].pct_change(10)

# EMAs
features["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
features["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
features["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
features["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

# EMA-Differenzen
features["ema_5_20_diff"] = features["ema_5"] - features["ema_20"]
features["ema_10_50_diff"] = features["ema_10"] - features["ema_50"]

# Volatilität
features["volatility_10"] = df["close"].pct_change().rolling(10).std()
features["volatility_20"] = df["close"].pct_change().rolling(20).std()

# RSI
features["rsi14"] = compute_rsi(df["close"], 14)

# z-Normalisierung
for col in features.columns:
    features[col] = z_norm(features[col], window=50)


# --------------------------------------------------------
# 4. Target generieren (morgen steigt?)
# --------------------------------------------------------

df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

# Letzte Zeile entfernen (kein Target möglich)
df = df.iloc[:-1]
features = features.iloc[:-1]


# --------------------------------------------------------
# 5. Feature-Daten zusammenführen + NaN entfernen
# --------------------------------------------------------

df_final = pd.concat([df, features], axis=1)
df_final = df_final.dropna().reset_index(drop=True)

print("\n=== Finale Daten Vorschau ===")
print(df_final.head())


# --------------------------------------------------------
# 6. Speichern
# --------------------------------------------------------

df_final.to_csv(output_path, index=False)
print("\nDatensatz gespeichert unter:", output_path)


# --------------------------------------------------------
# 7. Heatmap speichern
# --------------------------------------------------------

corr = df_final[features.columns.tolist() + ["target"]].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature-Korrelationen (Daily)")
heatmap_path = os.path.join(IMG_DIR, "correlations_daily.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()

print("Heatmap gespeichert unter:", heatmap_path)


# --------------------------------------------------------
# 8. Target-Verteilung speichern
# --------------------------------------------------------

plt.figure(figsize=(6, 4))
df_final["target"].value_counts().plot(kind="bar", color=["red", "green"])
plt.title("Target-Verteilung (0 = fällt, 1 = steigt)")
plt.xticks([0, 1], ["0", "1"])
target_plot_path = os.path.join(IMG_DIR, "target_distribution.png")
plt.savefig(target_plot_path, dpi=300)
plt.close()

print("Target-Verteilung gespeichert unter:", target_plot_path)


print("\n✓ Datenaufbereitung abgeschlossen!")
