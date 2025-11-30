"""
02_data_understanding.py
----------------------------
Data Understanding für tägliche Kursdaten (Daily Bars).

Dieses Skript:
- lädt die gespeicherten Daily-Daten aus dem /data Ordner
- berechnet deskriptive Statistiken
- prüft Missing Values
- zeigt Zeitraum und Anzahl der Datenpunkte
- berechnet Tagesrenditen
- erstellt Plots und speichert sie im Ordner /images

Plots:
- Close Price Verlauf
- Volumen Verlauf
- Histogramm der Tagesrenditen
"""


import os
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("Agg")

# ------------------------------------------------------------
# 1. Pfade korrekt setzen (automatisch, egal wo ausgeführt wird)
# ------------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))   # /Trade/scripts
PROJECT_ROOT = os.path.abspath(os.path.join(BASE, ".."))  # /Trade

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")

# Ordner für Plots erstellen
os.makedirs(IMAGES_DIR, exist_ok=True)

# Pfad zur CSV-Datei
csv_path = os.path.join(DATA_DIR, "AAPL_daily.csv")
print("Lade Datei:", csv_path)


# ------------------------------------------------------------
# 2. Daten laden
# ------------------------------------------------------------

df = pd.read_csv(csv_path)

# Datentypen bereinigen
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

print("\n=== HEAD ===")
print(df.head())

print("\n=== TAIL ===")
print(df.tail())


# ------------------------------------------------------------
# 3. Deskriptive Statistik
# ------------------------------------------------------------

print("\n=== DESCRIPTIVE STATISTICS ===")
print(df.describe())

print("\n=== MISSING VALUES ===")
print(df.isna().sum())

print("\n=== DATUMSBEREICH ===")
print(df["timestamp"].min(), " → ", df["timestamp"].max())

print("\n=== ANZAHL DER TAGE ===")
print(len(df))


# ------------------------------------------------------------
# 4. Tagesrenditen berechnen
# ------------------------------------------------------------

df["return_1d"] = df["close"].pct_change()


# ------------------------------------------------------------
# 5. Plots erstellen + speichern
# ------------------------------------------------------------

# ---- Plot 1: Close Price ----
plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"], df["close"], linewidth=1.2)
plt.title("AAPL – Daily Close Price")
plt.xlabel("Zeit")
plt.ylabel("Close Price")
plt.grid(True)

plot_path = os.path.join(IMAGES_DIR, "aapl_close_price.png")
plt.savefig(plot_path, dpi=300)
plt.show()
print("Gespeichert:", plot_path)


# ---- Plot 2: Volume ----
plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"], df["volume"], color="orange", linewidth=1.0)
plt.title("AAPL – Daily Volume")
plt.xlabel("Zeit")
plt.ylabel("Volume")
plt.grid(True)

plot_path = os.path.join(IMAGES_DIR, "aapl_volume.png")
plt.savefig(plot_path, dpi=300)
plt.show()
print("Gespeichert:", plot_path)


# ---- Plot 3: Histogram der Tagesrenditen ----
plt.figure(figsize=(8, 5))
returns = df["return_1d"].clip(lower=-0.1, upper=0.1)
plt.hist(returns, bins=50)
plt.title("Histogram – Daily Returns")
plt.xlabel("Rendite")
plt.ylabel("Häufigkeit")
plt.grid(False)

plot_path = os.path.join(IMAGES_DIR, "aapl_returns_hist.png")
plt.savefig(plot_path, dpi=300)
plt.show()
print("Gespeichert:", plot_path)

print("\n✓ Data Understanding abgeschlossen! Plots liegen im Ordner /images.")
