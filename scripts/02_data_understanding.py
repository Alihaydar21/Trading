"""
02_data_understanding.py
----------------------------
Data Understanding für Intraday-Kursdaten (1-Minute Bars)
mit Fokus auf den Trend der nächsten Stunde (60 Minuten).

Dieses Skript:
- lädt gespeicherte 1-Minuten-Daten aus dem /data Ordner
- berechnet deskriptive Statistiken
- prüft Missing Values
- zeigt Zeitraum und Anzahl der Minutenbars
- berechnet 1-Minuten- und 1-Stunden-Renditen
- erstellt zielkonforme Intraday-Plots
- speichert alle Plots im Ordner /images
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ------------------------------------------------------------
# 1. Pfade
# ------------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

csv_path = os.path.join(DATA_DIR, "AAPL_1min.csv")
print("Lade Datei:", csv_path)

# ------------------------------------------------------------
# 2. Daten laden & vorbereiten
# ------------------------------------------------------------

df = pd.read_csv(csv_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

numeric_cols = ["open", "high", "low", "close", "volume", "trade_count"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

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

print("\n=== ZEITRAUM ===")
print(df["timestamp"].min(), " → ", df["timestamp"].max())

print("\n=== ANZAHL DER MINUTENBARS ===")
print(len(df))

# ------------------------------------------------------------
# 4. Renditen & Zielgrößen
# ------------------------------------------------------------

# 1-Minuten-Rendite
df["return_1min"] = df["close"].pct_change()

# 1-Stunden-Rendite (60 Minuten)
df["return_1h"] = df["close"].pct_change(periods=60)

# Ø-Preis der nächsten Stunde (nur zur Visualisierung!)
df["future_mean_1h"] = (
    df["close"]
    .shift(-1)
    .rolling(window=60)
    .mean()
)

# ------------------------------------------------------------
# 5. Plots – abgestimmt auf 1-Stunden-Trend
# ------------------------------------------------------------

# ========== Plot 1: Intraday Close (letzter Handelstag) ==========
example_day = df["timestamp"].dt.date.iloc[-1]
df_day = df[df["timestamp"].dt.date == example_day]

plt.figure(figsize=(12, 4))
plt.plot(df_day["timestamp"], df_day["close"], linewidth=1.2)
plt.title(f"AAPL – Intraday-Preisverlauf (1-Min) am {example_day}")
plt.xlabel("Zeit")
plt.ylabel("Preis")
plt.grid(True)

path = os.path.join(IMAGES_DIR, "aapl_intraday_close_example_day.png")
plt.savefig(path, dpi=300)
plt.close()
print("Gespeichert:", path)


# ========== Plot 2: Dichte – 1-Stunden-Renditen ==========
returns_1h = df["return_1h"].dropna()

plt.figure(figsize=(8, 5))
plt.hist(returns_1h, bins=100, density=True, alpha=0.7)

plt.axvline(returns_1h.median(), linestyle="--", linewidth=1.2, label="Median")
plt.axvline(returns_1h.quantile(0.05), linestyle=":", linewidth=1.0, label="5%-Quantil")
plt.axvline(returns_1h.quantile(0.95), linestyle=":", linewidth=1.0, label="95%-Quantil")

plt.title("Verteilung der 1-Stunden-Renditen (Dichte)")
plt.xlabel("Rendite (60 Minuten)")
plt.ylabel("Dichte")
plt.legend()

path = os.path.join(IMAGES_DIR, "aapl_1h_returns_hist.png")
plt.savefig(path, dpi=300)
plt.close()
print("Gespeichert:", path)


# ========== Plot 3: Preis vs. Ø-Preis der nächsten Stunde ==========
# kurzer, lokaler Ausschnitt (keine Linien über Nacht!)
df_trend_plot = df.dropna().iloc[-180:]  # ca. 3 Handelsstunden

plt.figure(figsize=(12, 4))
plt.plot(
    df_trend_plot["timestamp"],
    df_trend_plot["close"],
    label="Aktueller Preis",
    linewidth=1.2
)
plt.plot(
    df_trend_plot["timestamp"],
    df_trend_plot["future_mean_1h"],
    label="Ø-Preis nächste Stunde",
    linestyle="--",
    linewidth=1.2
)

plt.title("Trend-Definition: Aktueller Preis vs. Ø-Preis der nächsten Stunde")
plt.xlabel("Zeit")
plt.ylabel("Preis")
plt.legend()
plt.grid(True)

path = os.path.join(IMAGES_DIR, "aapl_price_vs_future_mean_1h.png")
plt.savefig(path, dpi=300)
plt.close()
print("Gespeichert:", path)


# ========== Plot 4: Intraday-Volumenprofil (nur Handelszeiten) ==========
# US-Handelszeiten in UTC ca. 14–21 Uhr
df["hour"] = df["timestamp"].dt.hour
df_market = df[(df["hour"] >= 14) & (df["hour"] <= 21)]

volume_by_hour = df_market.groupby("hour")["volume"].mean()

plt.figure(figsize=(10, 4))
volume_by_hour.plot(kind="bar")
plt.title("AAPL – Durchschnittliches Intraday-Volumen (Handelszeiten)")
plt.xlabel("Stunde (UTC)")
plt.ylabel("Ø Volumen")

path = os.path.join(IMAGES_DIR, "aapl_intraday_volume_profile.png")
plt.savefig(path, dpi=300)
plt.close()
print("Gespeichert:", path)

print("\n✓ Intraday Data Understanding (Trend der nächsten Stunde) abgeschlossen!")
