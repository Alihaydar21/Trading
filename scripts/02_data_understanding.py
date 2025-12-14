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

# ------------------------------------------------------------
# 2. Assets
# ------------------------------------------------------------

symbols = ["AAPL", "MSFT", "NVDA"]

# ------------------------------------------------------------
# 3. Loop über alle Assets
# ------------------------------------------------------------

for symbol in symbols:

    print("\n==============================")
    print(f"📊 Data Understanding: {symbol}")
    print("==============================")

    csv_path = os.path.join(DATA_DIR, f"{symbol}_1min.csv")
    print("Lade Datei:", csv_path)

    # --------------------------------------------------------
    # 3.1 Daten laden & vorbereiten
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 3.2 Deskriptive Statistik
    # --------------------------------------------------------

    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(df.describe())

    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())

    print("\n=== ZEITRAUM ===")
    print(df["timestamp"].min(), " → ", df["timestamp"].max())

    print("\n=== ANZAHL DER MINUTENBARS ===")
    print(len(df))

    # --------------------------------------------------------
    # 3.3 Renditen & Zielgrößen
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 3.4 Plots – abgestimmt auf 1-Stunden-Trend
    # --------------------------------------------------------

    # ========== Plot 1: Intraday Close (letzter Handelstag) ==========
    example_day = df["timestamp"].dt.date.iloc[-1]
    df_day = df[df["timestamp"].dt.date == example_day]

    plt.figure(figsize=(12, 4))
    plt.plot(df_day["timestamp"], df_day["close"], linewidth=1.2)
    plt.title(f"{symbol} – Intraday-Preisverlauf (1-Min) am {example_day}")
    plt.xlabel("Zeit")
    plt.ylabel("Preis")
    plt.grid(True)

    path = os.path.join(
        IMAGES_DIR,
        f"{symbol.lower()}_intraday_close_example_day.png"
    )
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

    plt.title(f"{symbol} – Verteilung der 1-Stunden-Renditen (Dichte)")
    plt.xlabel("Rendite (60 Minuten)")
    plt.ylabel("Dichte")
    plt.legend()

    path = os.path.join(
        IMAGES_DIR,
        f"{symbol.lower()}_1h_returns_hist.png"
    )
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

    plt.title(f"{symbol} – Trend-Definition: Aktueller Preis vs. Ø-Preis nächste Stunde")
    plt.xlabel("Zeit")
    plt.ylabel("Preis")
    plt.legend()
    plt.grid(True)

    path = os.path.join(
        IMAGES_DIR,
        f"{symbol.lower()}_price_vs_future_mean_1h.png"
    )
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
    plt.title(f"{symbol} – Durchschnittliches Intraday-Volumen (Handelszeiten)")
    plt.xlabel("Stunde (UTC)")
    plt.ylabel("Ø Volumen")

    path = os.path.join(
        IMAGES_DIR,
        f"{symbol.lower()}_intraday_volume_profile.png"
    )
    plt.savefig(path, dpi=300)
    plt.close()
    print("Gespeichert:", path)

    print(f"\n✓ Intraday Data Understanding für {symbol} abgeschlossen!")

    print("\n📈 Erzeuge Vergleichsplot: AAPL vs. MSFT")

    # Dateien laden
    aapl_path = os.path.join(DATA_DIR, "AAPL_1min.csv")
    msft_path = os.path.join(DATA_DIR, "MSFT_1min.csv")

    df_aapl = pd.read_csv(aapl_path)
    df_msft = pd.read_csv(msft_path)

    df_aapl["timestamp"] = pd.to_datetime(df_aapl["timestamp"])
    df_msft["timestamp"] = pd.to_datetime(df_msft["timestamp"])

    # Zeitlich schneiden (gemeinsamer Zeitraum!)
    start_ts = max(df_aapl["timestamp"].min(), df_msft["timestamp"].min())
    end_ts = min(df_aapl["timestamp"].max(), df_msft["timestamp"].max())

    df_aapl = df_aapl[(df_aapl["timestamp"] >= start_ts) & (df_aapl["timestamp"] <= end_ts)]
    df_msft = df_msft[(df_msft["timestamp"] >= start_ts) & (df_msft["timestamp"] <= end_ts)]

    # Index auf Timestamp
    df_aapl = df_aapl.set_index("timestamp")
    df_msft = df_msft.set_index("timestamp")

    # Normalisierung (Start = 100)
    aapl_norm = df_aapl["close"] / df_aapl["close"].iloc[0] * 100
    msft_norm = df_msft["close"] / df_msft["close"].iloc[0] * 100

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(aapl_norm.index, aapl_norm, label="AAPL (normalisiert)", linewidth=1.4)
    plt.plot(msft_norm.index, msft_norm, label="MSFT (normalisiert)", linewidth=1.4)

    plt.title("Vergleich der Intraday-Preisentwicklung (normalisiert)")
    plt.xlabel("Zeit")
    plt.ylabel("Index (Start = 100)")
    plt.legend()
    plt.grid(True)

    path = os.path.join(IMAGES_DIR, "aapl_vs_msft_price_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print("Gespeichert:", path)

print("\n✅ Data Understanding für alle Assets abgeschlossen!")

# ============================================================
# 4. Vergleichsplot: AAPL vs. MSFT – Preisentwicklung
# ============================================================

