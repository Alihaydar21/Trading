"""
02_data_understanding.py
----------------------------------------
Intraday Data Understanding für 1-Min Bars (AAPL/MSFT/NVDA).

Fixes:
- Vergleichsplot AAPL vs MSFT wird nur 1x am Ende erstellt (nicht 3x)
- robust bei fehlenden Dateien
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

symbols = ["AAPL", "MSFT", "NVDA"]

def load_symbol(symbol: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{symbol}_1min.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fehlt: {path} (erst 01_data_acquisition.py laufen lassen)")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# -------------------------
# Pro Asset Plots
# -------------------------
loaded = {}

for symbol in symbols:
    print("\n==============================")
    print(f"📊 Data Understanding: {symbol}")
    print("==============================")

    df = load_symbol(symbol)
    loaded[symbol] = df

    print("\n=== HEAD ===")
    print(df.head())

    print("\n=== TAIL ===")
    print(df.tail())

    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(df.describe())

    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())

    print("\n=== ZEITRAUM ===")
    print(df["timestamp"].min(), " → ", df["timestamp"].max())

    print("\n=== ANZAHL DER MINUTENBARS ===")
    print(len(df))

    df["return_1min"] = df["close"].pct_change()
    df["return_1h"] = df["close"].pct_change(periods=60)
    df["future_mean_1h"] = df["close"].shift(-1).rolling(window=60).mean()

    # Plot 1: letzter Tag
    example_day = df["timestamp"].dt.date.iloc[-1]
    df_day = df[df["timestamp"].dt.date == example_day]

    plt.figure(figsize=(12, 4))
    plt.plot(df_day["timestamp"], df_day["close"], linewidth=1.2)
    plt.title(f"{symbol} – Intraday-Preisverlauf (1-Min) am {example_day}")
    plt.xlabel("Zeit")
    plt.ylabel("Preis")
    plt.grid(True)
    p1 = os.path.join(IMAGES_DIR, f"{symbol.lower()}_intraday_close_example_day.png")
    plt.savefig(p1, dpi=300); plt.close()
    print("Gespeichert:", p1)

    # Plot 2: 1h returns hist
    returns_1h = df["return_1h"].dropna()
    plt.figure(figsize=(8, 5))
    plt.hist(returns_1h, bins=100, density=True, alpha=0.7)
    plt.axvline(returns_1h.median(), linestyle="--", linewidth=1.2, label="Median")
    plt.axvline(returns_1h.quantile(0.05), linestyle=":", linewidth=1.0, label="5%-Quantil")
    plt.axvline(returns_1h.quantile(0.95), linestyle=":", linewidth=1.0, label="95%-Quantil")
    plt.title(f"{symbol} – Verteilung der 1-Stunden-Renditen")
    plt.xlabel("Rendite (60 Minuten)")
    plt.ylabel("Dichte")
    plt.legend()
    p2 = os.path.join(IMAGES_DIR, f"{symbol.lower()}_1h_returns_hist.png")
    plt.savefig(p2, dpi=300); plt.close()
    print("Gespeichert:", p2)

    # Plot 3: close vs future_mean_1h (letzte 180 Zeilen)
    df_trend_plot = df.dropna().iloc[-180:]
    plt.figure(figsize=(12, 4))
    plt.plot(df_trend_plot["timestamp"], df_trend_plot["close"], label="Aktueller Preis", linewidth=1.2)
    plt.plot(df_trend_plot["timestamp"], df_trend_plot["future_mean_1h"], label="Ø-Preis nächste Stunde", linestyle="--", linewidth=1.2)
    plt.title(f"{symbol} – Preis vs Ø-Preis nächste Stunde (Ausschnitt)")
    plt.xlabel("Zeit"); plt.ylabel("Preis")
    plt.legend(); plt.grid(True)
    p3 = os.path.join(IMAGES_DIR, f"{symbol.lower()}_price_vs_future_mean_1h.png")
    plt.savefig(p3, dpi=300); plt.close()
    print("Gespeichert:", p3)

    # Plot 4: Intraday volume by hour UTC (grob)
    df["hour"] = df["timestamp"].dt.hour
    df_market = df[(df["hour"] >= 14) & (df["hour"] <= 21)]
    volume_by_hour = df_market.groupby("hour")["volume"].mean()

    plt.figure(figsize=(10, 4))
    volume_by_hour.plot(kind="bar")
    plt.title(f"{symbol} – Durchschnittliches Intraday-Volumen (UTC 14–21)")
    plt.xlabel("Stunde (UTC)")
    plt.ylabel("Ø Volumen")
    p4 = os.path.join(IMAGES_DIR, f"{symbol.lower()}_intraday_volume_profile.png")
    plt.savefig(p4, dpi=300); plt.close()
    print("Gespeichert:", p4)

    print(f"\n✓ Intraday Data Understanding für {symbol} abgeschlossen!")

# -------------------------
# Vergleichsplot 1x am Ende
# -------------------------
if "AAPL" in loaded and "MSFT" in loaded:
    print("\n📈 Erzeuge Vergleichsplot: AAPL vs. MSFT")

    df_aapl = loaded["AAPL"][["timestamp", "close"]].copy()
    df_msft = loaded["MSFT"][["timestamp", "close"]].copy()

    start_ts = max(df_aapl["timestamp"].min(), df_msft["timestamp"].min())
    end_ts = min(df_aapl["timestamp"].max(), df_msft["timestamp"].max())

    df_aapl = df_aapl[(df_aapl["timestamp"] >= start_ts) & (df_aapl["timestamp"] <= end_ts)].set_index("timestamp")
    df_msft = df_msft[(df_msft["timestamp"] >= start_ts) & (df_msft["timestamp"] <= end_ts)].set_index("timestamp")

    aapl_norm = df_aapl["close"] / df_aapl["close"].iloc[0] * 100
    msft_norm = df_msft["close"] / df_msft["close"].iloc[0] * 100

    plt.figure(figsize=(12, 5))
    plt.plot(aapl_norm.index, aapl_norm, label="AAPL (norm)", linewidth=1.4)
    plt.plot(msft_norm.index, msft_norm, label="MSFT (norm)", linewidth=1.4)
    plt.title("Vergleich der Preisentwicklung (normalisiert)")
    plt.xlabel("Zeit"); plt.ylabel("Index (Start=100)")
    plt.legend(); plt.grid(True)
    path = os.path.join(IMAGES_DIR, "aapl_vs_msft_price_comparison.png")
    plt.savefig(path, dpi=300); plt.close()
    print("Gespeichert:", path)

print("\n✅ Data Understanding abgeschlossen!")
