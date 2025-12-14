"""
07_backtesting.py
----------------------------------------
Backtesting eines ML-basierten Trading-Ansatzes
Relative Features, Multi-Asset

Strategie:
- Long-only
- Entry: p_up >= 0.6
- Exit: nach 60 Minuten

Hinweis:
- Additives Backtesting (keine Reinvestition)
- Ziel: Bewertung der Signalqualität
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------------
# 1. Pfade
# --------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))

DATA_DIR = os.path.join(ROOT, "data")
IMAGES_DIR = os.path.join(ROOT, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# --------------------------------------------------------
# 2. ML-Daten laden (preisfrei!)
# --------------------------------------------------------

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()

X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv")).squeeze()

X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
test_len = len(X_test)

# --------------------------------------------------------
# 3. Rohpreise laden (NUR für Backtesting!)
# --------------------------------------------------------

price_frames = []

for symbol in ["AAPL", "MSFT", "NVDA"]:
    path = os.path.join(DATA_DIR, f"{symbol}_1min.csv")
    df_price = pd.read_csv(path)

    df_price["timestamp"] = pd.to_datetime(df_price["timestamp"])
    df_price = df_price.sort_values("timestamp")
    df_price["symbol"] = symbol

    price_frames.append(df_price[["timestamp", "symbol", "close"]])

df_prices = pd.concat(price_frames).reset_index(drop=True)
df_prices = df_prices.sort_values("timestamp").reset_index(drop=True)

df_test_prices = df_prices.iloc[-test_len:].reset_index(drop=True)

# --------------------------------------------------------
# 4. Modell neu trainieren (Train + Validation)
# --------------------------------------------------------

X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=200,
    max_samples=0.3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_full, y_train_full)

# --------------------------------------------------------
# 5. ML-Signale erzeugen
# --------------------------------------------------------

proba_up = rf.predict_proba(X_test)[:, 1]

df_test = pd.DataFrame({
    "timestamp": df_test_prices["timestamp"].values,
    "symbol": df_test_prices["symbol"].values,
    "close": df_test_prices["close"].values,
    "p_up": proba_up
})

# --------------------------------------------------------
# 6. Trading-Logik
# --------------------------------------------------------

THRESHOLD = 0.6
HOLD_MINUTES = 60

df_test["signal"] = (df_test["p_up"] >= THRESHOLD).astype(int)

df_test["future_close_60m"] = df_test["close"].shift(-HOLD_MINUTES)
df_test["trade_return"] = (
    df_test["future_close_60m"] / df_test["close"] - 1
)

# Nur Trades berücksichtigen
trades = df_test[df_test["signal"] == 1].copy()
trades = trades.dropna(subset=["trade_return"])

# --------------------------------------------------------
# 7. Performance-Kennzahlen (Signal-basiert!)
# --------------------------------------------------------

# Realistische Begrenzung extremer Returns
trades["trade_return"] = trades["trade_return"].clip(-0.05, 0.05)

num_trades = len(trades)
hit_rate = (trades["trade_return"] > 0).mean()
avg_return = trades["trade_return"].mean()
cum_signal_return = trades["trade_return"].sum()

print("\n================ BACKTEST RESULT =================")
print("Anzahl Trades:", num_trades)
print("Trefferquote:", round(hit_rate, 3))
print("Ø Trade-Return:", round(avg_return, 5))
print("Kumulierter Signal-Return:", round(cum_signal_return, 3))
print("=================================================")

# --------------------------------------------------------
# 8. Equity Curve (ADDITIV!)
# --------------------------------------------------------

trades["equity"] = trades["trade_return"].cumsum()

plt.figure(figsize=(10, 4))
plt.plot(trades["timestamp"], trades["equity"], label="ML-Signal-Equity")
plt.title("Equity Curve – ML Trading Strategie")
plt.xlabel("Zeit")
plt.ylabel("Kumulierter Signal-Return")
plt.grid(True)
plt.legend()

path = os.path.join(IMAGES_DIR, "equity_curve_ml_strategy.png")
plt.savefig(path, dpi=300)
plt.close()


# --------------------------------------------------------
# 9. Trade-Verteilung über Zeit
# --------------------------------------------------------

trades["month"] = trades["timestamp"].dt.to_period("M").astype(str)
trade_counts_month = trades.groupby("month").size()

plt.figure(figsize=(10, 4))
trade_counts_month.plot(kind="bar")
plt.title("Anzahl Trades pro Monat")
plt.xlabel("Monat")
plt.ylabel("Trades")
plt.grid(True)

plt.tight_layout()

path = os.path.join(IMAGES_DIR, "trade_distribution_per_month.png")
plt.savefig(path, dpi=300)
plt.close()

