"""
08_deployment.py
----------------------------------------
Paper Trading / Deployment Simulation
Relative Features, Multi-Asset

Simulation:
- sequenzielle Vorhersagen
- Long-only
- Entry: p_up >= 0.6
- Exit: nach 60 Minuten
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------------
# 1. Pfade
# --------------------------------------------------------
TRADE_ASSETS = ["NVDA"]

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))

DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "paper_trading")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------
# 2. ML-Daten laden
# --------------------------------------------------------

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()

X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv")).squeeze()

X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

# --------------------------------------------------------
# 3. Rohpreise laden
# --------------------------------------------------------

price_frames = []

for symbol in ["AAPL", "MSFT", "NVDA"]:
    df_price = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}_1min.csv"))
    df_price["timestamp"] = pd.to_datetime(df_price["timestamp"])
    df_price["symbol"] = symbol
    price_frames.append(df_price[["timestamp", "symbol", "close"]])

df_prices = (
    pd.concat(price_frames)
    .sort_values("timestamp")
    .reset_index(drop=True)
)

# --------------------------------------------------------
# 4. Modell trainieren (Train + Val)
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
# 5. Paper-Trading-Zeitraum definieren
# --------------------------------------------------------

PAPER_WINDOW = 5 * 390   # ca. 5 Handelstage (US-Markt)

X_paper = X_test.iloc[-PAPER_WINDOW:].reset_index(drop=True)
df_price_paper = df_prices.iloc[-PAPER_WINDOW:].reset_index(drop=True)

# --------------------------------------------------------
# 6. Sequentielle Vorhersage & Trades
# --------------------------------------------------------

records = []

for i in range(len(X_paper) - 60):

    x_now = X_paper.iloc[[i]]
    p_up = rf.predict_proba(x_now)[0, 1]

    symbol = df_price_paper.loc[i, "symbol"]

    if (symbol in TRADE_ASSETS) and (p_up >= 0.6):
        entry_price = df_price_paper.loc[i, "close"]
        exit_price = df_price_paper.loc[i + 60, "close"]

        trade_return = exit_price / entry_price - 1
        trade_return = np.clip(trade_return, -0.05, 0.05)

        records.append({
            "timestamp": df_price_paper.loc[i, "timestamp"],
            "symbol": df_price_paper.loc[i, "symbol"],
            "p_up": p_up,
            "trade_return": trade_return
        })

paper_trades = pd.DataFrame(records)

# --------------------------------------------------------
# 7. Auswertung
# --------------------------------------------------------

paper_trades["date"] = paper_trades["timestamp"].dt.date

summary_overall = {
    "num_trades": len(paper_trades),
    "hit_rate": (paper_trades["trade_return"] > 0).mean(),
    "avg_return": paper_trades["trade_return"].mean(),
    "cum_signal_return": paper_trades["trade_return"].sum()
}

summary_by_asset = (
    paper_trades
    .groupby("symbol")["trade_return"]
    .agg(["count", "mean", "sum"])
)

summary_by_day = (
    paper_trades
    .groupby("date")["trade_return"]
    .sum()
)

# --------------------------------------------------------
# 8. Speichern
# --------------------------------------------------------

paper_trades.to_csv(
    os.path.join(OUTPUT_DIR, "paper_trades.csv"),
    index=False
)

summary_by_asset.to_csv(
    os.path.join(OUTPUT_DIR, "paper_summary_by_asset.csv")
)

summary_by_day.to_csv(
    os.path.join(OUTPUT_DIR, "paper_summary_by_day.csv")
)


print("\n============= PAPER TRADING RESULT =============")
for k, v in summary_overall.items():
    print(f"{k}: {round(v, 4)}")
print("\n--- Performance pro Asset ---")
print(summary_by_asset)
print("\n✓ Paper Trading abgeschlossen")


