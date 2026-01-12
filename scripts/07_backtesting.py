import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
DATA_DIR = os.path.join(ROOT, "data")
IMAGES_DIR = os.path.join(ROOT, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# 1) ML-Daten laden
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()

X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv")).squeeze()

X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
test_len = len(X_test)

# 2) Rohpreise laden (nur fürs Backtesting)
price_frames = []
for symbol in ["AAPL", "MSFT", "NVDA"]:
    path = os.path.join(DATA_DIR, f"{symbol}_1min.csv")
    df_price = pd.read_csv(path)
    df_price["timestamp"] = pd.to_datetime(df_price["timestamp"], utc=True, errors="coerce")
    df_price["close"] = pd.to_numeric(df_price["close"], errors="coerce")
    df_price = df_price.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    df_price["symbol"] = symbol
    price_frames.append(df_price[["timestamp", "symbol", "close"]])

df_prices = pd.concat(price_frames).sort_values("timestamp").reset_index(drop=True)

# WICHTIG: wir nehmen die letzten test_len Zeilen als Test-Preise (wie bei dir)
df_test_prices = df_prices.iloc[-test_len:].reset_index(drop=True)

# 3) Modell trainieren (Train + Val)
X_train_full = pd.concat([X_train, X_val], ignore_index=True)
y_train_full = pd.concat([y_train, y_val], ignore_index=True)

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

# 4) Probas
proba_up = rf.predict_proba(X_test)[:, 1]

print("\n=== p_up distribution on TEST ===")
print(pd.Series(proba_up).describe(percentiles=[0.9, 0.95, 0.99]))
print("max p_up:", float(np.max(proba_up)))

# 5) Test-DF
df_test = pd.DataFrame({
    "timestamp": df_test_prices["timestamp"].values,
    "symbol": df_test_prices["symbol"].values,
    "close": df_test_prices["close"].values,
    "p_up": proba_up
}).sort_values("timestamp").reset_index(drop=True)

HOLD_MINUTES = 60

# ✅ Option A: FIX Threshold (z.B. 0.512)
THRESHOLD = 0.60

# ✅ Option B: Auto-threshold (z.B. Top 5% stärkste Signale)
# THRESHOLD = float(np.quantile(df_test["p_up"].values, 0.95))

print("Using threshold:", THRESHOLD)

# 6) Position-Backtest (keine Overlaps)
in_pos = False
entry_idx = None
entry_price = None
entry_time = None

trades = []

for i in range(len(df_test) - HOLD_MINUTES):
    p = float(df_test.loc[i, "p_up"])
    price = float(df_test.loc[i, "close"])
    ts = df_test.loc[i, "timestamp"]
    sym = df_test.loc[i, "symbol"]

    if not in_pos:
        if p >= THRESHOLD:
            in_pos = True
            entry_idx = i
            entry_price = price
            entry_time = ts
            entry_sym = sym
    else:
        # Exit nach HOLD_MINUTES
        if i - entry_idx >= HOLD_MINUTES:
            exit_price = float(df_test.loc[i, "close"])
            exit_time = df_test.loc[i, "timestamp"]

            ret = exit_price / entry_price - 1.0

            trades.append({
                "symbol": entry_sym,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "p_up_entry": float(df_test.loc[entry_idx, "p_up"]),
                "trade_return": ret
            })

            in_pos = False
            entry_idx = None
            entry_price = None
            entry_time = None

trades = pd.DataFrame(trades)

print("\n================ BACKTEST RESULT =================")
print("Anzahl Trades:", len(trades))
if len(trades) > 0:
    hit_rate = (trades["trade_return"] > 0).mean()
    avg_return = trades["trade_return"].mean()
    cum_return = trades["trade_return"].sum()

    print("Trefferquote:", round(float(hit_rate), 3))
    print("Ø Trade-Return:", round(float(avg_return), 5))
    print("Kumulierter Signal-Return:", round(float(cum_return), 3))
else:
    print("Keine Trades -> Threshold zu hoch oder p_up komprimiert.")
print("=================================================")

# 7) Plots nur wenn Trades existieren
if len(trades) > 0:
    trades["equity"] = trades["trade_return"].cumsum()

    plt.figure(figsize=(10, 4))
    plt.plot(trades["exit_time"], trades["equity"])
    plt.title("Equity Curve – ML Trading Strategie")
    plt.xlabel("Zeit")
    plt.ylabel("Kumulierter Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "equity_curve_ml_strategy.png"), dpi=300)
    plt.close()

    # --------------------------------------------------------
    # 9. Trade-Verteilung über Zeit (nur wenn Trades existieren!)
    # --------------------------------------------------------
    if len(trades) == 0:
        print("[INFO] Keine Trades -> überspringe Plots.")
    else:
        trades["month"] = pd.to_datetime(trades["timestamp"]).dt.to_period("M").astype(str)
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

