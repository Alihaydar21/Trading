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

# --------------------------------------------------------
# 1) ML-Daten laden
# --------------------------------------------------------
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()

X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv")).squeeze()

X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
test_len = len(X_test)

# --------------------------------------------------------
# 2) META für Test laden (timestamp+symbol+close passt 1:1 zu X_test)
# --------------------------------------------------------
meta_path = os.path.join(DATA_DIR, "meta_test.csv")
if not os.path.exists(meta_path):
    raise RuntimeError(
        f"meta_test.csv nicht gefunden unter {meta_path}. "
        "Es wird in 04_post_split_preparation.py gespeichert."
    )

meta_test = pd.read_csv(meta_path)
if len(meta_test) != test_len:
    raise RuntimeError(
        f"meta_test Länge ({len(meta_test)}) != X_test Länge ({test_len}). "
        "Pipeline inkonsistent -> bitte 03/04 neu laufen lassen."
    )

meta_test["timestamp"] = pd.to_datetime(meta_test["timestamp"], utc=True, errors="coerce")
meta_test["close"] = pd.to_numeric(meta_test["close"], errors="coerce")
meta_test = meta_test.dropna(subset=["timestamp", "close", "symbol"]).reset_index(drop=True)

# --------------------------------------------------------
# 3) Modell trainieren (Train + Val)
# --------------------------------------------------------
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

# --------------------------------------------------------
# 4) Probas auf TEST
# --------------------------------------------------------
proba_up = rf.predict_proba(X_test)[:, 1]

print("\n=== p_up distribution on TEST ===")
print(pd.Series(proba_up).describe(percentiles=[0.9, 0.95, 0.99]))
print("max p_up:", float(np.max(proba_up)))

# --------------------------------------------------------
# 5) Test-DF sauber zusammenbauen (ALIGNMENT FIX!)
# --------------------------------------------------------
df_test = meta_test.copy()
df_test["p_up"] = proba_up
df_test = df_test.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

HOLD_MINUTES = 60

#FIX Threshold
THRESHOLD = 0.60

#

print("Using threshold:", THRESHOLD)

# --------------------------------------------------------
# 6) Future Close (60m) pro Symbol, korrekt!
# --------------------------------------------------------
df_test["future_close_60m"] = df_test.groupby("symbol")["close"].shift(-HOLD_MINUTES)

# Trade-Return (punktuell, 60m ahead) – korrekt pro Symbol
df_test["trade_return"] = df_test["future_close_60m"] / df_test["close"] - 1.0

# Signal
df_test["signal"] = (df_test["p_up"] >= THRESHOLD).astype(int)

# Nur valide Zeilen (wo future existiert)
df_test_valid = df_test.dropna(subset=["future_close_60m", "trade_return"]).reset_index(drop=True)

# --------------------------------------------------------
# 7) Trades ohne Overlap (PRO SYMBOL!)
# --------------------------------------------------------
trades = []
for sym, g in df_test_valid.groupby("symbol", sort=False):
    g = g.sort_values("timestamp").reset_index(drop=True)

    in_pos = False
    entry_i = None

    for i in range(len(g)):
        if not in_pos:
            if int(g.loc[i, "signal"]) == 1:
                in_pos = True
                entry_i = i
        else:
            # Exit exakt nach 60 Minuten = entry_i + HOLD_MINUTES
            exit_i = entry_i + HOLD_MINUTES
            if exit_i >= len(g):
                break

            # Nur schließen, wenn wir wirklich am Exit-Punkt angekommen sind
            if i >= exit_i:
                trades.append({
                    "symbol": sym,
                    "entry_time": g.loc[entry_i, "timestamp"],
                    "exit_time": g.loc[exit_i, "timestamp"],
                    "entry_price": float(g.loc[entry_i, "close"]),
                    "exit_price": float(g.loc[exit_i, "close"]),
                    "p_up_entry": float(g.loc[entry_i, "p_up"]),
                    "trade_return": float(g.loc[entry_i, "trade_return"])  # Return ist entry->+60m
                })

                in_pos = False
                entry_i = None

trades = pd.DataFrame(trades)

# --------------------------------------------------------
# 8) Results
# --------------------------------------------------------
print("\n================ BACKTEST RESULT =================")
print("Anzahl Trades:", len(trades))
if len(trades) > 0:
    hit_rate = (trades["trade_return"] > 0).mean()
    avg_return = trades["trade_return"].mean()
    cum_return = trades["trade_return"].sum()

    print("Trefferquote:", round(float(hit_rate), 3))
    print("Ø Trade-Return:", round(float(avg_return), 6))
    print("Kumulierter Signal-Return:", round(float(cum_return), 6))

    # Optional: Return clipping für robustere Statistik (nicht fürs echte PnL!)
    # trades["trade_return_clipped"] = trades["trade_return"].clip(-0.05, 0.05)

else:
    print("Keine Trades -> Threshold zu hoch oder p_up zu eng.")
print("=================================================")

# --------------------------------------------------------
# 9) Plots nur wenn Trades existieren
# --------------------------------------------------------
if len(trades) > 0:
    trades = trades.sort_values("exit_time").reset_index(drop=True)
    trades["equity"] = trades["trade_return"].cumsum()

    plt.figure(figsize=(10, 4))
    plt.plot(trades["exit_time"], trades["equity"])
    plt.title("Equity Curve – ML Trading Strategie (signal-basiert)")
    plt.xlabel("Zeit")
    plt.ylabel("Kumulierter Signal-Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "equity_curve_ml_strategy.png"), dpi=300)
    plt.close()

    # Trades pro Monat (nutze entry_time oder exit_time – hier exit_time)
    trades["month"] = pd.to_datetime(trades["exit_time"], utc=True).dt.to_period("M").astype(str)
    trade_counts_month = trades.groupby("month").size()

    plt.figure(figsize=(12, 4))
    trade_counts_month.plot(kind="bar")
    plt.title("Anzahl Trades pro Monat")
    plt.xlabel("Monat")
    plt.ylabel("Trades")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "trade_distribution_per_month.png"), dpi=300)
    plt.close()

    print("\n✅ Plots gespeichert in:", IMAGES_DIR)
else:
    print("\n[INFO] Keine Trades -> überspringe Plots.")
