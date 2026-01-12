"""
04_post_split_preparation.py
----------------------------------------
Post-Split Preparation (Multi-Asset, leak-free)

Fixes:
- Symbol wird als Feature verwendet (One-Hot), sonst sehen Probas oft gleich aus
- Speichert auch meta_test.csv (timestamp,symbol,close,target) für Backtesting/Debug
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
DATA_DIR = os.path.join(ROOT, "data")

input_path = os.path.join(DATA_DIR, "MULTI_prepared_intraday.csv")
print("Lade vorbereitete Multi-Asset-Daten:", input_path)

df = pd.read_csv(input_path)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

# Zeitlich sortieren
df = df.sort_values("timestamp").reset_index(drop=True)

print("Gesamtzeilen:", len(df))
print("\nSymbol-Verteilung (gesamt):")
print(df["symbol"].value_counts())

TARGET_COL = "target_trend_1h"

# -------------------------
# One-Hot Symbol
# -------------------------
symbol_dummies = pd.get_dummies(df["symbol"], prefix="sym", dtype=int)

# Meta für Backtesting/Debug
meta = df[["timestamp", "symbol", "close", TARGET_COL]].copy()

# Features
base_features = df.drop(columns=["timestamp", "symbol", "close", TARGET_COL])
X = pd.concat([base_features, symbol_dummies], axis=1)
y = df[TARGET_COL]

print("\nFeature-Matrix:", X.shape)
print("Target-Vektor:", y.shape)

# -------------------------
# Split (time-based global)
# -------------------------
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]
meta_train = meta.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]
meta_val = meta.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]
meta_test = meta.iloc[val_end:]

print("\nSplit-Größen:")
print("Train:", X_train.shape, " Val:", X_val.shape, " Test:", X_test.shape)

print("\nSymbol-Verteilung (Train):")
print(meta_train["symbol"].value_counts(normalize=True))
print("\nSymbol-Verteilung (Val):")
print(meta_val["symbol"].value_counts(normalize=True))
print("\nSymbol-Verteilung (Test):")
print(meta_test["symbol"].value_counts(normalize=True))

print("\nTarget-Verteilung:")
print("Train:\n", y_train.value_counts(normalize=True))
print("Validation:\n", y_val.value_counts(normalize=True))
print("Test:\n", y_test.value_counts(normalize=True))

# -------------------------
# Scaling (nur auf train fitten)
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val_scaled   = pd.DataFrame(X_val_scaled, columns=X_val.columns)
X_test_scaled  = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# -------------------------
# Save
# -------------------------
X_train_scaled.to_csv(os.path.join(DATA_DIR, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(DATA_DIR, "y_train.csv"), index=False)

X_val_scaled.to_csv(os.path.join(DATA_DIR, "X_val.csv"), index=False)
y_val.to_csv(os.path.join(DATA_DIR, "y_val.csv"), index=False)

X_test_scaled.to_csv(os.path.join(DATA_DIR, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(DATA_DIR, "y_test.csv"), index=False)

meta_test.to_csv(os.path.join(DATA_DIR, "meta_test.csv"), index=False)

scaler_path = os.path.join(DATA_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)

print("\n==============================")
print("✅ Post-Split Data Preparation abgeschlossen!")
print("Scaler gespeichert unter:", scaler_path)
print("meta_test.csv gespeichert (für Backtesting/Debug).")
print("==============================")
