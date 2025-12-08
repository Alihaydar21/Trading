"""
04_post_split_preparation.py
----------------------------------------
Post-Split Data Preparation für Intraday-ML-Projekt

Dieses Skript:
- lädt das vorbereitete Intraday-Dataset
- führt einen zeitkonformen Train/Val/Test-Split durch
- skaliert Features OHNE Data Leakage
- speichert getrennte Datasets für Modeling
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------
# 1. Pfade
# --------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))

DATA_DIR = os.path.join(ROOT, "data")

input_path = os.path.join(DATA_DIR, "AAPL_prepared_intraday.csv")

print("Lade vorbereitete Daten:", input_path)

# --------------------------------------------------------
# 2. Daten laden
# --------------------------------------------------------

df = pd.read_csv(input_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

# --------------------------------------------------------
# 3. Feature / Target Trennung
# --------------------------------------------------------

TARGET_COL = "target_trend_1h"
DROP_COLS = ["timestamp", "symbol", TARGET_COL]

X = df.drop(columns=DROP_COLS)
y = df[TARGET_COL]

# --------------------------------------------------------
# 4. Zeitreihen-Split
# --------------------------------------------------------

n = len(df)

train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

print("\nSplit-Größen:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# --------------------------------------------------------
# 5. Scaling (nur Train fitten!)
# --------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Zurück in DataFrames (gut für Debugging)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val_scaled   = pd.DataFrame(X_val_scaled, columns=X_val.columns)
X_test_scaled  = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# --------------------------------------------------------
# 6. Target-Verteilung prüfen (wichtig!)
# --------------------------------------------------------

print("\nTarget-Verteilung:")
print("Train:\n", y_train.value_counts(normalize=True))
print("Validation:\n", y_val.value_counts(normalize=True))
print("Test:\n", y_test.value_counts(normalize=True))

# --------------------------------------------------------
# 7. Speichern für Modeling
# --------------------------------------------------------

X_train_scaled.to_csv(os.path.join(DATA_DIR, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(DATA_DIR, "y_train.csv"), index=False)

X_val_scaled.to_csv(os.path.join(DATA_DIR, "X_val.csv"), index=False)
y_val.to_csv(os.path.join(DATA_DIR, "y_val.csv"), index=False)

X_test_scaled.to_csv(os.path.join(DATA_DIR, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(DATA_DIR, "y_test.csv"), index=False)

print("\n✓ Post-Split Data Preparation abgeschlossen!")
