"""
06_model_random_forest.py
----------------------------------------
Zweites Modell: Random Forest Classifier

Dieses Skript:
- lädt Train- und Validation-Daten
- trainiert einen Random Forest
- evaluiert auf Train und Validation
- gibt Klassifikationsmetriken und Feature Importances aus
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------------
# 1. Pfade
# --------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
DATA_DIR = os.path.join(ROOT, "data")

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()

X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv")).squeeze()

print("Train Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)

# --------------------------------------------------------
# 2. Modell definieren & trainieren
# --------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,        # ↓ von 300
    max_depth=6,             # ↓ von 8
    min_samples_leaf=100,    # ↑ von 50
    max_samples=0.3,         # 🔑 nur 30% der Daten pro Baum
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


rf_model.fit(X_train, y_train)

# --------------------------------------------------------
# 3. Vorhersagen
# --------------------------------------------------------

y_train_pred = rf_model.predict(X_train)
y_val_pred = rf_model.predict(X_val)

# --------------------------------------------------------
# 4. Evaluation
# --------------------------------------------------------

def evaluate(y_true, y_pred, name):
    print(f"\n=== {name} ===")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1-Score :", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

evaluate(y_train, y_train_pred, "TRAIN")
evaluate(y_val, y_val_pred, "VALIDATION")

print("\nClassification Report (Validation):")
print(classification_report(y_val, y_val_pred))

# --------------------------------------------------------
# 5. Feature Importances
# --------------------------------------------------------

importances = pd.Series(
    rf_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nTop 10 Feature Importances:")
print(importances.head(10))
