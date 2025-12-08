"""
05_model_logistic_regression.py
----------------------------------------
Baseline-Modell: Logistic Regression

Dieses Skript:
- lädt die post-split vorbereiteten Daten
- trainiert eine Logistic Regression
- evaluiert auf Train und Validation
- gibt relevante Klassifikationsmetriken aus
"""

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
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

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",   # wichtig bei Intraday-Daten!
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------------
# 3. Vorhersagen
# --------------------------------------------------------

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

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

print("\n✓ Logistic Regression Training abgeschlossen!")
