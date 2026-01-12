"""
05_model_logistic_regression.py
----------------------------------------
Baseline Logistic Regression + Proba-Diagnostik
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
DATA_DIR = os.path.join(ROOT, "data")

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()

X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv")).squeeze()

print("Train Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
)

model.fit(X_train, y_train)

def evaluate(y_true, y_pred, name):
    print(f"\n=== {name} ===")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_true, y_pred, zero_division=0))
    print("F1-Score :", f1_score(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

y_train_pred = model.predict(X_train)
y_val_pred   = model.predict(X_val)

evaluate(y_train, y_train_pred, "TRAIN")
evaluate(y_val, y_val_pred, "VALIDATION")

print("\nClassification Report (Validation):")
print(classification_report(y_val, y_val_pred, zero_division=0))

# Proba-Diagnostik
p_val = model.predict_proba(X_val)[:, 1]
print("\nProba-Stats (VAL):")
print("min:", float(np.min(p_val)), "max:", float(np.max(p_val)))
print("p50:", float(np.quantile(p_val, 0.50)), "p90:", float(np.quantile(p_val, 0.90)), "p99:", float(np.quantile(p_val, 0.99)))

print("\n✓ Logistic Regression Baseline abgeschlossen!")
