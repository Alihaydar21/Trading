### Problem Definition

**Target**

Vorhersage der **Kursrichtung des nächsten Handelstages** (steigt oder fällt) für ausgewählte US-Large-Cap-Aktien.  
Für jede Aktie und jeden Handelstag im Zeitraum **2018-01-01 bis heute** wird das Target wie folgt definiert:

- 1, wenn der Schlusskurs des nächsten Tages höher ist als der heutige Schlusskurs  
- 0, ansonsten  

Formal für Tag *t*:  
target = 1, wenn *close*₍ₜ₊₁₎ > *close*₍ₜ₎, sonst 0.

**Input Features**

- Tägliche OHLCV-Daten (*open, high, low, close, volume*) und *vwap*  
- Normalisierte tägliche Renditen der letzten *k* Tage (k = 1, 3, 5, 10)  
- Exponentielle gleitende Durchschnitte (EMA) über 5, 10, 20 und 50 Handelstage  
- EMA-Differenzen (z. B. EMA₅ − EMA₂₀, EMA₁₀ − EMA₅₀)  
- Steigung der EMAs (erste Ableitung)  
- Rollende Volatilität (Standardabweichung) über 10 und 20 Tage  
- RSI14 (Relative-Strength-Index)  

---

### Procedure Overview

- Bezieht tägliche Kursdaten ausgewählter US-Large-Cap-Aktien über die **Alpaca Market Data API**.  
- Bereinigt die Daten (Entfernen nicht gehandelter Tage, Vereinheitlichung des Handelskalenders).  
- Berechnet die oben beschriebenen Features für jeden Handelstag (kein Data Leakage).  
- Erstellt die binäre Zielvariable (steigt/fällt).  
- Teilt die Daten zeitbasiert in **Train**, **Validation** und **Test** (z. B. Train: 2018–2022, Val: 2023, Test: 2024).  
- Trainiert ein Klassifikationsmodell (Logistic Regression, Random Forest oder Feed-Forward-NN).  
- Bewertet das Modell anhand von Accuracy, F1-Score und Confusion Matrix, inklusive Vergleich mit einer einfachen Baseline.  
- Implementiert eine einfache Long-Only-Strategie, die kauft, wenn das Modell „steigt (1)“ vorhersagt, und verkauft am folgenden Handelstag.  
- Analysiert die Strategie-Performance im Testzeitraum.

---# Data Acquisition

Dieser Schritt lädt die **täglichen Kursdaten (Daily Bars)** ausgewählter US-Large-Cap-Aktien.  
Die Daten werden über die **Alpaca Market Data API** bezogen und enthalten die Felder:

*timestamp, open, high, low, close, volume, trade_count, vwap*

**Script**  
[`scripts/01_data_acquisition/01_data_acquisition.py`](scripts/01_data_acquisition/01_data_acquisition.py)

Das Skript lädt **Daily Bars** für den Zeitraum  
**2018-01-01 bis zum aktuellen Datum**  
und speichert die Rohdaten als `symbol_daily.csv` im Ordner `/data/`.

---
