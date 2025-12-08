---

## Problem Definition

### **Target**

Vorhersage der **Kursrichtung über die nächste Stunde (60 Minuten)** auf Basis von  
**1-Minuten-Intraday-Daten** für eine US-Large-Cap-Aktie (AAPL).

Für jede Minute *t* im Zeitraum **2022-01-01 bis heute** wird das Target wie folgt definiert:

* **1**, wenn der **durchschnittliche Schlusskurs der nächsten 60 Minuten**
  höher ist als der aktuelle Schlusskurs
* **0**, ansonsten

Formal für Minute *t*:  
*target = 1*, wenn  
*(1 / 60) · Σ(closeₜ₊₁ … closeₜ₊₆₀) > closeₜ*,  
sonst *0*.

Diese Target-Definition modelliert einen **robusten Intraday-Trend** und reduziert kurzfristiges Marktrauschen im Vergleich zu einer Ein-Schritt-Vorhersage.

---

### **Input Features**

* 1-Minuten-OHLCV-Daten (*open, high, low, close, volume*)
* *trade_count* und *vwap*
* Kurzfristige Renditen:
  * 1-Minuten-Rendite
  * 15-Minuten-Rendite
  * 60-Minuten-Rendite
* Exponentielle gleitende Durchschnitte (EMA):
  * EMA über 5, 15, 30 und 60 Minuten
* EMA-Differenzen:
  * EMA₅ − EMA₃₀
  * EMA₁₅ − EMA₆₀
* Rollende Volatilität:
  * Fenster: 15 und 60 Minuten
* RSI14 (Relative-Strength-Index) auf Minutenbasis

Alle Features werden **ausschließlich aus vergangenen Daten bis Zeitpunkt *t***
berechnet (**kein Lookahead-Bias**).

---

## Step 1 – Data Acquisition

Lädt **1-Minuten-Intraday-Kursdaten** über die **Alpaca Market Data API**.

**Script**

`scripts/01_data_acquisition.py`

**Datenfelder**

*timestamp, open, high, low, close, volume, trade_count, vwap*

**Zeitraum**

* 2022-01-01 bis aktuelles Datum
* Speicherung als `AAPL_1min.csv` im Ordner `/data/`

---

## Step 2 – Data Understanding

Explorative Analyse der Intraday-Daten zur Überprüfung von Qualität, Verteilung und Struktur.

**Script**

`scripts/02_data_understanding.py`

**Analysen & Plots**

* Intraday-Close-Verlauf eines Beispiel-Handelstags
* Histogramm der 1-Stunden-Renditen
* Intraday-Volumenprofil nach Uhrzeit
* Vergleich aktueller Preis vs. Ø-Preis der nächsten Stunde

Ziel dieses Schrittes ist ein grundlegendes Verständnis der Intraday-Dynamik
und der Signal-Rausch-Struktur.

---

## Step 3 – Pre-Split Data Preparation

Feature Engineering und Target-Erzeugung **vor dem Datensplit**.

**Script**

`scripts/03_pre_split_prep.py`

**Schritte**

* Sortierung der Daten nach Zeitstempel
* Berechnung aller technischen Features
* Erstellung des binären Targets (*target_trend_1h*)
* Entfernen der letzten 60 Minuten ohne gültiges Target
* **Keine Normalisierung und keine globalen Statistiken**

Das Ergebnis dieses Schrittes ist ein vollständig vorbereiteter Datensatz ohne Data Leakage.

**Output**

`data/AAPL_prepared_intraday.csv`

---

## Step 4 – Post-Split Data Preparation

Zeitreihenkonforme Aufteilung und Feature-Scaling.

**Script**

`scripts/04_post_split_preparation.py`

**Schritte**

* Trennung in Features (*X*) und Target (*y*)
* Zeitbasierter Split:
  * ca. 70 % Train
  * ca. 15 % Validation
  * ca. 15 % Test
* **Kein Shuffling**
* Feature-Scaling mit *StandardScaler*:
  * `fit` nur auf Trainingsdaten
  * `transform` auf Validation- und Testdaten

Dieser Schritt simuliert realistische Vorhersagebedingungen ohne Informationsleckage.

---

## Step 5 – Modeling: Logistic Regression (Baseline)

Training eines linearen Baseline-Modells.

**Script**

`scripts/05_model_logistic_regression.py`

**Modell**

* Logistic Regression
* Lineares Modell mit guter Interpretierbarkeit
* Dient als Referenz für komplexere Modelle

**Evaluation**

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Vergleich Train vs. Validation

---

## Step 6 – Modeling: Random Forest

Training eines nicht-linearen Vergleichsmodells.

**Script**

`scripts/06_model_random_forest.py`

**Modell**

* Random Forest Classifier
* Ensemble aus Entscheidungsbäumen
* Modelliert nichtlineare Zusammenhänge

**Zusätzliche Analyse**

* Feature Importances zur Identifikation relevanter Indikatoren
* Vergleich der Generalisierungseigenschaften mit der Logistic Regression

---

## Results Summary

* Beide Modelle erzielen hohe Performances auf Train- und Validation-Daten
* Die **Logistic Regression übertrifft den Random Forest leicht**
* Der geringe Unterschied zwischen Train und Validation deutet auf
  **gute Generalisierungsfähigkeit** hin
* Die Ergebnisse sind im Kontext der Target-Definition zu interpretieren,
  da Features und Target ähnliche Trendinformationen abbilden

---
