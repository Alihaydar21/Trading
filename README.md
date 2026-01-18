# ML-based Intraday Trading System (Backtesting & Paper Trading)

---

## Problem Definition

### **Target**

Vorhersage der **Kursrichtung über die nächste Stunde (60 Minuten)** auf Basis von  
**1-Minuten-Intraday-Daten**.

Für jede Minute *t* wird das Target wie folgt definiert:

- **1**, wenn der **durchschnittliche Schlusskurs der nächsten 60 Minuten**
  höher ist als der aktuelle Schlusskurs
- **0**, ansonsten

Formal für Minute *t*:  
`target = 1`, wenn  
`(1 / 60) · Σ(closeₜ₊₁ … closeₜ₊₆₀) > closeₜ`,  
sonst `0`.

Diese Definition modelliert einen **robusten Intraday-Trend** und reduziert kurzfristiges Marktrauschen.

---

### **Input Features**

- 1-Minuten-OHLCV-Daten (*open, high, low, close, volume*)
- *trade_count* und *vwap*
- Kurzfristige Renditen:
  - 1-Minuten-Rendite
  - 15-Minuten-Rendite
  - 60-Minuten-Rendite
- Exponentielle gleitende Durchschnitte (EMA):
  - EMA über 5, 15, 30 und 60 Minuten
- EMA-Differenzen:
  - EMA₅ − EMA₃₀
  - EMA₁₅ − EMA₆₀
- Rollende Volatilität:
  - Fenster: 15 und 60 Minuten
- RSI14
- Asset-Information:
-One-Hot-Encoding (sym_AAPL, sym_MSFT, sym_NVDA)

Alle Features werden **ausschließlich aus historischen Daten bis Zeitpunkt *t***
berechnet (**kein Lookahead-Bias**).

---

## Step 1 – Data Acquisition

Lädt **1-Minuten-Intraday-Kursdaten** über die **Alpaca Market Data API**.
Yahoo Finance (**ergänzend**). Nutzung für sehr **aktuelle** oder **fehlende Intraday-Minuten**.
Zur Sicherstellung vollständiger Zeitabdeckung.

Zeitraum:
ca. 2022 bis **heute**

Enthält verschiedene Marktphasen (Trend, Seitwärts, hohe Volatilität)

**Script**

- `scripts/01_data_acquisition.py`

**Assets**

- AAPL
- MSFT
- NVDA

**Output**

- `/data/AAPL_1min.csv`
- `/data/MSFT_1min.csv`
- `/data/NVDA_1min.csv`

---

## Step 2 – Data Understanding

Explorative Analyse der Intraday-Daten.

**Script**

- `scripts/02_data_understanding.py`

**Analysen & Plots**

- Intraday-Close-Verlauf eines Beispiel-Handelstags
- Histogramm der 1-Stunden-Renditen
- Volumenverteilung über den Handelstag
- Vergleich aktueller Preis vs. Ø-Preis der nächsten Stunde

Ziel: Verständnis der **Intraday-Dynamik** und der **Signal-Rausch-Struktur**.

---

## Step 3 – Pre-Split Data Preparation

Feature Engineering und Target-Erzeugung **vor dem Datensplit**.

**Script**

- `scripts/03_pre_split_prep.py`

**Schritte**

- Sortierung nach Zeitstempel
- Berechnung aller technischen Features
- Erstellung des binären Targets (*target_trend_1h*)
- Entfernen der letzten 60 Minuten ohne gültiges Target
- Keine Normalisierung

**Output**

- `data/*_prepared_intraday.csv`
- `data/MULTI_prepared_intraday.csv`

---

## Step 4 – Post-Split Data Preparation

Zeitreihenkonformer Datensplit und Feature-Scaling.

**Script**

- `scripts/04_post_split_preparation.py`

**Schritte**

- Zeitbasierter Split:
  - ~70 % Training
  - ~15 % Validation
  - ~15 % Test
- Kein Shuffling
- StandardScaler:
  - Fit nur auf Trainingsdaten
- Speicherung von scaler.pk

---

## Step 5 – Modeling

### **Baseline: Logistic Regression**

**Script**

- `scripts/05_model_logistic_regression.py`

**Ziel**

- Interpretierbare Referenz
- Vergleichsbasis für komplexere Modelle

---

### **Random Forest Classifier**

**Script**

- `scripts/06_model_random_forest.py`

**Eigenschaften**

- Nichtlineares Ensemble-Modell
- Modelliert komplexe Feature-Interaktionen
- Analyse der Feature Importances
- Asset-spezifische Effekte erkennbar

---

## Step 6 – Backtesting (Trading Simulation)

Simulation einer **regelbasierten Trading-Strategie**, abgeleitet aus den ML-Vorhersagen.

**Script**

- `scripts/07_backtesting.py`

### **Trading-Logik**

- Long-only
- **Signal**: Modell gibt Wahrscheinlichkeit `p_up`
- **Entry**: `p_up ≥ 0.6`
- **Exit**: nach 60 Minuten
- Return-Clipping: ±5 %

### **Ergebnisse**

- Anzahl Trades: 432
- Trefferquote: ≈ 51.6 %
- Durchschnittlicher Trade-Return: positiv
- Kumulierte Signal-Equity: klar steigend

Backtesting zeigt, **wie die Strategie in der Vergangenheit performt hätte**.

---

## Step 7 – Paper Trading (Deployment-Simulation)

Simulation eines realistischen Live-Szenarios ohne echte Orders.

**Script**

- `scripts/08_deployment.py`

### **Setup**

- Sequentielle Vorhersagen (1-Minuten-Takt)
- Keine Zukunftsinformation
- Zeitfenster: letzte 5 Handelstage
- Multi-Asset-Setup (AAPL, MSFT, NVDA)
- Persistenter State (Restart-sicher)

### **Trading-Regeln**

- Long-only
- Entry: `p_up ≥ 0.56`
- Exit: nach 60 Minuten
- Asset-Filter optional aktivierbar

---

## Asset Filter (z. B. nur NVDA)

Analyse zeigt deutliche **Performance-Unterschiede je Asset**:

- **NVDA**: stark positiv
- **AAPL**: leicht positiv
- **MSFT**: negativ

**Ableitung**

- Die gleiche ML-Strategie funktioniert **nicht gleich gut für jedes Asset**
- Asset-Selektion ist ein **entscheidender Performance-Hebel**
- Optimierte Variante:
  - NVDA handeln
  - MSFT ausschließen
  - AAPL optional

---

## Evaluation Summary

- ML-Modell erzeugt **statistische Wahrscheinlichkeiten**, keine direkten Kauf-/Verkaufssignale
- Trading-Performance hängt stark von:
- Entry-Schwelle
- Haltedauer
- Asset-Auswahl
- Paper Trading bestätigt Backtesting-Ergebnisse qualitativ
- Asset-Filter erhöht Robustheit der Strategie

---

## Fazit

Das Projekt demonstriert den **vollständigen Data-Science-Zyklus von Datenakquise bis Deployment**,
inklusive realistischer Trading-Regeln, systematischer Evaluation und iterativer Verbesserung.
