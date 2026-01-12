"""
08_deployment.py
----------------------------------------
LIVE Paper Trading (Alpaca Paper Trading API)
- Long-only
- Entry: p_up >= threshold
- Exit: nach HOLD_MINUTES
- Modell: RandomForest, trainiert wie bei dir (X_train+X_val)
- Live-Features werden mit scaler.pkl skaliert

WICHTIG:
- Live-Feature-Set muss 1:1 zu X_train passen (inkl. sym_* one-hot)
"""

import os
import time
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

import joblib
from sklearn.ensemble import RandomForestClassifier

from alpaca.data.enums import DataFeed
from alpaca.data import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# --------------------------------------------------------
# 0) ENV / Settings
# --------------------------------------------------------
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
if not API_KEY or not SECRET_KEY:
    raise RuntimeError("ALPACA_API_KEY oder ALPACA_SECRET_KEY fehlen")

TRADE_ASSETS = ["AAPL", "MSFT", "NVDA"]

QTY = float(os.getenv("TRADE_QTY", "1"))
P_UP_THRESHOLD = float(os.getenv("P_UP_THRESHOLD", "0.512"))
HOLD_MINUTES = int(os.getenv("HOLD_MINUTES", "60"))

ALPACA_TRADE_BASE = os.getenv("ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets")
LOOKBACK_MINUTES = int(os.getenv("LOOKBACK_MINUTES", "360"))  # etwas mehr für RSI/EMA

# optionaler State, damit Exit nach Restart klappt
STATE_PATH = os.getenv("LIVE_STATE_PATH", "paper_trading/live_state.json")


# --------------------------------------------------------
# 1) Pfade
# --------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))

DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "paper_trading")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRADES_LOG_PATH = os.path.join(OUTPUT_DIR, "live_paper_trades.csv")
STATE_PATH = os.path.join(ROOT, STATE_PATH)


# --------------------------------------------------------
# ✅ (NEU) Robust: Trades-Log sicher lesen (kein KeyError mehr)
# --------------------------------------------------------
def safe_read_trades_log(path: str) -> pd.DataFrame:
    """
    Liest live_paper_trades.csv robust:
    - wenn Datei nicht existiert => leeres DF
    - wenn Datei existiert aber leer/kaputt => leeres DF + Debug
    """
    print("[DEBUG] TRADES_LOG_PATH:", path)
    print("[DEBUG] exists:", os.path.exists(path))

    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        print("[DEBUG] columns:", df.columns.tolist())
        print("[DEBUG] head:\n", df.head(3))

        # timestamp optional parse (nur wenn vorhanden)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return df
    except Exception as e:
        print("[WARN] Konnte Trades-Log nicht lesen:", e)
        return pd.DataFrame()


def summarize_trades_log(df: pd.DataFrame):
    """
    Optional: kurze Übersicht, ohne dass der Bot crasht.
    """
    if df is None or df.empty:
        print("[INFO] Trades-Log leer (noch keine Trades).")
        return

    if "timestamp" not in df.columns:
        print("[INFO] Trades-Log hat keine 'timestamp'-Spalte. Spalten:", df.columns.tolist())
        return

    if "action" not in df.columns:
        print("[INFO] Trades-Log hat keine 'action'-Spalte. Spalten:", df.columns.tolist())
        return

    today = datetime.now(timezone.utc).date()
    df_ok = df.dropna(subset=["timestamp"]).copy()
    df_today = df_ok[df_ok["timestamp"].dt.date == today]

    if df_today.empty:
        print("[INFO] Heute noch keine Trades im Log.")
        return

    print("\n[INFO] Trades heute (UTC):")
    print(df_today["action"].value_counts())


# --------------------------------------------------------
# 2) ML-Daten laden
# --------------------------------------------------------
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()

X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv")).squeeze()

MODEL_FEATURES = list(X_train.columns)


# --------------------------------------------------------
# 3) Scaler laden
# --------------------------------------------------------
scaler_path = os.path.join(DATA_DIR, "scaler.pkl")
if not os.path.exists(scaler_path):
    raise RuntimeError(f"Scaler nicht gefunden: {scaler_path}")
scaler = joblib.load(scaler_path)


# --------------------------------------------------------
# 4) Modell trainieren (Train + Val)
# --------------------------------------------------------
X_train_full = pd.concat([X_train, X_val], ignore_index=True)
y_train_full = pd.concat([y_train, y_val], ignore_index=True)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=14,
    min_samples_leaf=50,
    max_samples=0.5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_full, y_train_full)


# --------------------------------------------------------
# 5) Feature Engineering (wie 03, live intraday-safe)
# --------------------------------------------------------
def compute_rsi(series: pd.Series, window=14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

def build_features_from_close(close: pd.Series) -> pd.DataFrame:
    feats = pd.DataFrame(index=close.index)

    feats["return_1min"] = close.pct_change(1)
    feats["return_15min"] = close.pct_change(15)

    ema_5 = close.ewm(span=5, adjust=False).mean()
    ema_15 = close.ewm(span=15, adjust=False).mean()
    ema_30 = close.ewm(span=30, adjust=False).mean()

    feats["ema_5_30_diff"] = ema_5 - ema_30
    feats["ema_15_30_diff"] = ema_15 - ema_30

    feats["rsi_14"] = compute_rsi(close, 14)

    return feats


# --------------------------------------------------------
# 6) Alpaca Data: letzte N Minuten 1-min Bars (IEX!)
# --------------------------------------------------------
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

def fetch_alpaca_recent_1m(symbol: str, lookback_minutes: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=lookback_minutes)

    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        adjustment=Adjustment.ALL,
        start=start,
        end=end,
        feed=DataFeed.IEX
    )

    bars = data_client.get_stock_bars(req).df
    if bars is None or bars.empty:
        raise RuntimeError("Alpaca lieferte keine Bars (Market zu? IEX? Rate limit?)")

    # MultiIndex (symbol, timestamp)
    df_sym = bars.loc[symbol].copy().reset_index()

    df_sym["timestamp"] = pd.to_datetime(df_sym["timestamp"], utc=True, errors="coerce")
    df_sym["close"] = pd.to_numeric(df_sym["close"], errors="coerce")
    df_sym = df_sym.dropna(subset=["timestamp", "close"]).sort_values("timestamp")

    if len(df_sym) < 80:
        raise RuntimeError(f"Zu wenig Bars ({len(df_sym)}). Market evtl. zu/Lookback erhöhen.")

    return df_sym[["timestamp", "close"]]


# --------------------------------------------------------
# 7) Alpaca Trading (Paper): REST Orders / Positions
# --------------------------------------------------------
def alpaca_headers():
    return {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY,
        "Content-Type": "application/json",
    }

def get_position(symbol: str):
    url = f"{ALPACA_TRADE_BASE}/v2/positions/{symbol}"
    r = requests.get(url, headers=alpaca_headers(), timeout=30)
    if r.status_code in (404, 422):
        return None
    r.raise_for_status()
    return r.json()

def submit_market_order(symbol: str, qty: float, side: str):
    url = f"{ALPACA_TRADE_BASE}/v2/orders"
    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side.lower(),
        "type": "market",
        "time_in_force": "day",
    }
    r = requests.post(url, headers=alpaca_headers(), data=json.dumps(payload), timeout=30)
    r.raise_for_status()
    return r.json()

def close_position(symbol: str):
    url = f"{ALPACA_TRADE_BASE}/v2/positions/{symbol}"
    r = requests.delete(url, headers=alpaca_headers(), timeout=30)
    if r.status_code in (404, 422):
        return None
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------
# 8) State + Logging
# --------------------------------------------------------
def load_state():
    if not os.path.exists(STATE_PATH):
        return {s: None for s in TRADE_ASSETS}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in TRADE_ASSETS:
            data.setdefault(s, None)
        return data
    except Exception:
        return {s: None for s in TRADE_ASSETS}

def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f)

def append_trade_log(row: dict):
    df_row = pd.DataFrame([row])
    if not os.path.exists(TRADES_LOG_PATH):
        df_row.to_csv(TRADES_LOG_PATH, index=False)
    else:
        df_row.to_csv(TRADES_LOG_PATH, mode="a", header=False, index=False)


def add_symbol_onehot(df_feats: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df_feats = df_feats.copy()
    for s in TRADE_ASSETS:
        col = f"sym_{s}"
        if col in MODEL_FEATURES:
            df_feats[col] = 1 if symbol == s else 0
    return df_feats

def ensure_feature_alignment(df_feats: pd.DataFrame) -> pd.DataFrame:
    for c in MODEL_FEATURES:
        if c not in df_feats.columns:
            df_feats[c] = 0
    df_aligned = df_feats[MODEL_FEATURES].copy()
    df_aligned = df_aligned.dropna()
    if df_aligned.empty:
        raise RuntimeError("Features sind NaN (zu wenig Historie) – LOOKBACK erhöhen.")
    return df_aligned


# --------------------------------------------------------
# 9) LIVE Trading Loop
# --------------------------------------------------------
def main():
    print("============== LIVE PAPER TRADING START ==============")
    print(f"Assets: {TRADE_ASSETS}")
    print(f"Threshold: {P_UP_THRESHOLD} | Qty: {QTY} | Hold(min): {HOLD_MINUTES}")
    print(f"Paper Trading Base: {ALPACA_TRADE_BASE}")
    print(f"Lookback Minutes: {LOOKBACK_MINUTES}")
    print("======================================================")

    # ✅ (NEU) Log-Check beim Start (kein KeyError mehr)
    df_log = safe_read_trades_log(TRADES_LOG_PATH)
    summarize_trades_log(df_log)

    state = load_state()  # symbol -> iso timestamp string oder None

    while True:
        now = datetime.now(timezone.utc)

        for symbol in TRADE_ASSETS:
            try:
                bars = fetch_alpaca_recent_1m(symbol, LOOKBACK_MINUTES)

                today = now.date()
                bars_today = bars[bars["timestamp"].dt.date == today].copy()
                if len(bars_today) < 80:
                    bars_today = bars.copy()

                close = pd.Series(bars_today["close"].to_numpy(), index=bars_today["timestamp"], name="close")
                latest_price = float(close.iloc[-1])

                feats = build_features_from_close(close)
                feats = add_symbol_onehot(feats, symbol)
                feats = ensure_feature_alignment(feats).tail(1)

                x_scaled = scaler.transform(feats)
                x_scaled_df = pd.DataFrame(x_scaled, columns=MODEL_FEATURES)

                p_up = float(rf.predict_proba(x_scaled_df)[0, 1])

                pos = get_position(symbol)
                has_pos = pos is not None

                entry_iso = state.get(symbol)
                entry_dt = datetime.fromisoformat(entry_iso) if entry_iso else None

                print(f"[{now.isoformat()}] {symbol} p_up={p_up:.3f} price={latest_price:.2f} has_pos={has_pos} entry={entry_iso}")

                # Entry
                if (not has_pos) and (p_up >= P_UP_THRESHOLD):
                    od = submit_market_order(symbol, QTY, "buy")
                    state[symbol] = now.isoformat()
                    save_state(state)

                    append_trade_log({
                        "timestamp": now.isoformat(),
                        "symbol": symbol,
                        "action": "BUY",
                        "qty": QTY,
                        "price": latest_price,
                        "p_up": p_up,
                        "alpaca_order_id": od.get("id")
                    })
                    print(f"  -> BUY gesendet: {symbol} | order_id={od.get('id')}")

                # Exit nach HOLD_MINUTES (wenn Entry bekannt)
                if has_pos and entry_dt is not None:
                    if now - entry_dt >= timedelta(minutes=HOLD_MINUTES):
                        res = close_position(symbol)
                        state[symbol] = None
                        save_state(state)

                        append_trade_log({
                            "timestamp": now.isoformat(),
                            "symbol": symbol,
                            "action": "CLOSE",
                            "qty": QTY,
                            "price": latest_price,
                            "p_up": p_up,
                            "alpaca_order_id": (res.get("id") if isinstance(res, dict) else None)
                        })
                        print(f"  -> Position geschlossen: {symbol}")

            except Exception as e:
                print(f"[ERROR] {symbol}: {e}")

        time.sleep(60)


if __name__ == "__main__":
    main()
