import os
from datetime import datetime
from pathlib import Path

from alpaca.data import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# ===============================
# Alpaca API Keys
# ===============================
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise RuntimeError("ALPACA_API_KEY oder ALPACA_SECRET_KEY fehlen")


# ===============================
# Alpaca Client
# ===============================
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


# ===============================
# Zeitraum
# ===============================
start_date = "2022-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")


# ===============================
# Request
# ===============================
request_params = StockBarsRequest(
    symbol_or_symbols=["AAPL"],
    timeframe=TimeFrame.Minute,
    adjustment=Adjustment.ALL,
    start=start_date,
    end=end_date
)


# ===============================
# Daten abrufen
# ===============================
bars = client.get_stock_bars(request_params)
df = bars.df

print(df.head())
print(df.tail())


# ===============================
# ✅ DESKTOP-PFAD (JETZT SAUBER)
# ===============================
SCRIPT_DIR = Path(__file__).resolve().parent          # Desktop/Trade/scripts
PROJECT_ROOT = SCRIPT_DIR.parent                      # Desktop/Trade
DATA_DIR = PROJECT_ROOT / "data"

# Ordner sicher erstellen
os.makedirs(DATA_DIR, exist_ok=True)


# ===============================
# CSV speichern
# ===============================
csv_path = DATA_DIR / "AAPL_1min.csv"
df.to_csv(csv_path)


# ===============================
# Erfolg
# ===============================
print("\n✅ CSV erfolgreich gespeichert:")
print(csv_path)
print("Letzter Timestamp:", df.index.max())
