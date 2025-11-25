import os
from datetime import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Alpaca Client
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Dynamisches Enddatum (immer das heutige Datum)
end_date = datetime.today().strftime("%Y-%m-%d")


request_params = StockBarsRequest(
    symbol_or_symbols=["AAPL"],
    timeframe=TimeFrame.Day,
    start="2018-01-01",
    end=end_date
)
bars = client.get_stock_bars(request_params)

df = bars.df
print(df.head())
print(df.tail())

BASE = os.path.dirname(os.path.abspath(__file__))

# Projekt-Root (Trade/)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE, ".."))

# 100% sicher: Data-Ordner erstellen
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Datei abspeichern
csv_path = os.path.join(DATA_DIR, "AAPL_daily.csv")
df.to_csv(csv_path)
