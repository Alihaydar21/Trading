import os
from datetime import datetime

from alpaca.data import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# API Keys laden
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Alpaca Client erstellen
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Dynamisches Enddatum (heute)
end_date = datetime.today().strftime("%Y-%m-%d")

# Request für 1-Minuten-Daten ab 2022
request_params = StockBarsRequest(
    symbol_or_symbols=["AAPL"],
    timeframe=TimeFrame.Minute,
    adjustment=Adjustment.ALL,
    start="2022-01-01",
    end=end_date
)

# Daten abrufen
bars = client.get_stock_bars(request_params)

# In DataFrame umwandeln
df = bars.df
print(df.head())
print(df.tail())

# Pfad ermitteln
BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Ordner erstellen, falls er nicht existiert
os.makedirs(DATA_DIR, exist_ok=True)

# Datei speichern
csv_path = os.path.join(DATA_DIR, "AAPL_1min.csv")
df.to_csv(csv_path)

print("AAPL 1-Minute CSV erfolgreich gespeichert in:", csv_path)
print(bars.df['timestamp'].max())

