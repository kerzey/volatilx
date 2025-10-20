import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

ALPACA_API_KEY = 'PKYJLOK4LZBY56NZKXZLNSG665'
ALPACA_API_SECRET = '4VVHMnrYEqVv4Jd1oMZMow15DrRVn5p8VD7eEK6TjYZ1'

def fetch_stock_data(symbol: str, start: str, end: str, timeframe: str) -> pd.DataFrame:
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    tf_map = {
        'minute': TimeFrame.Minute,
        'hour': TimeFrame.Hour,
        'day': TimeFrame.Day
    }
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=tf_map.get(timeframe, TimeFrame.Minute),
        start=pd.to_datetime(start),
        end=pd.to_datetime(end)
    )
    bars = client.get_stock_bars(request_params)

    # If bars[symbol].df exists, use it directly
    if hasattr(bars[symbol], 'df'):
        df = bars[symbol].df
    else:
        # Unpack each row of tuples into a dictionary
        unpacked = [dict(item for item in row) for row in bars[symbol]]
        df = pd.DataFrame(unpacked)

    print("Fetched DataFrame head:")
    print(df.head())
    print("Fetched DataFrame columns:", df.columns)

    # Standardize column names
    df.columns = [str(col).lower() for col in df.columns]

    # Optionally rename
    rename_map = {
        'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'
    }
    df.rename(columns=rename_map, inplace=True)

    # Failsafe: only keep relevant columns
    keep = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in keep if col not in df.columns]
    if missing:
        raise Exception(f"Missing required columns: {missing}")
    df = df[keep]

    return df 