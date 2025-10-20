import pandas as pd
from data_fetcher import fetch_stock_data
from indicator_analysis import calculate_all_indicators
from signal_generator import ComprehensiveSignalGenerator
import numpy as np

if __name__ == "__main__":
    # Parameters -- adjust as needed
    symbol = "AAPL"
    start = "2025-09-01"
    end = "2025-10-18"
    timeframe = "hour"   # Supports: "minute", "hour", "day" based on your tf_map or indicator parameter mapping

    # Fetch data
    df = fetch_stock_data(symbol, start, end, timeframe)
    print("Raw columns:", df.columns)
    df.columns = [col.lower() for col in df.columns]

    # Validate key columns
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            raise Exception(f"Missing required column: {col}")

    # Calculate indicators
    indicators_df = calculate_all_indicators(df, timeframe=timeframe)
    # Add "close" column for direct price referencing in signals
    indicators_df['close'] = df['close']

    # Print head for inspection
    print("Sample indicators head:\n", indicators_df.head())
    
    # Instantiate new signal generator class
    signal_gen = ComprehensiveSignalGenerator()

    # -- Analyze signals for all bars (optional, for batch output) --
    all_signals = signal_gen.analyze(indicators_df)
    print("All signals", all_signals)
    # print("\nSignals for all bars (first 2 shown for brevity):")
    # for i, sig in enumerate(all_signals[:2]):
    #     print(f"Bar {i}:", sig)
    
    # # -- Analyze signal for most recent bar --
    # last_signal = signal_gen.last_signal(indicators_df)
    # print("\nMost recent signal interpretation:")
    # print(last_signal)
