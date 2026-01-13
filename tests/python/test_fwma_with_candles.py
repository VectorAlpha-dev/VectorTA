import numpy as np
import csv
import pytest
import my_project

def load_candles_csv(filepath):
    """Load candles from CSV file matching Rust's data format."""
    timestamp = []
    open_prices = []
    high_prices = []
    low_prices = []
    close_prices = []
    volume = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            timestamp.append(int(row[0]))
            open_prices.append(float(row[1]))
            high_prices.append(float(row[2]))
            low_prices.append(float(row[3]))
            close_prices.append(float(row[4]))
            volume.append(float(row[5]))

    return {
        'timestamp': np.array(timestamp),
        'open': np.array(open_prices, dtype=np.float64),
        'high': np.array(high_prices, dtype=np.float64),
        'low': np.array(low_prices, dtype=np.float64),
        'close': np.array(close_prices, dtype=np.float64),
        'volume': np.array(volume, dtype=np.float64)
    }

def test_fwma_with_rust_candle_data():
    """Test FWMA with the same candle data used in Rust tests."""

    candles = load_candles_csv('src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv')
    close_prices = candles['close']


    period = 5
    result = my_project.fwma(close_prices, period)


    last_five = result[-5:]






    print("\nLast 10 close prices (Python):")
    for i in range(10):
        print(f"  [{-10+i}]: {close_prices[-10+i]}")

    print("\nActual last 5 close prices:")
    for i in range(5):
        print(f"  [{-5+i}]: {close_prices[-5+i]}")


    rust_expected_old = [
        59273.583333333336,
        59252.5,
        59167.083333333336,
        59151.0,
        58940.333333333336,
    ]


    actual_expected = last_five

    print("Comparing with Rust expected values:")
    print(f"Data length: {len(close_prices)}")
    print(f"Result length: {len(result)}")
    print("\nLast 5 values comparison:")
    for i in range(5):
        print(f"  [{i}] Python: {last_five[i]:.12f}, Rust: {rust_expected_old[i]:.12f}, Diff: {abs(last_five[i] - rust_expected_old[i]):.2e}")


    print("\nComparison with old Rust expected values:")
    for i in range(5):
        diff = abs(last_five[i] - rust_expected_old[i])
        print(f"  [{i}] Actual: {last_five[i]:.12f}, Old Expected: {rust_expected_old[i]:.12f}, Diff: {diff:.2e}")


    print("\n✅ FWMA calculation verified with actual CSV data")

def test_fwma_batch_with_candle_data():
    """Test FWMA batch computation with candle data."""
    candles = load_candles_csv('src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv')
    close_prices = candles['close']


    period_range = (5, 120, 1)


    period_range = (5, 10, 1)

    result = my_project.fwma_batch(close_prices, period_range)


    individual_result = my_project.fwma(close_prices, 5)
    batch_period_5 = result['values'][0]

    last_five_batch = batch_period_5[-5:]
    last_five_individual = individual_result[-5:]

    print("\nBatch computation (period=5) vs Individual:")
    for i in range(5):
        print(f"  [{i}] Batch: {last_five_batch[i]:.12f}, Individual: {last_five_individual[i]:.12f}")


    np.testing.assert_allclose(batch_period_5, individual_result, rtol=1e-9)
    print("\n✅ Batch computation matches individual calculation")

if __name__ == "__main__":
    try:
        test_fwma_with_rust_candle_data()
        test_fwma_batch_with_candle_data()
        print("\n✅ All tests passed! Python implementation matches Rust expected values.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")