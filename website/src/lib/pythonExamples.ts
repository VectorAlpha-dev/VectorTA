// Python binding examples for indicators

export function getPythonExample(indicatorId: string): string {
	const functionName = `calculate_${indicatorId}`;
	
	switch (indicatorId) {
		case 'rsi':
			return `# Import the Python bindings
import rust_backtester as rb
import numpy as np

def calculate_rsi_example():
    # Sample price data (closing prices)
    prices = np.array([
        44.94, 44.95, 45.15, 45.29, 45.41,
        45.23, 45.08, 45.15, 45.57, 45.32,
        45.15, 45.39, 45.83, 45.85, 46.08
    ])
    
    # Calculate RSI with period of 14
    rsi_values = rb.${functionName}(prices, period=14)
    
    print("RSI Values:", rsi_values)
    return rsi_values

# Example usage
if __name__ == "__main__":
    rsi_result = calculate_rsi_example()`;

		case 'sma':
			return `# Import the Python bindings
import rust_backtester as rb
import numpy as np

def calculate_sma_example():
    # Sample price data
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    period = 5
    
    # Calculate Simple Moving Average
    sma_values = rb.${functionName}(prices, period=period)
    
    print(f"SMA({period}) Values:", sma_values)
    return sma_values

# Example usage
if __name__ == "__main__":
    sma_result = calculate_sma_example()`;

		case 'ema':
			return `# Import the Python bindings
import rust_backtester as rb
import numpy as np

def calculate_ema_example():
    # Sample price data
    prices = np.array([22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29])
    period = 10
    
    # Calculate Exponential Moving Average
    ema_values = rb.${functionName}(prices, period=period)
    
    print(f"EMA({period}) Values:", ema_values)
    return ema_values

# Example usage
if __name__ == "__main__":
    ema_result = calculate_ema_example()`;

		case 'macd':
			return `# Import the Python bindings
import rust_backtester as rb
import numpy as np

def calculate_macd_example():
    # Sample price data
    prices = np.array([/* your price data */])
    
    # MACD parameters
    fast_period = 12
    slow_period = 26
    signal_period = 9
    
    # Calculate MACD
    result = rb.${functionName}(
        prices, 
        fast_period=fast_period, 
        slow_period=slow_period, 
        signal_period=signal_period
    )
    
    print("MACD Line:", result['macd'])
    print("Signal Line:", result['signal'])
    print("Histogram:", result['histogram'])
    return result

# Example usage
if __name__ == "__main__":
    macd_result = calculate_macd_example()`;

		case 'bollinger_bands':
			return `# Import the Python bindings
import rust_backtester as rb
import numpy as np

def calculate_bollinger_bands_example():
    # Sample price data
    prices = np.array([/* your price data */])
    period = 20
    std_dev = 2.0
    
    # Calculate Bollinger Bands
    result = rb.${functionName}(
        prices, 
        period=period, 
        std_dev=std_dev
    )
    
    print("Upper Band:", result['upper'])
    print("Middle Band:", result['middle'])
    print("Lower Band:", result['lower'])
    return result

# Example usage
if __name__ == "__main__":
    bb_result = calculate_bollinger_bands_example()`;

		case 'atr':
			return `# Import the Python bindings
import rust_backtester as rb
import numpy as np

def calculate_atr_example():
    # Sample OHLC data
    highs = np.array([/* high prices */])
    lows = np.array([/* low prices */])
    closes = np.array([/* close prices */])
    period = 14
    
    # Calculate Average True Range
    atr_values = rb.${functionName}(
        highs, 
        lows, 
        closes, 
        period=period
    )
    
    print(f"ATR({period}) Values:", atr_values)
    return atr_values

# Example usage
if __name__ == "__main__":
    atr_result = calculate_atr_example()`;

		case 'stoch':
			return `# Import the Python bindings
import rust_backtester as rb
import numpy as np

def calculate_stochastic_example():
    # Sample OHLC data
    highs = np.array([/* high prices */])
    lows = np.array([/* low prices */])
    closes = np.array([/* close prices */])
    k_period = 14
    d_period = 3
    
    # Calculate Stochastic Oscillator
    result = rb.${functionName}(
        highs, 
        lows, 
        closes, 
        k_period=k_period, 
        d_period=d_period
    )
    
    print("%K Values:", result['k'])
    print("%D Values:", result['d'])
    return result

# Example usage
if __name__ == "__main__":
    stoch_result = calculate_stochastic_example()`;

		default:
			const installationNote = indicatorId === 'obv' || indicatorId === 'ad' ? 
				'# Note: This indicator only requires close prices and volume' :
				indicatorId.includes('ht_') ? 
				'# Note: Hilbert Transform indicators may require specific data preparation' :
				'# Note: Check documentation for specific parameter requirements';

			return `# Import the Python bindings
import rust_backtester as rb
import numpy as np

def calculate_${indicatorId}_example():
    """
    Example usage of ${indicatorId.toUpperCase()} indicator using Python bindings.
    
    The Rust-Backtester library provides high-performance implementations
    of technical analysis indicators with Python bindings for easy integration.
    """
    
    # Prepare your market data
    # ${installationNote}
    prices = np.array([/* your price data */])
    
    # Calculate ${indicatorId.toUpperCase()} with appropriate parameters
    result = rb.${functionName}(prices, **parameters)
    
    print("${indicatorId.toUpperCase()} Values:", result)
    return result

# Installation
# pip install rust-backtester

# Example usage
if __name__ == "__main__":
    ${indicatorId}_result = calculate_${indicatorId}_example()
    
    # Optional: Use with pandas for easier data handling
    import pandas as pd
    
    df = pd.DataFrame({
        'price': prices,
        '${indicatorId}': ${indicatorId}_result
    })
    print(df.head())`;
	}
}

export function getPythonInstallationExample(): string {
	return `# Install the Python bindings for Rust-Backtester
pip install rust-backtester

# Alternative: Install from source
# pip install git+https://github.com/DanielLiszka/Rust-Backtester.git

# Verify installation
python -c "import rust_backtester; print('Successfully installed!')"`;
}

export function getPythonPandasExample(indicatorId: string): string {
	return `# Integration with pandas for data analysis
import rust_backtester as rb
import pandas as pd
import numpy as np

# Load market data (example with yfinance)
# pip install yfinance
import yfinance as yf

def analyze_with_pandas():
    # Download sample data
    ticker = "AAPL"
    data = yf.download(ticker, period="1y")
    
    # Calculate ${indicatorId.toUpperCase()}
    ${indicatorId}_values = rb.calculate_${indicatorId}(
        data['Close'].values, 
        period=14  # adjust parameters as needed
    )
    
    # Add to DataFrame
    data['${indicatorId.toUpperCase()}'] = np.nan
    data['${indicatorId.toUpperCase()}'].iloc[len(data)-len(${indicatorId}_values):] = ${indicatorId}_values
    
    # Display results
    print(data[['Close', '${indicatorId.toUpperCase()}']].tail())
    
    # Optional: Plot results
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Price chart
    ax1.plot(data.index, data['Close'])
    ax1.set_title(f'{ticker} Price')
    ax1.set_ylabel('Price')
    
    # Indicator chart
    ax2.plot(data.index, data['${indicatorId.toUpperCase()}'])
    ax2.set_title(f'{indicatorId.toUpperCase()} Indicator')
    ax2.set_ylabel('${indicatorId.toUpperCase()}')
    ax2.set_xlabel('Date')
    
    plt.tight_layout()
    plt.show()
    
    return data

if __name__ == "__main__":
    result_df = analyze_with_pandas()`;
}