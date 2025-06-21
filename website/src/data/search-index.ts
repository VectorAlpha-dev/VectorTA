// Auto-generated search index
export const searchIndex: SearchItem[] = [
  {
    "id": "acosc",
    "title": "Acceleration/Deceleration Oscillator",
    "description": "The AC Oscillator measures whether the market's driving force is accelerating or decelerating. Think of it as the 'speedometer' of price momentum — it tells you not just which direction the market is moving, but whether it's speeding up or slowing down.",
    "category": "oscillators",
    "url": "/indicators/acosc",
    "type": "indicator"
  },
  {
    "id": "macd",
    "title": "Moving Average Convergence Divergence",
    "description": "A trend-following momentum indicator that shows the relationship between two moving averages.",
    "category": "momentum",
    "url": "/indicators/macd",
    "type": "indicator"
  },
  {
    "id": "roc",
    "title": "Rate of Change",
    "description": "Measures the percentage change in price from one period to another.",
    "category": "momentum",
    "url": "/indicators/roc",
    "type": "indicator"
  },
  {
    "id": "rsi",
    "title": "Relative Strength Index (RSI)",
    "description": "The Relative Strength Index is a momentum oscillator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of an asset.",
    "category": "momentum",
    "url": "/indicators/rsi",
    "type": "indicator"
  },
  {
    "id": "stochastic",
    "title": "Stochastic Oscillator",
    "description": "A momentum indicator comparing a particular closing price to a range of prices over time.",
    "category": "momentum",
    "url": "/indicators/stochastic",
    "type": "indicator"
  },
  {
    "id": "dema",
    "title": "Double Exponential Moving Average",
    "description": "A technical indicator that uses two exponential moving averages to eliminate lag.",
    "category": "moving_averages",
    "url": "/indicators/dema",
    "type": "indicator"
  },
  {
    "id": "ema",
    "title": "Exponential Moving Average (EMA)",
    "description": "The Exponential Moving Average is a weighted moving average that gives more importance to recent prices, making it more responsive to new information than the Simple Moving Average.",
    "category": "moving_averages",
    "url": "/indicators/ema",
    "type": "indicator"
  },
  {
    "id": "sma",
    "title": "Simple Moving Average (SMA)",
    "description": "The Simple Moving Average calculates the arithmetic mean of a given set of prices over a specific number of periods. It's one of the most fundamental technical indicators used to smooth price action and identify trends.",
    "category": "moving_averages",
    "url": "/indicators/sma",
    "type": "indicator"
  },
  {
    "id": "tema",
    "title": "Triple Exponential Moving Average",
    "description": "A technical indicator that uses three exponential moving averages to reduce lag.",
    "category": "moving_averages",
    "url": "/indicators/tema",
    "type": "indicator"
  },
  {
    "id": "wma",
    "title": "Weighted Moving Average",
    "description": "A moving average that assigns greater weight to more recent data points.",
    "category": "moving_averages",
    "url": "/indicators/wma",
    "type": "indicator"
  },
  {
    "id": "cci",
    "title": "Commodity Channel Index",
    "description": "Measures the variation of a price from its statistical mean.",
    "category": "oscillators",
    "url": "/indicators/cci",
    "type": "indicator"
  },
  {
    "id": "williams_r",
    "title": "Williams %R",
    "description": "A momentum indicator that measures overbought and oversold levels.",
    "category": "oscillators",
    "url": "/indicators/williams_r",
    "type": "indicator"
  },
  {
    "id": "adx",
    "title": "Average Directional Index",
    "description": "Measures the strength of a trend, regardless of its direction.",
    "category": "trend",
    "url": "/indicators/adx",
    "type": "indicator"
  },
  {
    "id": "ichimoku",
    "title": "Ichimoku Cloud",
    "description": "A comprehensive indicator that defines support/resistance, trend direction, and momentum.",
    "category": "trend",
    "url": "/indicators/ichimoku",
    "type": "indicator"
  },
  {
    "id": "atr",
    "title": "Average True Range",
    "description": "Measures market volatility by analyzing the range of price movement.",
    "category": "volatility",
    "url": "/indicators/atr",
    "type": "indicator"
  },
  {
    "id": "bollinger_bands",
    "title": "Bollinger Bands",
    "description": "Bollinger Bands are a volatility indicator that creates a band of three lines which are plotted in relation to a security's price. The middle band is a simple moving average, with upper and lower bands adjusted for volatility.",
    "category": "volatility",
    "url": "/indicators/bollinger_bands",
    "type": "indicator"
  },
  {
    "id": "standard_deviation",
    "title": "Standard Deviation",
    "description": "Measures the dispersion of a dataset relative to its mean.",
    "category": "volatility",
    "url": "/indicators/standard_deviation",
    "type": "indicator"
  },
  {
    "id": "obv",
    "title": "On Balance Volume",
    "description": "Uses volume flow to predict changes in stock price.",
    "category": "volume",
    "url": "/indicators/obv",
    "type": "indicator"
  },
  {
    "id": "vwap",
    "title": "Volume Weighted Average Price",
    "description": "The average price weighted by volume, often used as a trading benchmark.",
    "category": "volume",
    "url": "/indicators/vwap",
    "type": "indicator"
  },
  {
    "id": "home",
    "title": "Home",
    "description": "VectorTA - High-performance technical analysis library",
    "category": "page",
    "url": "/",
    "type": "page"
  },
  {
    "id": "indicators",
    "title": "All Indicators",
    "description": "Browse all technical analysis indicators",
    "category": "page",
    "url": "/indicators",
    "type": "page"
  }
];

export interface SearchItem {
  id: string;
  title: string;
  description: string;
  category: string;
  url: string;
  type: 'indicator' | 'guide' | 'page';
}
