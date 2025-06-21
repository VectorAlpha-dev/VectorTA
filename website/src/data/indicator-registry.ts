// Auto-generated indicator registry
export const indicators = {
  "acosc": {
    "id": "acosc",
    "name": "Acceleration Oscillator",
    "category": "momentum",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "ad": {
    "id": "ad",
    "name": "Accumulation/Distribution",
    "category": "volume",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "adosc": {
    "id": "adosc",
    "name": "Adosc",
    "category": "volume",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "adx": {
    "id": "adx",
    "name": "Average Directional Index",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "adxr": {
    "id": "adxr",
    "name": "Adxr",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "alligator": {
    "id": "alligator",
    "name": "Alligator",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "ao": {
    "id": "ao",
    "name": "Awesome Oscillator",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "apo": {
    "id": "apo",
    "name": "Absolute Price Oscillator",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "aroon": {
    "id": "aroon",
    "name": "Aroon",
    "category": "trend",
    "parameters": [],
    "description": "Create a new streaming Aroon from `params`.  Extracts `length = params.length.unwrap_or(14)`."
  },
  "aroonosc": {
    "id": "aroonosc",
    "name": "Aroonosc",
    "category": "trend",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "atr": {
    "id": "atr",
    "name": "Average True Range",
    "category": "volatility",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "avgprice": {
    "id": "avgprice",
    "name": "Average Price",
    "category": "price",
    "parameters": [],
    "description": "# Average Price"
  },
  "bandpass": {
    "id": "bandpass",
    "name": "Bandpass",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "bollinger_bands": {
    "id": "bollinger_bands",
    "name": "Bollinger Bands",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "bollinger_bands_width": {
    "id": "bollinger_bands_width",
    "name": "Bollinger Bands Width",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "bop": {
    "id": "bop",
    "name": "Balance of Power",
    "category": "trend",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "cci": {
    "id": "cci",
    "name": "Commodity Channel Index",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "cfo": {
    "id": "cfo",
    "name": "Chaikin Flow Oscillator",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "cg": {
    "id": "cg",
    "name": "Center of Gravity",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "chande": {
    "id": "chande",
    "name": "Chande Indicator",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "chop": {
    "id": "chop",
    "name": "Choppiness Index",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "cksp": {
    "id": "cksp",
    "name": "Chande Kroll Stop",
    "category": "support_resistance",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "cmo": {
    "id": "cmo",
    "name": "Chande Momentum Oscillator",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "coppock": {
    "id": "coppock",
    "name": "Coppock Curve",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "correl_hl": {
    "id": "correl_hl",
    "name": "High-Low Correlation",
    "category": "statistical",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "correlation_cycle": {
    "id": "correlation_cycle",
    "name": "Correlation Cycle",
    "category": "statistical",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Holds the (real, imag, angle, state) computed for the “previous” window,"
  },
  "cvi": {
    "id": "cvi",
    "name": "Cvi",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "damiani_volatmeter": {
    "id": "damiani_volatmeter",
    "name": "Damiani Volatmeter",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "dec_osc": {
    "id": "dec_osc",
    "name": "Detrended Oscillator",
    "category": "cycles",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "decycler": {
    "id": "decycler",
    "name": "Decycler",
    "category": "cycles",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "deviation": {
    "id": "deviation",
    "name": "Deviation",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Input for deviation indicator."
  },
  "devstop": {
    "id": "devstop",
    "name": "Deviation Stop",
    "category": "support_resistance",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "di": {
    "id": "di",
    "name": "Directional Indicator (+DI/-DI)",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "dm": {
    "id": "dm",
    "name": "Directional Movement",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "donchian": {
    "id": "donchian",
    "name": "Donchian Channel",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "dpo": {
    "id": "dpo",
    "name": "Detrended Price Oscillator",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "dti": {
    "id": "dti",
    "name": "Directional Trend Index",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "dx": {
    "id": "dx",
    "name": "Dx",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "efi": {
    "id": "efi",
    "name": "Efi",
    "category": "volume",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "emd": {
    "id": "emd",
    "name": "Empirical Mode Decomposition",
    "category": "volume",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "emv": {
    "id": "emv",
    "name": "Ease of Movement",
    "category": "volume",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "er": {
    "id": "er",
    "name": "Efficiency Ratio",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "eri": {
    "id": "eri",
    "name": "Elder Ray Index",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "fisher": {
    "id": "fisher",
    "name": "Fisher",
    "category": "volume",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "fosc": {
    "id": "fosc",
    "name": "Fosc",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "gatorosc": {
    "id": "gatorosc",
    "name": "Gatorosc",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "heikin_ashi_candles": {
    "id": "heikin_ashi_candles",
    "name": "Heikin Ashi Candles",
    "category": "price",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "ht_dcperiod": {
    "id": "ht_dcperiod",
    "name": "Hilbert Transform - Dominant Cycle Period",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "# Hilbert Transform - Dominant Cycle Period (HT_DCPERIOD)"
  },
  "ht_dcphase": {
    "id": "ht_dcphase",
    "name": "Hilbert Transform - Dominant Cycle Phase",
    "category": "volatility",
    "parameters": [],
    "description": "# Hilbert Transform - Dominant Cycle Phase (HT_DCPHASE)"
  },
  "ht_phasor": {
    "id": "ht_phasor",
    "name": "Hilbert Transform - Phasor",
    "category": "cycles",
    "parameters": [],
    "description": "# Hilbert Transform Phasor (HT_PHASOR)"
  },
  "ht_sine": {
    "id": "ht_sine",
    "name": "Hilbert Transform - SineWave",
    "category": "moving_averages",
    "parameters": [],
    "description": "# Hilbert Transform - SineWave (HT_SINE)"
  },
  "ht_trendline": {
    "id": "ht_trendline",
    "name": "Hilbert Transform - Instantaneous Trendline",
    "category": "volatility",
    "parameters": [],
    "description": "# HT_TRENDLINE (Hilbert Transform - Instantaneous Trendline)"
  },
  "ht_trendmode": {
    "id": "ht_trendmode",
    "name": "Hilbert Transform - Trend vs Cycle Mode",
    "category": "volatility",
    "parameters": [],
    "description": "# Hilbert Transform - Trend vs Cycle Mode (HT_TRENDMODE)"
  },
  "ift_rsi": {
    "id": "ift_rsi",
    "name": "Ift Rsi",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "kaufmanstop": {
    "id": "kaufmanstop",
    "name": "Kaufman Stop",
    "category": "support_resistance",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "kdj": {
    "id": "kdj",
    "name": "KDJ Indicator",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "keltner": {
    "id": "keltner",
    "name": "Keltner Channel",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "kst": {
    "id": "kst",
    "name": "Know Sure Thing",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "kurtosis": {
    "id": "kurtosis",
    "name": "Kurtosis",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "kvo": {
    "id": "kvo",
    "name": "Kvo",
    "category": "volume",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "linearreg_angle": {
    "id": "linearreg_angle",
    "name": "Linear Regression Angle",
    "category": "statistical",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "linearreg_intercept": {
    "id": "linearreg_intercept",
    "name": "Linear Regression Intercept",
    "category": "statistical",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "linearreg_slope": {
    "id": "linearreg_slope",
    "name": "Linear Regression Slope",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "lrsi": {
    "id": "lrsi",
    "name": "Lrsi",
    "category": "momentum",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "mab": {
    "id": "mab",
    "name": "Moving Average Bands",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "macd": {
    "id": "macd",
    "name": "Moving Average Convergence Divergence",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "marketefi": {
    "id": "marketefi",
    "name": "Marketefi",
    "category": "volume",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "mass": {
    "id": "mass",
    "name": "Mass",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "mean_ad": {
    "id": "mean_ad",
    "name": "Mean Ad",
    "category": "volume",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "medium_ad": {
    "id": "medium_ad",
    "name": "Medium Ad",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "medprice": {
    "id": "medprice",
    "name": "Median Price",
    "category": "price",
    "parameters": [],
    "description": "Source data for medprice indicator."
  },
  "mfi": {
    "id": "mfi",
    "name": "Money Flow Index",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "midpoint": {
    "id": "midpoint",
    "name": "Midpoint",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "midprice": {
    "id": "midprice",
    "name": "Midprice",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "minmax": {
    "id": "minmax",
    "name": "Minmax",
    "category": "trend",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "mom": {
    "id": "mom",
    "name": "Momentum",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "msw": {
    "id": "msw",
    "name": "Mesa Sine Wave",
    "category": "cycles",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "natr": {
    "id": "natr",
    "name": "Normalized Average True Range",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "nvi": {
    "id": "nvi",
    "name": "Negative Volume Index",
    "category": "volume",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "obv": {
    "id": "obv",
    "name": "On Balance Volume",
    "category": "volume",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "pattern_recognition": {
    "id": "pattern_recognition",
    "name": "Pattern Recognition",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "pfe": {
    "id": "pfe",
    "name": "Polarized Fractal Efficiency",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Pushes one new price into the stream.  Returns `None` until we have"
  },
  "pivot": {
    "id": "pivot",
    "name": "Pivot Points",
    "category": "volume",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "pma": {
    "id": "pma",
    "name": "Pivot Moving Average",
    "category": "moving_averages",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "ppo": {
    "id": "ppo",
    "name": "Percentage Price Oscillator",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Update the stream with a new value and return the latest PPO if available."
  },
  "pvi": {
    "id": "pvi",
    "name": "Positive Volume Index",
    "category": "volume",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "qstick": {
    "id": "qstick",
    "name": "Qstick",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "roc": {
    "id": "roc",
    "name": "Rate of Change",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "rocp": {
    "id": "rocp",
    "name": "Rocp",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "rocr": {
    "id": "rocr",
    "name": "Rocr",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "rsi": {
    "id": "rsi",
    "name": "Relative Strength Index",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "rsmk": {
    "id": "rsmk",
    "name": "Relative Strength Market",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "rsx": {
    "id": "rsx",
    "name": "Relative Strength Index Smoothed",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "rvi": {
    "id": "rvi",
    "name": "Rvi",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "safezonestop": {
    "id": "safezonestop",
    "name": "SafeZone Stop",
    "category": "support_resistance",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "sar": {
    "id": "sar",
    "name": "Parabolic SAR",
    "category": "trend",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "squeeze_momentum": {
    "id": "squeeze_momentum",
    "name": "Squeeze Momentum",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "srsi": {
    "id": "srsi",
    "name": "Stochastic RSI",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "stc": {
    "id": "stc",
    "name": "Schaff Trend Cycle",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "stddev": {
    "id": "stddev",
    "name": "Standard Deviation",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "stoch": {
    "id": "stoch",
    "name": "Stochastic Oscillator",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "stochf": {
    "id": "stochf",
    "name": "Stochf",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "supertrend": {
    "id": "supertrend",
    "name": "Supertrend",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "trix": {
    "id": "trix",
    "name": "Trix",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "tsf": {
    "id": "tsf",
    "name": "Time Series Forecast",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "tsi": {
    "id": "tsi",
    "name": "True Strength Index",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "ttm_trend": {
    "id": "ttm_trend",
    "name": "Ttm Trend",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "ui": {
    "id": "ui",
    "name": "Ulcer Index",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "ultosc": {
    "id": "ultosc",
    "name": "Ultimate Oscillator",
    "category": "momentum",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "utility_functions": {
    "id": "utility_functions",
    "name": "Utility Functions",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "var": {
    "id": "var",
    "name": "Var",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "vi": {
    "id": "vi",
    "name": "Vortex Indicator",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "vidya": {
    "id": "vidya",
    "name": "Variable Index Dynamic Average",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "vlma": {
    "id": "vlma",
    "name": "Variable Length Moving Average",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "vosc": {
    "id": "vosc",
    "name": "Vosc",
    "category": "volume",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "voss": {
    "id": "voss",
    "name": "Voss",
    "category": "volume",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "vpci": {
    "id": "vpci",
    "name": "Volume Price Confirmation Indicator",
    "category": "volume",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "vpt": {
    "id": "vpt",
    "name": "Volume Price Trend",
    "category": "volume",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "vwmacd": {
    "id": "vwmacd",
    "name": "Vwmacd",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "wad": {
    "id": "wad",
    "name": "Wad",
    "category": "volume",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "wavetrend": {
    "id": "wavetrend",
    "name": "Wavetrend",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Push one new `price`.  Returns `None` (→ “NaN” in the test harness) until"
  },
  "wclprice": {
    "id": "wclprice",
    "name": "Weighted Close Price",
    "category": "price",
    "parameters": [],
    "description": "Technical analysis indicator"
  },
  "willr": {
    "id": "willr",
    "name": "Williams %R",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  },
  "zscore": {
    "id": "zscore",
    "name": "Z-Score",
    "category": "statistical",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 1,
        "max": 500,
        "description": "Lookback period"
      }
    ],
    "description": "Technical analysis indicator"
  }
};

export type IndicatorId = keyof typeof indicators;
