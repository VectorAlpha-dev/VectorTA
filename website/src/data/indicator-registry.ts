// Auto-generated indicator registry
export const indicators = {
  "acosc": {
    "id": "acosc",
    "name": "Acceleration Oscillator",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "ad": {
    "id": "ad",
    "name": "Accumulation/Distribution",
    "category": "volume",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "adosc": {
    "id": "adosc",
    "name": "Adosc",
    "category": "volume",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "adx": {
    "id": "adx",
    "name": "Average Directional Index",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "plus_di",
      "minus_di"
    ],
    "description": ""
  },
  "adxr": {
    "id": "adxr",
    "name": "Adxr",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "plus_di",
      "minus_di"
    ],
    "description": ""
  },
  "alligator": {
    "id": "alligator",
    "name": "Alligator",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "ao": {
    "id": "ao",
    "name": "Awesome Oscillator",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "apo": {
    "id": "apo",
    "name": "Absolute Price Oscillator",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "aroon": {
    "id": "aroon",
    "name": "Aroon",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": "Create a new streaming Aroon from `params`.  Extracts `length = params.length.unwrap_or(14)`."
  },
  "aroonosc": {
    "id": "aroonosc",
    "name": "Aroon Oscillator",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "atr": {
    "id": "atr",
    "name": "Average True Range",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "bandpass": {
    "id": "bandpass",
    "name": "Band Pass Filter",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
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
        "min": 2,
        "max": 200,
        "description": "Moving average period"
      },
      {
        "name": "std_dev",
        "type": "number",
        "default": 2,
        "min": 0.1,
        "max": 5,
        "description": "Standard deviation multiplier"
      }
    ],
    "outputs": [
      "upper",
      "middle",
      "lower"
    ],
    "description": ""
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
        "min": 2,
        "max": 200,
        "description": "Moving average period"
      },
      {
        "name": "std_dev",
        "type": "number",
        "default": 2,
        "min": 0.1,
        "max": 5,
        "description": "Standard deviation multiplier"
      }
    ],
    "outputs": [
      "upper",
      "middle",
      "lower"
    ],
    "description": ""
  },
  "bop": {
    "id": "bop",
    "name": "Balance Of Power",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "cci": {
    "id": "cci",
    "name": "Commodity Channel Index",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "cfo": {
    "id": "cfo",
    "name": "Chande Forcast Oscillator",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "cg": {
    "id": "cg",
    "name": "Cg",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "chande": {
    "id": "chande",
    "name": "Chande",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "chop": {
    "id": "chop",
    "name": "Chop",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "cksp": {
    "id": "cksp",
    "name": "Cksp",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "cmo": {
    "id": "cmo",
    "name": "Chande Momentum Oscillator",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "coppock": {
    "id": "coppock",
    "name": "Coppock Curve",
    "category": "other",
    "parameters": [
      {
        "name": "k_period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "%K period"
      },
      {
        "name": "d_period",
        "type": "number",
        "default": 3,
        "min": 1,
        "max": 200,
        "description": "%D period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "correlation_cycle": {
    "id": "correlation_cycle",
    "name": "Correlation Cycle",
    "category": "cycle",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "correl_hl": {
    "id": "correl_hl",
    "name": "Correl Hl",
    "category": "statistics",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "cvi": {
    "id": "cvi",
    "name": "Cvi",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "damiani_volatmeter": {
    "id": "damiani_volatmeter",
    "name": "Damiani Volatmeter",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "decycler": {
    "id": "decycler",
    "name": "Decycler",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "dec_osc": {
    "id": "dec_osc",
    "name": "Dec Osc",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "deviation": {
    "id": "deviation",
    "name": "Deviation",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 2,
        "max": 200,
        "description": "Moving average period"
      },
      {
        "name": "std_dev",
        "type": "number",
        "default": 2,
        "min": 0.1,
        "max": 5,
        "description": "Standard deviation multiplier"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "devstop": {
    "id": "devstop",
    "name": "Devstop",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "di": {
    "id": "di",
    "name": "Di",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "plus_di",
      "minus_di"
    ],
    "description": ""
  },
  "dm": {
    "id": "dm",
    "name": "Dm",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "donchian": {
    "id": "donchian",
    "name": "Donchian Channels",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "upper",
      "middle",
      "lower"
    ],
    "description": ""
  },
  "dpo": {
    "id": "dpo",
    "name": "Detrended Price Oscillator",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "dti": {
    "id": "dti",
    "name": "Dti",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "dx": {
    "id": "dx",
    "name": "Dx",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "plus_di",
      "minus_di"
    ],
    "description": ""
  },
  "efi": {
    "id": "efi",
    "name": "Efi",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "emd": {
    "id": "emd",
    "name": "Emd",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "upper",
      "middle",
      "lower"
    ],
    "description": ""
  },
  "emv": {
    "id": "emv",
    "name": "Emv",
    "category": "volume",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "er": {
    "id": "er",
    "name": "Er",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "eri": {
    "id": "eri",
    "name": "Elder Ray Index",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "fisher": {
    "id": "fisher",
    "name": "Fisher",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "fosc": {
    "id": "fosc",
    "name": "Forecast Oscillator",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "gatorosc": {
    "id": "gatorosc",
    "name": "Gatorosc",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "upper",
      "middle",
      "lower"
    ],
    "description": ""
  },
  "heikin_ashi_candles": {
    "id": "heikin_ashi_candles",
    "name": "Heikin Ashi Candles",
    "category": "pattern",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "ht_dcperiod": {
    "id": "ht_dcperiod",
    "name": "Ht Dcperiod",
    "category": "cycle",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "ht_dcphase": {
    "id": "ht_dcphase",
    "name": "Ht Dcphase",
    "category": "cycle",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "ht_phasor": {
    "id": "ht_phasor",
    "name": "Ht Phasor",
    "category": "cycle",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "ht_sine": {
    "id": "ht_sine",
    "name": "Ht Sine",
    "category": "cycle",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "ht_trendline": {
    "id": "ht_trendline",
    "name": "Ht Trendline",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "ht_trendmode": {
    "id": "ht_trendmode",
    "name": "Ht Trendmode",
    "category": "cycle",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "ift_rsi": {
    "id": "ift_rsi",
    "name": "Inverse Fisher Transform RSI",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "kaufmanstop": {
    "id": "kaufmanstop",
    "name": "Kaufmanstop",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "kdj": {
    "id": "kdj",
    "name": "Kdj",
    "category": "other",
    "parameters": [
      {
        "name": "k_period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "%K period"
      },
      {
        "name": "d_period",
        "type": "number",
        "default": 3,
        "min": 1,
        "max": 200,
        "description": "%D period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "keltner": {
    "id": "keltner",
    "name": "Keltner Channels",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "upper",
      "middle",
      "lower"
    ],
    "description": ""
  },
  "kst": {
    "id": "kst",
    "name": "Know Sure Thing",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "kurtosis": {
    "id": "kurtosis",
    "name": "Kurtosis",
    "category": "statistics",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "kvo": {
    "id": "kvo",
    "name": "Kvo",
    "category": "volume",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "linearreg_angle": {
    "id": "linearreg_angle",
    "name": "Linearreg Angle",
    "category": "statistics",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "linearreg_intercept": {
    "id": "linearreg_intercept",
    "name": "Linearreg Intercept",
    "category": "statistics",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "linearreg_slope": {
    "id": "linearreg_slope",
    "name": "Linearreg Slope",
    "category": "statistics",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "lrsi": {
    "id": "lrsi",
    "name": "Lrsi",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "mab": {
    "id": "mab",
    "name": "Mab",
    "category": "other",
    "parameters": [
      {
        "name": "fast_period",
        "type": "number",
        "default": 12,
        "min": 2,
        "max": 200,
        "description": "Fast period"
      },
      {
        "name": "slow_period",
        "type": "number",
        "default": 26,
        "min": 2,
        "max": 200,
        "description": "Slow period"
      }
    ],
    "outputs": [
      "upper",
      "middle",
      "lower"
    ],
    "description": ""
  },
  "macd": {
    "id": "macd",
    "name": "Moving Average Convergence Divergence",
    "category": "momentum",
    "parameters": [
      {
        "name": "fast_period",
        "type": "number",
        "default": 12,
        "min": 2,
        "max": 200,
        "description": "Fast EMA period"
      },
      {
        "name": "slow_period",
        "type": "number",
        "default": 26,
        "min": 2,
        "max": 200,
        "description": "Slow EMA period"
      },
      {
        "name": "signal_period",
        "type": "number",
        "default": 9,
        "min": 2,
        "max": 200,
        "description": "Signal line period"
      }
    ],
    "outputs": [
      "macd",
      "signal",
      "histogram"
    ],
    "description": ""
  },
  "marketefi": {
    "id": "marketefi",
    "name": "Marketefi",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "mass": {
    "id": "mass",
    "name": "Mass Index",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "mean_ad": {
    "id": "mean_ad",
    "name": "Mean Ad",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "medium_ad": {
    "id": "medium_ad",
    "name": "Medium Ad",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "medprice": {
    "id": "medprice",
    "name": "Medprice",
    "category": "price_transform",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "mfi": {
    "id": "mfi",
    "name": "Money Flow Index",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "midpoint": {
    "id": "midpoint",
    "name": "Midpoint",
    "category": "price_transform",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "midprice": {
    "id": "midprice",
    "name": "Midprice",
    "category": "price_transform",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "minmax": {
    "id": "minmax",
    "name": "Minmax",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "mom": {
    "id": "mom",
    "name": "Mom",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "msw": {
    "id": "msw",
    "name": "Msw",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "natr": {
    "id": "natr",
    "name": "Natr",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "nvi": {
    "id": "nvi",
    "name": "Nvi",
    "category": "volume",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "obv": {
    "id": "obv",
    "name": "On Balance Volume",
    "category": "volume",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "pattern_recognition": {
    "id": "pattern_recognition",
    "name": "Pattern Recognition",
    "category": "pattern",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "upper",
      "middle",
      "lower"
    ],
    "description": ""
  },
  "pfe": {
    "id": "pfe",
    "name": "Polarized Fractal Efficiency",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "pivot": {
    "id": "pivot",
    "name": "Pivot Points",
    "category": "support_resistance",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "pma": {
    "id": "pma",
    "name": "Pma",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "ppo": {
    "id": "ppo",
    "name": "Percentage Price Oscillator",
    "category": "momentum",
    "parameters": [
      {
        "name": "fast_period",
        "type": "number",
        "default": 12,
        "min": 2,
        "max": 200,
        "description": "Fast period"
      },
      {
        "name": "slow_period",
        "type": "number",
        "default": 26,
        "min": 2,
        "max": 200,
        "description": "Slow period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "pvi": {
    "id": "pvi",
    "name": "Pvi",
    "category": "volume",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "qstick": {
    "id": "qstick",
    "name": "Qstick",
    "category": "other",
    "parameters": [
      {
        "name": "k_period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "%K period"
      },
      {
        "name": "d_period",
        "type": "number",
        "default": 3,
        "min": 1,
        "max": 200,
        "description": "%D period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "roc": {
    "id": "roc",
    "name": "Rate of Change",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "rocp": {
    "id": "rocp",
    "name": "Rate of Change Percentage",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "rocr": {
    "id": "rocr",
    "name": "Rate of Change Ratio",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "rsi": {
    "id": "rsi",
    "name": "Relative Strength Index",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "rsmk": {
    "id": "rsmk",
    "name": "Rsmk",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "rsx": {
    "id": "rsx",
    "name": "Relative Strength Index - Smoothed",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "rvi": {
    "id": "rvi",
    "name": "Relative Volatility Index",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 20,
        "min": 2,
        "max": 200,
        "description": "Moving average period"
      },
      {
        "name": "std_dev",
        "type": "number",
        "default": 2,
        "min": 0.1,
        "max": 5,
        "description": "Standard deviation multiplier"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "safezonestop": {
    "id": "safezonestop",
    "name": "Safezonestop",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "sar": {
    "id": "sar",
    "name": "Parabolic SAR",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "squeeze_momentum": {
    "id": "squeeze_momentum",
    "name": "Squeeze Momentum",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "upper",
      "middle",
      "lower"
    ],
    "description": ""
  },
  "srsi": {
    "id": "srsi",
    "name": "Srsi",
    "category": "other",
    "parameters": [
      {
        "name": "k_period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "%K period"
      },
      {
        "name": "d_period",
        "type": "number",
        "default": 3,
        "min": 1,
        "max": 200,
        "description": "%D period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "stc": {
    "id": "stc",
    "name": "Schaff Trend Cycle",
    "category": "momentum",
    "parameters": [
      {
        "name": "fast_period",
        "type": "number",
        "default": 12,
        "min": 2,
        "max": 200,
        "description": "Fast period"
      },
      {
        "name": "slow_period",
        "type": "number",
        "default": 26,
        "min": 2,
        "max": 200,
        "description": "Slow period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "stddev": {
    "id": "stddev",
    "name": "Stddev",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "stoch": {
    "id": "stoch",
    "name": "Stochastic Oscillator",
    "category": "momentum",
    "parameters": [
      {
        "name": "k_period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "%K period"
      },
      {
        "name": "d_period",
        "type": "number",
        "default": 3,
        "min": 1,
        "max": 200,
        "description": "%D period"
      }
    ],
    "outputs": [
      "k",
      "d"
    ],
    "description": ""
  },
  "stochf": {
    "id": "stochf",
    "name": "Stochf",
    "category": "momentum",
    "parameters": [
      {
        "name": "k_period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "%K period"
      },
      {
        "name": "d_period",
        "type": "number",
        "default": 3,
        "min": 1,
        "max": 200,
        "description": "%D period"
      }
    ],
    "outputs": [
      "k",
      "d"
    ],
    "description": ""
  },
  "supertrend": {
    "id": "supertrend",
    "name": "Supertrend",
    "category": "trend",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "upper",
      "middle",
      "lower"
    ],
    "description": ""
  },
  "trix": {
    "id": "trix",
    "name": "Triple Exponential Average",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "tsf": {
    "id": "tsf",
    "name": "Tsf",
    "category": "statistics",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "tsi": {
    "id": "tsi",
    "name": "True Strength Index",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "ttm_trend": {
    "id": "ttm_trend",
    "name": "Ttm Trend",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "ui": {
    "id": "ui",
    "name": "Ui",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "ultosc": {
    "id": "ultosc",
    "name": "Ultimate Oscillator",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "utility_functions": {
    "id": "utility_functions",
    "name": "Utility Functions",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "var": {
    "id": "var",
    "name": "Var",
    "category": "volatility",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "vi": {
    "id": "vi",
    "name": "Vortex Indicator",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "vidya": {
    "id": "vidya",
    "name": "Variable Index Dynamic Average",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "vlma": {
    "id": "vlma",
    "name": "Vlma",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "vosc": {
    "id": "vosc",
    "name": "Vosc",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "voss": {
    "id": "voss",
    "name": "Voss",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "vpci": {
    "id": "vpci",
    "name": "Vpci",
    "category": "volume",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "vpt": {
    "id": "vpt",
    "name": "Vpt",
    "category": "volume",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "vwmacd": {
    "id": "vwmacd",
    "name": "Vwmacd",
    "category": "volume",
    "parameters": [
      {
        "name": "fast_period",
        "type": "number",
        "default": 12,
        "min": 2,
        "max": 200,
        "description": "Fast EMA period"
      },
      {
        "name": "slow_period",
        "type": "number",
        "default": 26,
        "min": 2,
        "max": 200,
        "description": "Slow EMA period"
      },
      {
        "name": "signal_period",
        "type": "number",
        "default": 9,
        "min": 2,
        "max": 200,
        "description": "Signal line period"
      }
    ],
    "outputs": [
      "macd",
      "signal",
      "histogram"
    ],
    "description": ""
  },
  "wad": {
    "id": "wad",
    "name": "Wad",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "wavetrend": {
    "id": "wavetrend",
    "name": "WaveTrend",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "wclprice": {
    "id": "wclprice",
    "name": "Wclprice",
    "category": "price_transform",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "willr": {
    "id": "willr",
    "name": "Williams %R",
    "category": "momentum",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "zscore": {
    "id": "zscore",
    "name": "Zscore",
    "category": "other",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "alma": {
    "id": "alma",
    "name": "Arnaud Legoux Moving Average",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "cwma": {
    "id": "cwma",
    "name": "Cwma",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "dema": {
    "id": "dema",
    "name": "Double Exponential Moving Average",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "edcf": {
    "id": "edcf",
    "name": "Edcf",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "ehlers_itrend": {
    "id": "ehlers_itrend",
    "name": "Ehlers Itrend",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "ema": {
    "id": "ema",
    "name": "Exponential Moving Average",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "epma": {
    "id": "epma",
    "name": "Epma",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "frama": {
    "id": "frama",
    "name": "Fractal Adaptive Moving Average",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "fwma": {
    "id": "fwma",
    "name": "Fwma",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "gaussian": {
    "id": "gaussian",
    "name": "Gaussian Filter",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "highpass": {
    "id": "highpass",
    "name": "High Pass Filter",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "highpass_2_pole": {
    "id": "highpass_2_pole",
    "name": "Highpass 2 Pole",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "hma": {
    "id": "hma",
    "name": "Hull Moving Average",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "hwma": {
    "id": "hwma",
    "name": "Henderson Weighted Moving Average",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "jma": {
    "id": "jma",
    "name": "Jurik Moving Average",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "jsa": {
    "id": "jsa",
    "name": "Jsa",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "kama": {
    "id": "kama",
    "name": "Kaufman Adaptive Moving Average",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "upper",
      "middle",
      "lower"
    ],
    "description": ""
  },
  "linreg": {
    "id": "linreg",
    "name": "Linreg",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "ma": {
    "id": "ma",
    "name": "Ma",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "fast_period",
        "type": "number",
        "default": 12,
        "min": 2,
        "max": 200,
        "description": "Fast period"
      },
      {
        "name": "slow_period",
        "type": "number",
        "default": 26,
        "min": 2,
        "max": 200,
        "description": "Slow period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "maaq": {
    "id": "maaq",
    "name": "Maaq",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "fast_period",
        "type": "number",
        "default": 12,
        "min": 2,
        "max": 200,
        "description": "Fast period"
      },
      {
        "name": "slow_period",
        "type": "number",
        "default": 26,
        "min": 2,
        "max": 200,
        "description": "Slow period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "mama": {
    "id": "mama",
    "name": "MESA Adaptive Moving Average",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "mwdx": {
    "id": "mwdx",
    "name": "Mwdx",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "nma": {
    "id": "nma",
    "name": "Nma",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "pwma": {
    "id": "pwma",
    "name": "Pwma",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "reflex": {
    "id": "reflex",
    "name": "Reflex",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "sinwma": {
    "id": "sinwma",
    "name": "Sine Weighted Moving Average",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "sma": {
    "id": "sma",
    "name": "Simple Moving Average",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "smma": {
    "id": "smma",
    "name": "Smma",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "sqwma": {
    "id": "sqwma",
    "name": "Sqwma",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "srwma": {
    "id": "srwma",
    "name": "Srwma",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "supersmoother": {
    "id": "supersmoother",
    "name": "Super Smoother",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "supersmoother_3_pole": {
    "id": "supersmoother_3_pole",
    "name": "Supersmoother 3 Pole",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "swma": {
    "id": "swma",
    "name": "Symmetric Weighted Moving Average",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "tema": {
    "id": "tema",
    "name": "Triple Exponential Moving Average",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "tilson": {
    "id": "tilson",
    "name": "Tilson",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "trendflex": {
    "id": "trendflex",
    "name": "Trendflex",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "trima": {
    "id": "trima",
    "name": "Trima",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": "Feed a single new raw price into the TRIMA‐stream."
  },
  "vpwma": {
    "id": "vpwma",
    "name": "Vpwma",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "vwap": {
    "id": "vwap",
    "name": "Volume Weighted Average Price",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": "VWAP input data"
  },
  "vwma": {
    "id": "vwma",
    "name": "Volume Weighted Moving Average",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "wilders": {
    "id": "wilders",
    "name": "Wilders",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "wma": {
    "id": "wma",
    "name": "Weighted Moving Average",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  },
  "zlema": {
    "id": "zlema",
    "name": "Zero Lag Exponential Moving Average",
    "category": "moving_averages",
    "parameters": [
      {
        "name": "period",
        "type": "number",
        "default": 14,
        "min": 2,
        "max": 200,
        "description": "Lookback period"
      }
    ],
    "outputs": [
      "value"
    ],
    "description": ""
  }
} as const;

export type IndicatorId = keyof typeof indicators;
