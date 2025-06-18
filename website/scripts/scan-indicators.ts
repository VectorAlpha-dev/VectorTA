import { readdir, readFile } from 'fs/promises';
import { join } from 'path';
import { writeFileSync } from 'fs';

interface IndicatorInfo {
  id: string;
  name: string;
  category: string;
  subcategory?: string;
  description?: string;
  parameters: Array<{
    name: string;
    type: 'number' | 'boolean';
    default: any;
    min?: number;
    max?: number;
    description?: string;
  }>;
  outputs: string[];
}

async function scanIndicators() {
  const indicators: Record<string, IndicatorInfo> = {};
  
  // Scan main indicators directory
  const indicatorFiles = await readdir('../src/indicators');
  
  for (const file of indicatorFiles) {
    if (file.endsWith('.rs') && file !== 'mod.rs') {
      const content = await readFile(join('../src/indicators', file), 'utf-8');
      const info = extractIndicatorInfo(file.replace('.rs', ''), content);
      indicators[info.id] = info;
    }
  }
  
  // Scan moving_averages subdirectory
  const maFiles = await readdir('../src/indicators/moving_averages');
  for (const file of maFiles) {
    if (file.endsWith('.rs') && file !== 'mod.rs') {
      const content = await readFile(join('../src/indicators/moving_averages', file), 'utf-8');
      const info = extractIndicatorInfo(file.replace('.rs', ''), content, 'moving_averages');
      indicators[info.id] = info;
    }
  }
  
  // Write registry
  const output = `// Auto-generated indicator registry
export const indicators = ${JSON.stringify(indicators, null, 2)} as const;

export type IndicatorId = keyof typeof indicators;
`;
  
  writeFileSync('./src/data/indicator-registry.ts', output);
  console.log(`Generated registry with ${Object.keys(indicators).length} indicators`);
}

function extractIndicatorInfo(id: string, content: string, category?: string): IndicatorInfo {
  const nameMap: Record<string, string> = {
    'sma': 'Simple Moving Average',
    'ema': 'Exponential Moving Average',
    'rsi': 'Relative Strength Index',
    'macd': 'Moving Average Convergence Divergence',
    'bollinger_bands': 'Bollinger Bands',
    'atr': 'Average True Range',
    'adx': 'Average Directional Index',
    'stoch': 'Stochastic Oscillator',
    'cci': 'Commodity Channel Index',
    'willr': 'Williams %R',
    'roc': 'Rate of Change',
    'obv': 'On Balance Volume',
    'ad': 'Accumulation/Distribution',
    'mfi': 'Money Flow Index',
    'kst': 'Know Sure Thing',
    'trix': 'Triple Exponential Average',
    'ultosc': 'Ultimate Oscillator',
    'ao': 'Awesome Oscillator',
    'keltner': 'Keltner Channels',
    'donchian': 'Donchian Channels',
    'sar': 'Parabolic SAR',
    'supertrend': 'Supertrend',
    'vwap': 'Volume Weighted Average Price',
    'pivot': 'Pivot Points',
    'dema': 'Double Exponential Moving Average',
    'tema': 'Triple Exponential Moving Average',
    'wma': 'Weighted Moving Average',
    'hma': 'Hull Moving Average',
    'kama': 'Kaufman Adaptive Moving Average',
    'mama': 'MESA Adaptive Moving Average',
    'frama': 'Fractal Adaptive Moving Average',
    'vidya': 'Variable Index Dynamic Average',
    'alma': 'Arnaud Legoux Moving Average',
    'zlema': 'Zero Lag Exponential Moving Average',
    'hwma': 'Henderson Weighted Moving Average',
    'jma': 'Jurik Moving Average',
    'mcginley': 'McGinley Dynamic',
    'swma': 'Symmetric Weighted Moving Average',
    'vwma': 'Volume Weighted Moving Average',
    'sinwma': 'Sine Weighted Moving Average',
    'supersmoother': 'Super Smoother',
    'gaussian': 'Gaussian Filter',
    'highpass': 'High Pass Filter',
    'bandpass': 'Band Pass Filter',
    'ift_rsi': 'Inverse Fisher Transform RSI',
    'cfo': 'Chande Forcast Oscillator',
    'cmo': 'Chande Momentum Oscillator',
    'eri': 'Elder Ray Index',
    'mass': 'Mass Index',
    'dpo': 'Detrended Price Oscillator',
    'bop': 'Balance Of Power',
    'acosc': 'Acceleration Oscillator',
    'ao': 'Awesome Oscillator',
    'apo': 'Absolute Price Oscillator',
    'aroon': 'Aroon',
    'aroonosc': 'Aroon Oscillator',
    'coppock': 'Coppock Curve',
    'ppo': 'Percentage Price Oscillator',
    'roc': 'Rate of Change',
    'rocp': 'Rate of Change Percentage',
    'rocr': 'Rate of Change Ratio',
    'tsi': 'True Strength Index',
    'fosc': 'Forecast Oscillator',
    'pfe': 'Polarized Fractal Efficiency',
    'qstick': 'Qstick',
    'rvi': 'Relative Volatility Index',
    'rsx': 'Relative Strength Index - Smoothed',
    'rvgi': 'Relative Vigor Index',
    'stc': 'Schaff Trend Cycle',
    'vi': 'Vortex Indicator',
    'wavetrend': 'WaveTrend',
  };
  
  // Extract parameter information from function signatures
  const parameters = extractParameters(content);
  
  // Categorize based on file path or name patterns
  let derivedCategory = category || categorizeIndicator(id);
  
  return {
    id,
    name: nameMap[id] || id.replace(/[_-]/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    category: derivedCategory,
    parameters,
    outputs: extractOutputs(content),
    description: extractDescription(content)
  };
}

function categorizeIndicator(id: string): string {
  // Categorize based on indicator patterns
  const categories: Record<string, string[]> = {
    'momentum': ['rsi', 'stoch', 'stochf', 'willr', 'roc', 'rocp', 'rocr', 'mom', 'cci', 'cmo', 'mfi', 'tsi', 'ultosc', 'ao', 'apo', 'ppo', 'macd', 'pfe', 'rsx', 'rvgi', 'stc'],
    'trend': ['sma', 'ema', 'wma', 'dema', 'tema', 'adx', 'adxr', 'di', 'dm', 'dx', 'sar', 'supertrend', 'aroon', 'aroonosc', 'psar'],
    'volatility': ['atr', 'natr', 'bollinger_bands', 'bollinger_bands_width', 'keltner', 'donchian', 'stddev', 'var', 'cvol', 'mass'],
    'volume': ['obv', 'ad', 'adosc', 'vpt', 'vwap', 'nvi', 'pvi', 'mfi', 'emv', 'vwma', 'vpci', 'vwmacd', 'kvo'],
    'support_resistance': ['pivot', 'donchian', 'keltner', 'bollinger_bands'],
    'pattern': ['pattern_recognition', 'heikin_ashi_candles'],
    'cycle': ['ht_dcperiod', 'ht_dcphase', 'ht_phasor', 'ht_sine', 'ht_trendmode', 'correlation_cycle'],
    'statistics': ['linearreg_slope', 'linearreg_intercept', 'linearreg_angle', 'tsf', 'stddev', 'var', 'correl_hl', 'kurtosis'],
    'price_transform': ['medprice', 'typprice', 'wclprice', 'midpoint', 'midprice'],
  };
  
  for (const [category, indicators] of Object.entries(categories)) {
    if (indicators.includes(id)) {
      return category;
    }
  }
  
  return 'other';
}

function extractParameters(content: string): any[] {
  // Simple parameter extraction - looking for period/length parameters
  const defaultParams: Record<string, any[]> = {
    'period': [{ name: 'period', type: 'number', default: 14, min: 2, max: 200, description: 'Lookback period' }],
    'fast_slow': [
      { name: 'fast_period', type: 'number', default: 12, min: 2, max: 200, description: 'Fast period' },
      { name: 'slow_period', type: 'number', default: 26, min: 2, max: 200, description: 'Slow period' }
    ],
    'macd': [
      { name: 'fast_period', type: 'number', default: 12, min: 2, max: 200, description: 'Fast EMA period' },
      { name: 'slow_period', type: 'number', default: 26, min: 2, max: 200, description: 'Slow EMA period' },
      { name: 'signal_period', type: 'number', default: 9, min: 2, max: 200, description: 'Signal line period' }
    ],
    'bollinger': [
      { name: 'period', type: 'number', default: 20, min: 2, max: 200, description: 'Moving average period' },
      { name: 'std_dev', type: 'number', default: 2, min: 0.1, max: 5, description: 'Standard deviation multiplier' }
    ],
    'stoch': [
      { name: 'k_period', type: 'number', default: 14, min: 2, max: 200, description: '%K period' },
      { name: 'd_period', type: 'number', default: 3, min: 1, max: 200, description: '%D period' }
    ]
  };
  
  // Try to detect which parameters this indicator uses
  if (content.includes('fast_period') && content.includes('slow_period')) {
    if (content.includes('signal_period')) {
      return defaultParams['macd'];
    }
    return defaultParams['fast_slow'];
  } else if (content.includes('k_period') || content.includes('stoch')) {
    return defaultParams['stoch'];
  } else if (content.includes('std_dev') || content.includes('bollinger')) {
    return defaultParams['bollinger'];
  } else {
    return defaultParams['period'];
  }
}

function extractOutputs(content: string): string[] {
  // Most indicators return a single output
  if (content.includes('upper') && content.includes('lower')) {
    return ['upper', 'middle', 'lower'];
  } else if (content.includes('macd') && content.includes('signal')) {
    return ['macd', 'signal', 'histogram'];
  } else if (content.includes('k_values') || content.includes('%K')) {
    return ['k', 'd'];
  } else if (content.includes('plus_di') && content.includes('minus_di')) {
    return ['plus_di', 'minus_di'];
  }
  return ['value'];
}

function extractDescription(content: string): string {
  // Extract from doc comments if available
  const docMatch = content.match(/\/\/\/\s*(.+?)(?:\n|$)/);
  if (docMatch) {
    return docMatch[1].trim();
  }
  return '';
}

// Run the scanner
scanIndicators().catch(console.error);