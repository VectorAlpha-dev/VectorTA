import { readdir, readFile, writeFile } from 'fs/promises';
import { join } from 'path';

async function scanIndicators() {
  const indicators = {};
  
  try {
    // Scan indicators from parent directory's Rust source
    const srcPath = '../src/indicators';
    const files = await readdir(srcPath);
    
    for (const file of files) {
      if (file.endsWith('.rs') && file !== 'mod.rs') {
        const name = file.replace('.rs', '');
        const content = await readFile(join(srcPath, file), 'utf-8');
        
        // Extract basic info from filename
        indicators[name] = {
          id: name,
          name: formatIndicatorName(name),
          category: categorizeIndicator(name),
          parameters: extractParameters(content),
          description: extractDescription(content)
        };
      }
    }
    
    // Write registry
    const registryContent = `// Auto-generated indicator registry
export const indicators = ${JSON.stringify(indicators, null, 2)};

export type IndicatorId = keyof typeof indicators;
`;
    
    await writeFile('./src/data/indicator-registry.ts', registryContent);
    console.log(`Generated registry with ${Object.keys(indicators).length} indicators`);
    
  } catch (error) {
    console.error('Error scanning indicators:', error);
    // Create a minimal registry if scanning fails
    const fallbackRegistry = `// Auto-generated indicator registry
export const indicators = {
  sma: { id: 'sma', name: 'Simple Moving Average', category: 'moving_averages', parameters: [], description: 'Calculates the simple moving average' },
  ema: { id: 'ema', name: 'Exponential Moving Average', category: 'moving_averages', parameters: [], description: 'Calculates the exponential moving average' },
  rsi: { id: 'rsi', name: 'Relative Strength Index', category: 'momentum', parameters: [], description: 'Calculates the relative strength index' }
};

export type IndicatorId = keyof typeof indicators;
`;
    await writeFile('./src/data/indicator-registry.ts', fallbackRegistry);
    console.log('Created fallback registry with 3 indicators');
  }
}

function formatIndicatorName(id) {
  const specialNames = {
    // Moving Averages
    'sma': 'Simple Moving Average',
    'ema': 'Exponential Moving Average',
    'wma': 'Weighted Moving Average',
    'dema': 'Double Exponential Moving Average',
    'tema': 'Triple Exponential Moving Average',
    'hma': 'Hull Moving Average',
    'kama': 'Kaufman Adaptive Moving Average',
    'alma': 'Arnaud Legoux Moving Average',
    'vwma': 'Volume Weighted Moving Average',
    'smma': 'Smoothed Moving Average',
    'frama': 'Fractal Adaptive Moving Average',
    'vidya': 'Variable Index Dynamic Average',
    'zlema': 'Zero Lag Exponential Moving Average',
    't3': 'T3 Moving Average',
    'trima': 'Triangular Moving Average',
    'vlma': 'Variable Length Moving Average',
    
    // Momentum Indicators
    'rsi': 'Relative Strength Index',
    'macd': 'Moving Average Convergence Divergence',
    'stoch': 'Stochastic Oscillator',
    'cci': 'Commodity Channel Index',
    'mom': 'Momentum',
    'roc': 'Rate of Change',
    'williams_r': 'Williams %R',
    'willr': 'Williams %R',
    'tsi': 'True Strength Index',
    'uo': 'Ultimate Oscillator',
    'ultosc': 'Ultimate Oscillator',
    'ao': 'Awesome Oscillator',
    'kst': 'Know Sure Thing',
    'ppo': 'Percentage Price Oscillator',
    'apo': 'Absolute Price Oscillator',
    'cmo': 'Chande Momentum Oscillator',
    'rmi': 'Relative Momentum Index',
    'srsi': 'Stochastic RSI',
    'rsx': 'Relative Strength Index Smoothed',
    'kdj': 'KDJ Indicator',
    'cfo': 'Chaikin Flow Oscillator',
    'cg': 'Center of Gravity',
    'dti': 'Directional Trend Index',
    'eri': 'Elder Ray Index',
    'ui': 'Ulcer Index',
    'di': 'Directional Indicator (+DI/-DI)',
    
    // Trend Indicators
    'adx': 'Average Directional Index',
    'aroon': 'Aroon',
    'psar': 'Parabolic SAR',
    'sar': 'Parabolic SAR',
    'supertrend': 'Supertrend',
    'vortex': 'Vortex Indicator',
    'vi': 'Vortex Indicator',
    'dpo': 'Detrended Price Oscillator',
    'mi': 'Mass Index',
    'bop': 'Balance of Power',
    'coppock': 'Coppock Curve',
    'tsf': 'Time Series Forecast',
    'pfe': 'Polarized Fractal Efficiency',
    'stc': 'Schaff Trend Cycle',
    
    // Volatility Indicators
    'atr': 'Average True Range',
    'bb': 'Bollinger Bands',
    'bollinger_bands': 'Bollinger Bands',
    'kc': 'Keltner Channel',
    'keltner': 'Keltner Channel',
    'dc': 'Donchian Channel',
    'donchian': 'Donchian Channel',
    'stddev': 'Standard Deviation',
    'variance': 'Variance',
    'cv': 'Coefficient of Variation',
    'hv': 'Historical Volatility',
    'natr': 'Normalized Average True Range',
    'chop': 'Choppiness Index',
    'deviation': 'Deviation',
    'kurtosis': 'Kurtosis',
    
    // Volume Indicators
    'obv': 'On Balance Volume',
    'ad': 'Accumulation/Distribution',
    'mfi': 'Money Flow Index',
    'vwap': 'Volume Weighted Average Price',
    'vpt': 'Volume Price Trend',
    'nvi': 'Negative Volume Index',
    'pvi': 'Positive Volume Index',
    'pvt': 'Price Volume Trend',
    'emv': 'Ease of Movement',
    'fi': 'Force Index',
    'klinger': 'Klinger Oscillator',
    'emd': 'Empirical Mode Decomposition',
    'vpci': 'Volume Price Confirmation Indicator',
    
    // Support/Resistance & Stops
    'pivot': 'Pivot Points',
    'fibonacci': 'Fibonacci Retracement',
    'murrey': 'Murrey Math Lines',
    'camarilla': 'Camarilla Pivot Points',
    'woodie': 'Woodie Pivot Points',
    'demark': 'DeMark Pivot Points',
    'devstop': 'Deviation Stop',
    'kaufmanstop': 'Kaufman Stop',
    'safezonestop': 'SafeZone Stop',
    
    // Price Transformations
    'heikin_ashi': 'Heikin Ashi',
    'heikin_ashi_candles': 'Heikin Ashi Candles',
    'avgprice': 'Average Price',
    'medprice': 'Median Price',
    'wclprice': 'Weighted Close Price',
    'hlcc4': 'HLCC/4 Price',
    'ohlc4': 'OHLC/4 Price',
    
    // Statistical
    'zscore': 'Z-Score',
    'correlation': 'Correlation',
    'correlation_cycle': 'Correlation Cycle',
    'linearreg': 'Linear Regression',
    'linearreg_angle': 'Linear Regression Angle',
    'linearreg_intercept': 'Linear Regression Intercept',
    'linearreg_slope': 'Linear Regression Slope',
    
    // Cycles
    'ht_dcperiod': 'Hilbert Transform - Dominant Cycle Period',
    'ht_dcphase': 'Hilbert Transform - Dominant Cycle Phase',
    'ht_phasor': 'Hilbert Transform - Phasor',
    'ht_sine': 'Hilbert Transform - SineWave',
    'ht_trendline': 'Hilbert Transform - Instantaneous Trendline',
    'ht_trendmode': 'Hilbert Transform - Trend vs Cycle Mode',
    'decycler': 'Decycler',
    'dec_osc': 'Detrended Oscillator',
    
    // Other indicators
    'rsmk': 'Relative Strength Market',
    'pma': 'Pivot Moving Average',
    'msw': 'Mesa Sine Wave',
    'chande': 'Chande Indicator',
    'dm': 'Directional Movement',
    'er': 'Efficiency Ratio',
    'mab': 'Moving Average Bands',
    'acosc': 'Acceleration Oscillator',
    'correl_hl': 'High-Low Correlation',
    'cksp': 'Chande Kroll Stop'
  };
  
  return specialNames[id] || id.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function categorizeIndicator(name) {
  // Moving Averages
  const movingAverages = [
    'sma', 'ema', 'wma', 'dema', 'tema', 'hma', 'kama', 'alma', 'vwma', 
    'smma', 'frama', 'vidya', 'zlema', 't3', 'trima', 'mcgd', 'hull', 
    'lwma', 'vwma', 'swma', 'mama', 'fama', 'gaussian', 'sine', 'ehlers',
    'lsma', 'covma', 'rma', 'mma', 'tma', 'gmma', 'edma'
  ];
  
  // Momentum Indicators  
  const momentum = [
    'rsi', 'macd', 'stoch', 'stochastic', 'cci', 'mom', 'momentum', 'roc', 
    'williams_r', 'williams', 'tsi', 'uo', 'ultimate', 'ao', 'awesome', 'kst', 
    'ppo', 'cmo', 'rmi', 'srsi', 'stochrsi', 'mfi', 'cog', 'qstick', 'trix',
    'elder_ray', 'elder', 'dmi', 'di', 'dx', 'divergence', 'fosc', 'gator',
    'macd_histogram', 'macdh', 'mi_', 'pmo', 'roi', 'rvgi', 'smi', 'squeeze'
  ];
  
  // Trend Indicators
  const trend = [
    'adx', 'aroon', 'psar', 'parabolic', 'supertrend', 'vortex', 'dpo', 
    'mi', 'mass', 'bop', 'balance', 'coppock', 'chandelier', 'trend_strength',
    'trend_intensity', 'trend_flex', 'asi', 'alligator', 'fractals', 'gann',
    'hhll', 'highest', 'lowest', 'lookback', 'rainbow', 'schaff', 'slope',
    'ttm', 'zigzag', 'swing', 'trend_', 'directional', 'adxr', 'vhf', 'aroon_oscillator'
  ];
  
  // Volatility Indicators
  const volatility = [
    'atr', 'bb', 'bollinger', 'kc', 'keltner', 'dc', 'donchian', 'stddev', 
    'std', 'variance', 'var', 'cv', 'hv', 'historical', 'natr', 'chop', 
    'choppiness', 'acceleration', 'bands', 'envelope', 'channel', 'projection',
    'range', 'true_range', 'tr', 'wtr', 'rvi', 'ulcer', 'vertical', 'vola',
    'volatility', 'vix', 'garch', 'yang_zhang', 'rogers', 'garman', 'parkinson'
  ];
  
  // Volume Indicators
  const volume = [
    'obv', 'ad', 'accumulation', 'distribution', 'mfi', 'money', 'vwap', 
    'vpt', 'volume_price', 'nvi', 'negative_volume', 'pvi', 'positive_volume', 
    'pvt', 'price_volume', 'emv', 'ease', 'fi', 'force', 'klinger', 'chaikin',
    'cmf', 'cvi', 'vo', 'volume_', 'vr', 'volume_ratio', 'vwma', 'vzo', 'wad',
    'williams_ad', 'twiggs', 'volume_oscillator', 'volume_trend', 'vfi'
  ];
  
  // Support/Resistance
  const supportResistance = [
    'pivot', 'fibonacci', 'fib', 'murrey', 'camarilla', 'woodie', 'demark', 
    'floor', 'cpr', 'support', 'resistance', 'levels', 'retracement', 'extension'
  ];
  
  // Pattern Recognition
  const patterns = [
    'candlestick', 'doji', 'hammer', 'shooting_star', 'engulfing', 'harami',
    'piercing', 'dark_cloud', 'morning_star', 'evening_star', 'three_soldiers',
    'three_crows', 'pattern', 'chart_pattern', 'harmonic', 'elliott', 'wave'
  ];
  
  // Market Breadth
  const breadth = [
    'advance_decline', 'ad_line', 'ad_ratio', 'mcclellan', 'summation', 'trin',
    'arms', 'thrust', 'breadth', 'tick', 'new_highs', 'new_lows', 'up_down'
  ];
  
  // Check each category
  const lowerName = name.toLowerCase();
  
  if (movingAverages.some(ma => lowerName.includes(ma))) return 'moving_averages';
  if (momentum.some(m => lowerName.includes(m))) return 'momentum';
  if (trend.some(t => lowerName.includes(t))) return 'trend';
  if (volatility.some(v => lowerName.includes(v))) return 'volatility';
  if (volume.some(v => lowerName.includes(v))) return 'volume';
  if (supportResistance.some(sr => lowerName.includes(sr))) return 'support_resistance';
  if (patterns.some(p => lowerName.includes(p))) return 'patterns';
  if (breadth.some(b => lowerName.includes(b))) return 'breadth';
  
  // Specific indicators that don't match patterns
  const specificMappings = {
    // Momentum
    'apo': 'momentum', 'cfo': 'momentum', 'cg': 'momentum', 'dti': 'momentum', 
    'eri': 'momentum', 'kdj': 'momentum', 'rsx': 'momentum', 'ui': 'momentum',
    'willr': 'momentum', 'ultosc': 'momentum', 'acosc': 'momentum',
    
    // Trend  
    'sar': 'trend', 'tsf': 'trend', 'pfe': 'trend', 'stc': 'trend', 'vi': 'trend',
    'dm': 'trend', 'er': 'trend', 'chande': 'trend',
    
    // Volatility
    'deviation': 'volatility', 'kurtosis': 'volatility', 'mab': 'volatility',
    
    // Volume
    'emd': 'volume', 'vpci': 'volume',
    
    // Support/Resistance & Stops
    'devstop': 'support_resistance', 'kaufmanstop': 'support_resistance', 
    'safezonestop': 'support_resistance', 'cksp': 'support_resistance',
    
    // Price Transformations
    'avgprice': 'price', 'medprice': 'price', 'wclprice': 'price',
    'heikin_ashi_candles': 'price',
    
    // Statistical
    'zscore': 'statistical', 'correlation_cycle': 'statistical',
    'linearreg_angle': 'statistical', 'linearreg_intercept': 'statistical',
    'correl_hl': 'statistical',
    
    // Cycles
    'ht_phasor': 'cycles', 'decycler': 'cycles', 'dec_osc': 'cycles',
    'msw': 'cycles',
    
    // Moving Averages
    'vlma': 'moving_averages', 'pma': 'moving_averages',
    
    // Others that truly don't fit
    'rsmk': 'other', 'utility_functions': 'other'
  };
  
  if (specificMappings[lowerName]) return specificMappings[lowerName];
  
  // Additional checks for common indicator types
  if (lowerName.includes('oscillator')) return 'momentum';
  if (lowerName.includes('average') && !lowerName.includes('true_range')) return 'moving_averages';
  if (lowerName.includes('band') || lowerName.includes('channel')) return 'volatility';
  if (lowerName.includes('flow') || lowerName.includes('volume')) return 'volume';
  if (lowerName.includes('trend')) return 'trend';
  if (lowerName.includes('stop')) return 'support_resistance';
  if (lowerName.includes('price') && !lowerName.includes('oscillator')) return 'price';
  if (lowerName.includes('regression') || lowerName.includes('correlation')) return 'statistical';
  
  return 'other';
}

function extractParameters(content) {
  // Simple parameter extraction - would need enhancement for real use
  const params = [];
  if (content.includes('period:')) {
    params.push({
      name: 'period',
      type: 'number',
      default: 20,
      min: 1,
      max: 500,
      description: 'Lookback period'
    });
  }
  return params;
}

function extractDescription(content) {
  // Extract from doc comments
  const docMatch = content.match(/\/\/\/\s*(.+)/);
  return docMatch ? docMatch[1] : 'Technical analysis indicator';
}

scanIndicators().catch(console.error);