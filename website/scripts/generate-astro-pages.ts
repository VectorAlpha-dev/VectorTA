import { indicators } from '../src/data/indicator-registry.js';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';

function generateIndicatorPages() {
  const pagesDir = './src/pages/indicators';
  
  // Ensure directory exists
  if (!existsSync(pagesDir)) {
    mkdirSync(pagesDir, { recursive: true });
  }

  Object.entries(indicators).forEach(([id, indicator]) => {
    const content = `---
import IndicatorLayout from '../../layouts/IndicatorLayout.astro';

const indicatorId = '${id}';
const indicatorName = '${indicator.name}';
const description = \`${(indicator.description || `Technical analysis indicator: ${indicator.name}`).replace(/`/g, "'")}\`;
const parameters = ${JSON.stringify(indicator.parameters, null, 2)};
---

<IndicatorLayout
  indicatorId={indicatorId}
  indicatorName={indicatorName}
  description={description}
  parameters={parameters}
>
  <h2>Overview</h2>
  <p>{description}</p>

  <h2>How it works</h2>
  <p>${generateExplanation(id, indicator)}</p>

  <h2>Parameters</h2>
  <ul>
    ${indicator.parameters.map(param => `<li><strong>${param.name}</strong>: ${param.description || 'Parameter for calculation'} (default: ${param.default})</li>`).join('\n    ')}
  </ul>

  <h2>Interpretation</h2>
  <p>${generateInterpretation(id, indicator).replace(/\n/g, '<br />')}</p>

  <h2>Example Usage</h2>
  <pre><code class="language-rust">
// Example of using ${indicator.name} in Rust
let result = indicators::${id}(&data, ${indicator.parameters.map(p => p.default).join(', ')});
  </code></pre>
</IndicatorLayout>`;

    // Write the .astro file
    writeFileSync(join(pagesDir, `${id}.astro`), content);
  });

  console.log(`Generated ${Object.keys(indicators).length} indicator pages`);
}

function generateExplanation(id: string, indicator: any): string {
  const explanations: Record<string, string> = {
    'sma': 'The Simple Moving Average calculates the arithmetic mean of prices over a specified period. It smooths out price action by creating a single flowing line that represents the average price over time.',
    'ema': 'The Exponential Moving Average gives more weight to recent prices, making it more responsive to new information compared to the SMA. It uses a smoothing factor to exponentially decrease the weights of older observations.',
    'rsi': 'The Relative Strength Index measures momentum by comparing the magnitude of recent gains to recent losses. It oscillates between 0 and 100, with readings above 70 indicating overbought conditions and below 30 indicating oversold conditions.',
    'macd': 'The Moving Average Convergence Divergence shows the relationship between two moving averages of prices. It consists of the MACD line (12-day EMA - 26-day EMA), signal line (9-day EMA of MACD), and histogram (MACD - Signal).',
    'bollinger_bands': 'Bollinger Bands consist of a middle band (SMA) and two outer bands that are standard deviations away from the middle. They expand and contract based on market volatility.',
    'stoch': 'The Stochastic Oscillator compares a security\'s closing price to its price range over a given time period. It generates values between 0 and 100 to identify overbought and oversold conditions.',
    'atr': 'The Average True Range measures market volatility by decomposing the entire range of an asset price for that period. Higher ATR values indicate higher volatility.',
    'cci': 'The Commodity Channel Index identifies cyclical trends in a security. It measures the variation of a security\'s price from its statistical mean.',
    'obv': 'On Balance Volume uses volume flow to predict changes in stock price. It adds volume on up days and subtracts volume on down days.',
    'willr': 'Williams %R is a momentum indicator that measures overbought and oversold levels. It ranges from -100 to 0, with readings near 0 indicating overbought conditions.',
    'sar': 'The Parabolic SAR (Stop and Reverse) is used to determine trend direction and potential reversals in price. It appears as dots above or below the price.',
    'mfi': 'The Money Flow Index is a momentum indicator that uses price and volume data to identify overbought or oversold signals. It\'s similar to RSI but incorporates volume.',
  };
  
  return explanations[id] || `The ${indicator.name} helps traders identify potential trading opportunities by analyzing price patterns and market conditions.`;
}

function generateInterpretation(id: string, indicator: any): string {
  const interpretations: Record<string, string> = {
    'sma': `- Price above SMA: Potential uptrend\n- Price below SMA: Potential downtrend\n- Price crossing SMA: Possible trend change`,
    'rsi': `- RSI > 70: Potentially overbought\n- RSI < 30: Potentially oversold\n- Divergences with price: Potential reversals`,
    'macd': `- MACD line above signal line: Bullish signal\n- MACD line below signal line: Bearish signal\n- Histogram increasing: Strengthening trend\n- Histogram decreasing: Weakening trend`,
    'bollinger_bands': `- Price near upper band: Potentially overbought\n- Price near lower band: Potentially oversold\n- Band squeeze: Low volatility, potential breakout coming\n- Band expansion: High volatility`,
    'stoch': `- %K > 80: Overbought territory\n- %K < 20: Oversold territory\n- %K crossing above %D: Potential buy signal\n- %K crossing below %D: Potential sell signal`,
  };
  
  return interpretations[id] || `Traders use the ${indicator.name} to make informed decisions about entry and exit points.`;
}

// Run the generator
generateIndicatorPages();