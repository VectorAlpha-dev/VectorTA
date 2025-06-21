import { readFile, writeFile, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import { join, dirname } from 'path';

interface IndicatorInfo {
  id: string;
  name: string;
  category: string;
  description: string;
  parameters: Array<{
    name: string;
    type: string;
    default: any;
    min?: number;
    max?: number;
    description: string;
  }>;
}

// Common indicator definitions
const indicators: IndicatorInfo[] = [
  // Moving Averages
  {
    id: 'wma',
    name: 'Weighted Moving Average',
    category: 'moving_averages',
    description: 'A moving average that assigns greater weight to more recent data points.',
    parameters: [
      { name: 'period', type: 'number', default: 20, min: 2, max: 500, description: 'The number of periods' }
    ]
  },
  {
    id: 'dema',
    name: 'Double Exponential Moving Average',
    category: 'moving_averages',
    description: 'A technical indicator that uses two exponential moving averages to eliminate lag.',
    parameters: [
      { name: 'period', type: 'number', default: 20, min: 2, max: 500, description: 'The number of periods' }
    ]
  },
  {
    id: 'tema',
    name: 'Triple Exponential Moving Average',
    category: 'moving_averages',
    description: 'A technical indicator that uses three exponential moving averages to reduce lag.',
    parameters: [
      { name: 'period', type: 'number', default: 20, min: 2, max: 500, description: 'The number of periods' }
    ]
  },
  
  // Momentum Indicators
  {
    id: 'macd',
    name: 'Moving Average Convergence Divergence',
    category: 'momentum',
    description: 'A trend-following momentum indicator that shows the relationship between two moving averages.',
    parameters: [
      { name: 'fast_period', type: 'number', default: 12, min: 2, max: 100, description: 'Fast EMA period' },
      { name: 'slow_period', type: 'number', default: 26, min: 2, max: 200, description: 'Slow EMA period' },
      { name: 'signal_period', type: 'number', default: 9, min: 2, max: 50, description: 'Signal line EMA period' }
    ]
  },
  {
    id: 'stochastic',
    name: 'Stochastic Oscillator',
    category: 'momentum',
    description: 'A momentum indicator comparing a particular closing price to a range of prices over time.',
    parameters: [
      { name: 'k_period', type: 'number', default: 14, min: 2, max: 100, description: '%K period' },
      { name: 'd_period', type: 'number', default: 3, min: 1, max: 50, description: '%D period (SMA of %K)' }
    ]
  },
  {
    id: 'roc',
    name: 'Rate of Change',
    category: 'momentum',
    description: 'Measures the percentage change in price from one period to another.',
    parameters: [
      { name: 'period', type: 'number', default: 10, min: 1, max: 200, description: 'Lookback period' }
    ]
  },
  
  // Volatility Indicators
  {
    id: 'atr',
    name: 'Average True Range',
    category: 'volatility',
    description: 'Measures market volatility by analyzing the range of price movement.',
    parameters: [
      { name: 'period', type: 'number', default: 14, min: 1, max: 100, description: 'The number of periods' }
    ]
  },
  {
    id: 'standard_deviation',
    name: 'Standard Deviation',
    category: 'volatility',
    description: 'Measures the dispersion of a dataset relative to its mean.',
    parameters: [
      { name: 'period', type: 'number', default: 20, min: 2, max: 500, description: 'The number of periods' }
    ]
  },
  
  // Volume Indicators
  {
    id: 'obv',
    name: 'On Balance Volume',
    category: 'volume',
    description: 'Uses volume flow to predict changes in stock price.',
    parameters: []
  },
  {
    id: 'vwap',
    name: 'Volume Weighted Average Price',
    category: 'volume',
    description: 'The average price weighted by volume, often used as a trading benchmark.',
    parameters: []
  },
  
  // Trend Indicators
  {
    id: 'adx',
    name: 'Average Directional Index',
    category: 'trend',
    description: 'Measures the strength of a trend, regardless of its direction.',
    parameters: [
      { name: 'period', type: 'number', default: 14, min: 2, max: 100, description: 'The number of periods' }
    ]
  },
  {
    id: 'ichimoku',
    name: 'Ichimoku Cloud',
    category: 'trend',
    description: 'A comprehensive indicator that defines support/resistance, trend direction, and momentum.',
    parameters: [
      { name: 'conversion_period', type: 'number', default: 9, min: 1, max: 100, description: 'Tenkan-sen period' },
      { name: 'base_period', type: 'number', default: 26, min: 1, max: 200, description: 'Kijun-sen period' },
      { name: 'span_period', type: 'number', default: 52, min: 1, max: 500, description: 'Senkou Span B period' }
    ]
  },
  
  // Oscillators
  {
    id: 'cci',
    name: 'Commodity Channel Index',
    category: 'oscillators',
    description: 'Measures the variation of a price from its statistical mean.',
    parameters: [
      { name: 'period', type: 'number', default: 20, min: 2, max: 100, description: 'The number of periods' }
    ]
  },
  {
    id: 'williams_r',
    name: 'Williams %R',
    category: 'oscillators',
    description: 'A momentum indicator that measures overbought and oversold levels.',
    parameters: [
      { name: 'period', type: 'number', default: 14, min: 2, max: 100, description: 'The lookback period' }
    ]
  }
];

// Template for indicator documentation
function generateIndicatorMDX(indicator: IndicatorInfo): string {
  const parameterDocs = indicator.parameters.length > 0 
    ? indicator.parameters.map(p => `### ${p.name} (default: ${p.default})
${p.description}

- Type: \`${p.type}\`
${p.min !== undefined ? `- Min: ${p.min}` : ''}
${p.max !== undefined ? `- Max: ${p.max}` : ''}
`).join('\n')
    : 'This indicator has no configurable parameters.';

  return `---
title: "${indicator.name}"
description: "${indicator.description}"
category: "${indicator.category}"
parameters:
${indicator.parameters.length > 0 ? indicator.parameters.map(p => `  - name: "${p.name}"
    type: "${p.type}"
    default: ${p.default}
${p.min !== undefined ? `    min: ${p.min}` : ''}
${p.max !== undefined ? `    max: ${p.max}` : ''}
    description: "${p.description}"`).join('\n') : '  []'}
returns:
  type: "Vec<Option<f64>>"
  description: "A vector of calculated ${indicator.name} values"
complexity: "O(n)"
implementationStatus: "planned"
---

import { IndicatorLayout } from '../../../layouts/IndicatorLayout.astro';
import { CodeBlock } from '../../../components/ui/CodeBlock.astro';

<IndicatorLayout 
  indicatorId="${indicator.id}" 
  indicatorName={frontmatter.title}
  description={frontmatter.description}
  parameters={frontmatter.parameters}
>

## Overview {#overview}

${indicator.description}

*Note: This is placeholder documentation. Full documentation will be added when the indicator is implemented.*

## Interpretation {#interpretation}

Details about how to interpret ${indicator.name} values will be added here.

## Calculation {#calculation}

The mathematical formula and calculation steps for ${indicator.name} will be documented here.

## Parameters {#parameters}

${parameterDocs}

## Returns & Output {#returns}

The ${indicator.name} function returns a vector of optional floating-point values representing the indicator calculations.

## Example Usage {#usage}

<CodeBlock lang="rust">
use vectorta::indicators::${indicator.id};

// Example usage will be added when implemented
let result = ${indicator.id}(&data${indicator.parameters.map(p => `, ${p.default}`).join('')})?;
</CodeBlock>

## Common Use Cases {#use-cases}

Common trading strategies and applications using ${indicator.name} will be documented here.

## Edge Cases & Errors {#edge-cases}

Information about edge cases and error handling will be added here.

## References {#references}

Academic and industry references for ${indicator.name} will be listed here.

</IndicatorLayout>`;
}

// Main function to generate all indicator docs
async function generateAllIndicatorDocs() {
  console.log('Starting indicator documentation generation...');
  
  for (const indicator of indicators) {
    const categoryDir = join('src/content/indicators', indicator.category);
    const filePath = join(categoryDir, `${indicator.id}.mdx`);
    
    // Check if file already exists
    if (existsSync(filePath)) {
      console.log(`Skipping ${indicator.name} - file already exists`);
      continue;
    }
    
    // Ensure directory exists
    if (!existsSync(categoryDir)) {
      await mkdir(categoryDir, { recursive: true });
    }
    
    // Generate and write MDX content
    const content = generateIndicatorMDX(indicator);
    await writeFile(filePath, content);
    
    console.log(`Generated documentation for ${indicator.name}`);
  }
  
  console.log(`\nGenerated documentation for ${indicators.length} indicators`);
  console.log('\nNext steps:');
  console.log('1. Review generated documentation files');
  console.log('2. Update with actual implementation details as indicators are built');
  console.log('3. Add real-world examples and trading strategies');
}

// Run the generator
generateAllIndicatorDocs().catch(console.error);