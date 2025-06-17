# Technical Analysis Documentation Website - Implementation Instructions

## Project Overview
You are building a documentation and demo website for a Rust-based technical analysis library with 100+ indicators. The website will showcase each indicator with interactive charts, adjustable parameters, and comprehensive documentation. All computation will eventually use WebAssembly, but for now, we'll use mock implementations.

## Key Requirements

- **Framework**: Astro with Islands architecture for optimal performance
- **Charting**: TradingView Lightweight Charts for all visualizations
- **Styling**: Tailwind CSS with dark mode support
- **Structure**: One page per indicator with interactive demos
- **Data**: Use existing CSV files from parent directory (read-only)
- **WASM**: Stub implementations for now, prepared for future integration

## Project Structure
The website code lives in the `website/` folder within the main TA library project. You'll work from this directory and read files from the parent directory without modifying them.

## Step-by-Step Implementation

### Phase 1: Project Initialization

1. **Initialize Astro Project**

```bash
# From the website/ directory
npm create astro@latest . -- --template minimal --typescript --skip-houston
npm install -D tailwindcss @astrojs/tailwind @astrojs/mdx @astrojs/sitemap
npm install lightweight-charts papaparse
npm install -D @types/papaparse vite-plugin-wasm vite-plugin-top-level-await
```

2. **Configure Astro (astro.config.mjs)**

```javascript
import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
  site: 'https://ta-indicators-demo.com', // Update with actual domain
  integrations: [tailwind(), mdx(), sitemap()],
  vite: {
    plugins: [wasm(), topLevelAwait()]
  }
});
```

3. **Create Directory Structure**

```
website/
├── src/
│   ├── components/
│   │   ├── charts/
│   │   │   ├── IndicatorChart.tsx
│   │   │   ├── ChartControls.tsx
│   │   │   └── BacktestChart.tsx
│   │   ├── layout/
│   │   │   ├── Sidebar.astro
│   │   │   ├── Header.astro
│   │   │   └── Search.tsx
│   │   └── ui/
│   │       ├── ParameterInput.tsx
│   │       └── ThemeToggle.tsx
│   ├── content/
│   │   ├── config.ts
│   │   └── indicators/
│   ├── data/
│   │   ├── indicator-registry.ts
│   │   └── sample-data.ts
│   ├── layouts/
│   │   ├── BaseLayout.astro
│   │   └── IndicatorLayout.astro
│   ├── lib/
│   │   ├── mock-wasm/
│   │   │   ├── index.ts
│   │   │   └── indicators/
│   │   └── utils/
│   │       ├── data-loader.ts
│   │       └── rust-parser.ts
│   ├── pages/
│   │   ├── index.astro
│   │   ├── indicators/
│   │   │   └── [slug].astro
│   │   └── search.astro
│   └── styles/
│       └── global.css
├── scripts/
│   ├── scan-indicators.ts
│   ├── generate-pages.ts
│   └── extract-rust-docs.ts
└── public/
    └── data/
```

### Phase 2: Indicator Discovery and Registry

1. **Create Indicator Scanner (scripts/scan-indicators.ts)**

```typescript
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
}

function extractIndicatorInfo(id: string, content: string, category?: string): IndicatorInfo {
  // Extract from Rust doc comments and function signatures
  // This is a simplified version - enhance based on your Rust code structure
  
  const nameMap: Record<string, string> = {
    'sma': 'Simple Moving Average',
    'ema': 'Exponential Moving Average',
    'rsi': 'Relative Strength Index',
    'macd': 'Moving Average Convergence Divergence',
    'bollinger_bands': 'Bollinger Bands',
    'atr': 'Average True Range',
    'adx': 'Average Directional Index',
    'stoch': 'Stochastic Oscillator',
    // Add all your indicators here
  };
  
  return {
    id,
    name: nameMap[id] || id.replace(/\_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    category: category || categorizeIndicator(id),
    parameters: extractParameters(content),
    outputs: extractOutputs(content),
    description: extractDescription(content)
  };
}

// Run the scanner
scanIndicators().catch(console.error);
```

2. **Create Sample Data Loader (src/lib/utils/data-loader.ts)**

```typescript
import Papa from 'papaparse';
import { readFile } from 'fs/promises';

export interface OHLCV {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export async function loadSampleData(size: 'small' | 'medium' | 'large' = 'small'): Promise<OHLCV[]> {
  const fileMap = {
    small: '../src/data/10kCandles.csv',
    medium: '../src/data/bitfinex btc-usd 100,000 candles ends 09-01-24.csv',
    large: '../src/data/1MillionCandles.csv'
  };
  
  const csvContent = await readFile(fileMap[size], 'utf-8');
  const parsed = Papa.parse(csvContent, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true
  });
  
  return parsed.data.map((row: any) => ({
    time: Math.floor(new Date(row.timestamp || row.date).getTime() / 1000),
    open: parseFloat(row.open),
    high: parseFloat(row.high),
    low: parseFloat(row.low),
    close: parseFloat(row.close),
    volume: parseFloat(row.volume || 0)
  }));
}

// Cache sample data for client-side use
export async function cacheSampleData() {
  const data = await loadSampleData('small');
  // Take last 1000 candles for demo
  const demoData = data.slice(-1000);
  
  // Write to public directory for client access
  const { writeFileSync } = await import('fs');
  writeFileSync('./public/data/sample-ohlcv.json', JSON.stringify(demoData));
}
```

### Phase 3: Mock WASM Implementation

**Create Mock TA Library (src/lib/mock-wasm/index.ts)**

```typescript
import { OHLCV } from '../utils/data-loader';

export class MockTALib {
  // Simple Moving Average
  sma(data: number[], period: number): (number | null)[] {
    const result: (number | null)[] = [];
    for (let i = 0; i < data.length; i++) {
      if (i < period - 1) {
        result.push(null);
      } else {
        const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
        result.push(sum / period);
      }
    }
    return result;
  }

  // Exponential Moving Average
  ema(data: number[], period: number): (number | null)[] {
    const result: (number | null)[] = [];
    const multiplier = 2 / (period + 1);
    
    // Start with SMA for first value
    const sma = this.sma(data.slice(0, period), period);
    result.push(...new Array(period - 1).fill(null));
    result.push(sma[period - 1]);
    
    // Calculate EMA for remaining values
    for (let i = period; i < data.length; i++) {
      const ema = (data[i] - result[i - 1]!) * multiplier + result[i - 1]!;
      result.push(ema);
    }
    return result;
  }

  // Relative Strength Index
  rsi(data: number[], period: number = 14): (number | null)[] {
    const changes = data.slice(1).map((val, i) => val - data[i]);
    const gains = changes.map(c => c > 0 ? c : 0);
    const losses = changes.map(c => c < 0 ? -c : 0);
    
    const avgGain = this.sma(gains, period);
    const avgLoss = this.sma(losses, period);
    
    const result: (number | null)[] = [null]; // First value has no change
    
    for (let i = 0; i < avgGain.length; i++) {
      if (avgGain[i] === null || avgLoss[i] === null) {
        result.push(null);
      } else {
        const rs = avgLoss[i] === 0 ? 100 : avgGain[i]! / avgLoss[i]!;
        result.push(100 - (100 / (1 + rs)));
      }
    }
    return result;
  }

  // Bollinger Bands
  bollingerBands(data: number[], period: number = 20, stdDev: number = 2): {
    upper: (number | null)[];
    middle: (number | null)[];
    lower: (number | null)[];
  } {
    const middle = this.sma(data, period);
    const upper: (number | null)[] = [];
    const lower: (number | null)[] = [];
    
    for (let i = 0; i < data.length; i++) {
      if (i < period - 1) {
        upper.push(null);
        lower.push(null);
      } else {
        const slice = data.slice(i - period + 1, i + 1);
        const mean = middle[i]!;
        const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period;
        const std = Math.sqrt(variance);
        
        upper.push(mean + std * stdDev);
        lower.push(mean - std * stdDev);
      }
    }
    
    return { upper, middle, lower };
  }

  // MACD
  macd(data: number[], fastPeriod: number = 12, slowPeriod: number = 26, signalPeriod: number = 9): {
    macd: (number | null)[];
    signal: (number | null)[];
    histogram: (number | null)[];
  } {
    const fastEMA = this.ema(data, fastPeriod);
    const slowEMA = this.ema(data, slowPeriod);
    
    const macdLine = fastEMA.map((fast, i) => 
      fast !== null && slowEMA[i] !== null ? fast - slowEMA[i]! : null
    );
    
    const signalLine = this.ema(macdLine.filter(v => v !== null) as number[], signalPeriod);
    
    // Align signal line with MACD line
    let signalIndex = 0;
    const alignedSignal = macdLine.map(m => {
      if (m === null) return null;
      return signalLine[signalIndex++] || null;
    });
    
    const histogram = macdLine.map((m, i) => 
      m !== null && alignedSignal[i] !== null ? m - alignedSignal[i]! : null
    );
    
    return { macd: macdLine, signal: alignedSignal, histogram };
  }

  // Stochastic Oscillator
  stochastic(high: number[], low: number[], close: number[], kPeriod: number = 14, dPeriod: number = 3): {
    k: (number | null)[];
    d: (number | null)[];
  } {
    const k: (number | null)[] = [];
    
    for (let i = 0; i < close.length; i++) {
      if (i < kPeriod - 1) {
        k.push(null);
      } else {
        const highMax = Math.max(...high.slice(i - kPeriod + 1, i + 1));
        const lowMin = Math.min(...low.slice(i - kPeriod + 1, i + 1));
        const kValue = ((close[i] - lowMin) / (highMax - lowMin)) * 100;
        k.push(kValue);
      }
    }
    
    const d = this.sma(k.filter(v => v !== null) as number[], dPeriod);
    
    // Align D with K
    let dIndex = 0;
    const alignedD = k.map(kVal => {
      if (kVal === null) return null;
      return d[dIndex++] || null;
    });
    
    return { k, d: alignedD };
  }

  // Add more indicators following similar patterns...
  // ATR, ADX, CCI, Williams %R, etc.

  // Generic indicator calculator
  calculate(indicatorId: string, data: OHLCV[], params: Record<string, any>): any {
    const closes = data.map(d => d.close);
    
    switch(indicatorId) {
      case 'sma':
        return this.sma(closes, params.period || 20);
      case 'ema':
        return this.ema(closes, params.period || 20);
      case 'rsi':
        return this.rsi(closes, params.period || 14);
      case 'bollinger_bands':
        return this.bollingerBands(closes, params.period || 20, params.stdDev || 2);
      case 'macd':
        return this.macd(closes, params.fast || 12, params.slow || 26, params.signal || 9);
      case 'stoch':
        return this.stochastic(
          data.map(d => d.high),
          data.map(d => d.low),
          closes,
          params.kPeriod || 14,
          params.dPeriod || 3
        );
      // Add all other indicators...
      default:
        // Return mock data for unimplemented indicators
        return this.generateMockIndicatorData(closes.length, indicatorId);
    }
  }

  private generateMockIndicatorData(length: number, seed: string): (number | null)[] {
    // Generate deterministic mock data based on indicator name
    const result: (number | null)[] = [];
    let value = 50;
    const hash = seed.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    
    for (let i = 0; i < length; i++) {
      if (i < 20) {
        result.push(null); // Most indicators have some initial null values
      } else {
        // Random walk around midpoint
        value += (Math.sin(i / 10 + hash) * 5) + (Math.random() - 0.5) * 2;
        value = Math.max(0, Math.min(100, value)); // Bound between 0-100
        result.push(value);
      }
    }
    return result;
  }
}

// Export singleton instance
export const talib = new MockTALib();
```

### Phase 4: Chart Components

**Create Main Chart Component (src/components/charts/IndicatorChart.tsx)**

```typescript
import { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, ColorType } from 'lightweight-charts';
import { talib } from '../../lib/mock-wasm';
import type { OHLCV } from '../../lib/utils/data-loader';

interface IndicatorChartProps {
  indicatorId: string;
  data: OHLCV[];
  parameters: Record<string, any>;
  darkMode?: boolean;
}

export function IndicatorChart({ indicatorId, data, parameters, darkMode = false }: IndicatorChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 500,
      layout: {
        background: { type: ColorType.Solid, color: darkMode ? '#1a1a1a' : '#ffffff' },
        textColor: darkMode ? '#d1d5db' : '#333',
      },
      grid: {
        vertLines: { color: darkMode ? '#2a2a2a' : '#e0e0e0' },
        horzLines: { color: darkMode ? '#2a2a2a' : '#e0e0e0' },
      },
      crosshair: {
        mode: 0,
      },
      rightPriceScale: {
        borderColor: darkMode ? '#2a2a2a' : '#e0e0e0',
      },
      timeScale: {
        borderColor: darkMode ? '#2a2a2a' : '#e0e0e0',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

    // Add candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    });

    candlestickSeries.setData(data);

    // Calculate and add indicator
    try {
      const indicatorData = talib.calculate(indicatorId, data, parameters);
      
      if (Array.isArray(indicatorData)) {
        // Single line indicator
        addLineSeries(chart, indicatorData, data, 'Indicator', '#2962ff');
      } else if (typeof indicatorData === 'object') {
        // Multi-line indicator (like Bollinger Bands, MACD)
        const colors = ['#2962ff', '#ff6b6b', '#4ecdc4', '#ffe66d'];
        let colorIndex = 0;
        
        Object.entries(indicatorData).forEach(([key, values]) => {
          if (Array.isArray(values)) {
            addLineSeries(
              chart, 
              values as (number | null)[], 
              data, 
              key.charAt(0).toUpperCase() + key.slice(1),
              colors[colorIndex++ % colors.length]
            );
          }
        });
      }
      
      setIsLoading(false);
    } catch (error) {
      console.error('Error calculating indicator:', error);
      setIsLoading(false);
    }

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [indicatorId, data, parameters, darkMode]);

  function addLineSeries(
    chart: IChartApi,
    values: (number | null)[],
    ohlcv: OHLCV[],
    title: string,
    color: string
  ) {
    const lineSeries = chart.addLineSeries({
      color,
      lineWidth: 2,
      title,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 5,
    });

    const lineData = values
      .map((value, index) => ({
        time: ohlcv[index]?.time,
        value,
      }))
      .filter(item => item.value !== null && item.time !== undefined);

    lineSeries.setData(lineData as any);
  }

  return (
    <div className="relative">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/80 dark:bg-gray-900/80 z-10">
          <div className="text-gray-600 dark:text-gray-400">Loading chart...</div>
        </div>
      )}
      <div ref={chartContainerRef} className="w-full" />
    </div>
  );
}
```

**Create Parameter Controls (src/components/charts/ChartControls.tsx)**

```typescript
import { useState } from 'react';
import type { indicators } from '../../data/indicator-registry';

interface ChartControlsProps {
  indicatorId: keyof typeof indicators;
  parameters: Record<string, any>;
  onParameterChange: (params: Record<string, any>) => void;
}

export function ChartControls({ indicatorId, parameters, onParameterChange }: ChartControlsProps) {
  const [localParams, setLocalParams] = useState(parameters);
  
  const indicator = indicators[indicatorId];

  const handleChange = (paramName: string, value: any) => {
    const newParams = { ...localParams, [paramName]: value };
    setLocalParams(newParams);
    onParameterChange(newParams);
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 space-y-4">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
        Parameters
      </h3>
      
      {indicator.parameters.map((param) => (
        <div key={param.name} className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            {param.name.charAt(0).toUpperCase() + param.name.slice(1).replace(/([A-Z])/g, ' $1')}
          </label>
          
          {param.type === 'number' && (
            <div className="flex items-center space-x-4">
              <input
                type="range"
                min={param.min || 1}
                max={param.max || 100}
                value={localParams[param.name] || param.default}
                onChange={(e) => handleChange(param.name, Number(e.target.value))}
                className="flex-1"
              />
              <input
                type="number"
                min={param.min || 1}
                max={param.max || 100}
                value={localParams[param.name] || param.default}
                onChange={(e) => handleChange(param.name, Number(e.target.value))}
                className="w-20 px-2 py-1 border rounded dark:bg-gray-700 dark:border-gray-600"
              />
            </div>
          )}
          
          {param.type === 'boolean' && (
            <input
              type="checkbox"
              checked={localParams[param.name] || param.default}
              onChange={(e) => handleChange(param.name, e.target.checked)}
              className="rounded"
            />
          )}
        </div>
      ))}
      
      <button
        onClick={() => {
          const defaults = indicator.parameters.reduce((acc, param) => ({
            ...acc,
            [param.name]: param.default
          }), {});
          setLocalParams(defaults);
          onParameterChange(defaults);
        }}
        className="w-full px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
      >
        Reset to Defaults
      </button>
    </div>
  );
}
```

### Phase 5: Page Generation

**Create Page Generator (scripts/generate-pages.ts)**

```typescript
import { indicators } from '../src/data/indicator-registry';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';

function generateIndicatorPages() {
  const contentDir = './src/content/indicators';
  
  // Ensure directory exists
  if (!existsSync(contentDir)) {
    mkdirSync(contentDir, { recursive: true });
  }

  Object.entries(indicators).forEach(([id, indicator]) => {
    const content = `---
title: "${indicator.name}"
description: "${indicator.description || `Technical analysis indicator: ${indicator.name}`}"
category: "${indicator.category}"
parameters: ${JSON.stringify(indicator.parameters)}
---

# {frontmatter.title}

${indicator.description || `The ${indicator.name} is a technical analysis indicator used in trading.`}

## How it works

${generateExplanation(id, indicator)}

## Parameters

${indicator.parameters.map(param => `- **${param.name}**: ${param.description || 'Parameter for calculation'} (default: ${param.default})`).join('\n')}

## Interpretation

${generateInterpretation(id, indicator)}

## Example Usage

\`\`\`rust
// Example of using ${indicator.name} in Rust
let result = indicators::${id}(&data, ${indicator.parameters.map(p => p.default).join(', ')});
\`\`\`

## Interactive Demo

<IndicatorDemo indicatorId="${id}" />
`;

    writeFileSync(join(contentDir, `${id}.mdx`), content);
  });

  console.log(`Generated ${Object.keys(indicators).length} indicator pages`);
}

function generateExplanation(id: string, indicator: any): string {
  // Add specific explanations for common indicators
  const explanations: Record<string, string> = {
    'sma': 'The Simple Moving Average calculates the arithmetic mean of prices over a specified period. It smooths out price action by creating a single flowing line that represents the average price over time.',
    'ema': 'The Exponential Moving Average gives more weight to recent prices, making it more responsive to new information compared to the SMA. It uses a smoothing factor to exponentially decrease the weights of older observations.',
    'rsi': 'The Relative Strength Index measures momentum by comparing the magnitude of recent gains to recent losses. It oscillates between 0 and 100, with readings above 70 indicating overbought conditions and below 30 indicating oversold conditions.',
    'macd': 'The Moving Average Convergence Divergence shows the relationship between two moving averages of prices. It consists of the MACD line (12-day EMA - 26-day EMA), signal line (9-day EMA of MACD), and histogram (MACD - Signal).',
    // Add more explanations...
  };
  
  return explanations[id] || `The ${indicator.name} helps traders identify potential trading opportunities by analyzing price patterns and market conditions.`;
}

function generateInterpretation(id: string, indicator: any): string {
  const interpretations: Record<string, string> = {
    'sma': `- Price above SMA: Potential uptrend\n- Price below SMA: Potential downtrend\n- Price crossing SMA: Possible trend change`,
    'rsi': `- RSI > 70: Potentially overbought\n- RSI < 30: Potentially oversold\n- Divergences with price: Potential reversals`,
    // Add more interpretations...
  };
  
  return interpretations[id] || `Traders use the ${indicator.name} to make informed decisions about entry and exit points.`;
}

// Run the generator
generateIndicatorPages();
```

### Phase 6: Layout Components

**Create Base Layout (src/layouts/BaseLayout.astro)**

```astro
---
import Header from '../components/layout/Header.astro';
import Sidebar from '../components/layout/Sidebar.astro';
import '../styles/global.css';

export interface Props {
  title: string;
  description?: string;
}

const { title, description = 'Technical Analysis Indicators Documentation' } = Astro.props;
---

<!DOCTYPE html>
<html lang="en" class="dark">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{title} | TA Indicators</title>
    <meta name="description" content={description} />
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
  </head>
  <body class="bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100">
    <Header />
    <div class="flex">
      <Sidebar />
      <main class="flex-1 px-6 py-8 ml-64">
        <slot />
      </main>
    </div>
    
    <script>
      // Theme toggle logic
      const theme = localStorage.getItem('theme') || 'dark';
      document.documentElement.classList.toggle('dark', theme === 'dark');
    </script>
  </body>
</html>
```

**Create Sidebar Navigation (src/components/layout/Sidebar.astro)**

```astro
---
import { indicators } from '../../data/indicator-registry';

// Group indicators by category
const categories = Object.entries(indicators).reduce((acc, [id, indicator]) => {
  const category = indicator.category;
  if (!acc[category]) acc[category] = [];
  acc[category].push({ id, ...indicator });
  return acc;
}, {} as Record<string, any[]>);

const currentPath = Astro.url.pathname;
---

<aside class="fixed left-0 top-16 h-[calc(100vh-4rem)] w-64 bg-gray-50 dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 overflow-y-auto">
  <nav class="p-4 space-y-4">
    <a href="/" class="block px-4 py-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors">
      Home
    </a>
    
    {Object.entries(categories).map(([category, items]) => (
      <div class="space-y-2">
        <h3 class="px-4 text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
          {category.replace(/_/g, ' ')}
        </h3>
        <div class="space-y-1">
          {items.map(indicator => (
            <a 
              href={`/indicators/${indicator.id}`}
              class={`block px-4 py-2 text-sm rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors ${
                currentPath === `/indicators/${indicator.id}` 
                  ? 'bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-300' 
                  : ''
              }`}
            >
              {indicator.name}
            </a>
          ))}
        </div>
      </div>
    ))}
  </nav>
</aside>
```

### Phase 7: Dynamic Indicator Pages

**Create Dynamic Route (src/pages/indicators/[slug].astro)**

```astro
---
import BaseLayout from '../../layouts/BaseLayout.astro';
import { IndicatorChart } from '../../components/charts/IndicatorChart';
import { ChartControls } from '../../components/charts/ChartControls';
import { indicators } from '../../data/indicator-registry';
import { getCollection } from 'astro:content';

export async function getStaticPaths() {
  const indicatorEntries = await getCollection('indicators');
  return indicatorEntries.map(entry => ({
    params: { slug: entry.slug },
    props: { entry },
  }));
}

const { entry } = Astro.props;
const { Content } = await entry.render();
const indicator = indicators[entry.slug];

// Load sample data
const response = await fetch('/data/sample-ohlcv.json');
const sampleData = await response.json();
---

<BaseLayout title={indicator.name} description={indicator.description}>
  <div class="max-w-7xl mx-auto">
    <h1 class="text-4xl font-bold mb-4">{indicator.name}</h1>
    
    <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
      <div class="lg:col-span-3 space-y-6">
        <!-- Chart -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <IndicatorChart
            client:load
            indicatorId={entry.slug}
            data={sampleData}
            parameters={indicator.parameters.reduce((acc, p) => ({ ...acc, [p.name]: p.default }), {})}
          />
        </div>
        
        <!-- Documentation -->
        <div class="prose dark:prose-invert max-w-none">
          <Content />
        </div>
      </div>
      
      <!-- Controls -->
      <div class="lg:col-span-1">
        <div class="sticky top-20">
          <ChartControls
            client:load
            indicatorId={entry.slug}
            parameters={indicator.parameters.reduce((acc, p) => ({ ...acc, [p.name]: p.default }), {})}
            onParameterChange={(params) => {
              // Update chart with new parameters
              // This will be handled by React state in the parent component
            }}
          />
        </div>
      </div>
    </div>
  </div>
</BaseLayout>
```

### Phase 8: Homepage with Backtester

**Create Homepage (src/pages/index.astro)**

```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';
import { BacktestChart } from '../components/charts/BacktestChart';
---

<BaseLayout title="Home">
  <div class="max-w-7xl mx-auto">
    <section class="text-center py-12">
      <h1 class="text-5xl font-bold mb-4">
        Technical Analysis Indicators
      </h1>
      <p class="text-xl text-gray-600 dark:text-gray-400 mb-8">
        Explore 300+ technical indicators with real-time calculations powered by WebAssembly
      </p>
    </section>

    <section class="mb-12">
      <h2 class="text-3xl font-bold mb-6">Moving Average Crossover Backtester</h2>
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <BacktestChart client:load />
      </div>
    </section>

    <section>
      <h2 class="text-3xl font-bold mb-6">Featured Indicators</h2>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {['sma', 'rsi', 'macd', 'bollinger_bands', 'stoch', 'atr'].map(id => (
          <a 
            href={`/indicators/${id}`}
            class="block p-6 bg-white dark:bg-gray-800 rounded-lg shadow hover:shadow-lg transition-shadow"
          >
            <h3 class="text-xl font-semibold mb-2">{indicators[id].name}</h3>
            <p class="text-gray-600 dark:text-gray-400">
              {indicators[id].description || 'Click to explore this indicator'}
            </p>
          </a>
        ))}
      </div>
    </section>
  </div>
</BaseLayout>
```

### Phase 9: Search Implementation

**Create Search Component (src/components/layout/Search.tsx)**

```typescript
import { useState, useEffect, useRef } from 'react';
import { indicators } from '../../data/indicator-registry';

export function Search() {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsOpen(true);
      }
      if (e.key === 'Escape') {
        setIsOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    if (query) {
      const filtered = Object.entries(indicators)
        .filter(([id, indicator]) => 
          indicator.name.toLowerCase().includes(query.toLowerCase()) ||
          id.includes(query.toLowerCase()) ||
          indicator.category.includes(query.toLowerCase())
        )
        .slice(0, 10)
        .map(([id, indicator]) => ({ id, ...indicator }));
      
      setResults(filtered);
    } else {
      setResults([]);
    }
  }, [query]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/50" onClick={() => setIsOpen(false)}>
      <div 
        className="mx-auto mt-20 max-w-2xl bg-white dark:bg-gray-800 rounded-lg shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center p-4 border-b dark:border-gray-700">
          <svg className="w-5 h-5 text-gray-400 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search indicators..."
            className="flex-1 bg-transparent outline-none"
          />
          <kbd className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 rounded">ESC</kbd>
        </div>

        {results.length > 0 && (
          <div className="max-h-96 overflow-y-auto">
            {results.map(result => (
              <a
                key={result.id}
                href={`/indicators/${result.id}`}
                className="block px-4 py-3 hover:bg-gray-100 dark:hover:bg-gray-700 border-b dark:border-gray-700"
              >
                <div className="font-medium">{result.name}</div>
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  {result.category.replace(/_/g, ' ')}
                </div>
              </a>
            ))}
          </div>
        )}

        {query && results.length === 0 && (
          <div className="p-8 text-center text-gray-500">
            No indicators found for "{query}"
          </div>
        )}
      </div>
    </div>
  );
}
```

### Phase 10: Deployment Configuration

**Add Build Scripts (package.json)**

```json
{
  "scripts": {
    "dev": "npm run prebuild && astro dev",
    "build": "npm run prebuild && astro build",
    "preview": "astro preview",
    "prebuild": "npm run scan-indicators && npm run generate-pages && npm run cache-data",
    "scan-indicators": "tsx scripts/scan-indicators.ts",
    "generate-pages": "tsx scripts/generate-pages.ts",
    "cache-data": "tsx scripts/cache-sample-data.ts"
  }
}
```

**Create Cache Script (scripts/cache-sample-data.ts)**

```typescript
import { cacheSampleData } from '../src/lib/utils/data-loader';

cacheSampleData()
  .then(() => console.log('Sample data cached successfully'))
  .catch(console.error);
```

## Implementation Checklist

- [ ] Initialize Astro project with TypeScript
- [ ] Install all dependencies
- [ ] Create directory structure
- [ ] Set up Tailwind CSS with dark mode
- [ ] Create indicator scanner script
- [ ] Generate indicator registry from Rust files
- [ ] Implement mock WASM library with basic indicators
- [ ] Create chart components with TradingView Lightweight Charts
- [ ] Build parameter control components
- [ ] Generate MDX pages for all indicators
- [ ] Create layouts and navigation
- [ ] Implement search functionality
- [ ] Build homepage with backtester demo
- [ ] Add caching for sample data
- [ ] Test all indicators with mock data
- [ ] Configure build pipeline
- [ ] Deploy to static hosting

## Important Notes

- The mock WASM implementation should closely mirror the eventual real WASM API
- All file reads from parent directory should be read-only operations
- Chart components should gracefully handle missing or invalid data
- Parameter controls should validate inputs based on indicator specifications
- Search should be fast and work client-side
- Dark mode should be persistent across sessions
- Mobile responsiveness is important for all components

## Future WASM Integration

When ready to integrate real WASM:

1. Replace MockTALib with actual WASM module imports
2. Update the calculate method to call real Rust functions
3. Ensure WASM module is loaded asynchronously
4. Add proper error handling for WASM initialization
5. Update build process to include wasm-pack step

Start with Phase 1 and work through each phase sequentially. Test thoroughly after each phase before proceeding to the next.