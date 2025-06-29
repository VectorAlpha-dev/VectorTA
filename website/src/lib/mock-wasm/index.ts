
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

  // Average True Range
  atr(high: number[], low: number[], close: number[], period: number = 14): (number | null)[] {
    const tr: number[] = [];
    
    for (let i = 0; i < high.length; i++) {
      if (i === 0) {
        tr.push(high[i] - low[i]);
      } else {
        const hl = high[i] - low[i];
        const hc = Math.abs(high[i] - close[i - 1]);
        const lc = Math.abs(low[i] - close[i - 1]);
        tr.push(Math.max(hl, hc, lc));
      }
    }
    
    return this.sma(tr, period);
  }

  // Williams %R
  willr(high: number[], low: number[], close: number[], period: number = 14): (number | null)[] {
    const result: (number | null)[] = [];
    
    for (let i = 0; i < close.length; i++) {
      if (i < period - 1) {
        result.push(null);
      } else {
        const highMax = Math.max(...high.slice(i - period + 1, i + 1));
        const lowMin = Math.min(...low.slice(i - period + 1, i + 1));
        const willrValue = ((highMax - close[i]) / (highMax - lowMin)) * -100;
        result.push(willrValue);
      }
    }
    
    return result;
  }

  // Commodity Channel Index
  cci(high: number[], low: number[], close: number[], period: number = 20): (number | null)[] {
    const result: (number | null)[] = [];
    const typicalPrice = high.map((h, i) => (h + low[i] + close[i]) / 3);
    const smaTP = this.sma(typicalPrice, period);
    
    for (let i = 0; i < typicalPrice.length; i++) {
      if (i < period - 1 || smaTP[i] === null) {
        result.push(null);
      } else {
        const slice = typicalPrice.slice(i - period + 1, i + 1);
        const meanDev = slice.reduce((sum, tp) => sum + Math.abs(tp - smaTP[i]!), 0) / period;
        const cci = (typicalPrice[i] - smaTP[i]!) / (0.015 * meanDev);
        result.push(cci);
      }
    }
    
    return result;
  }

  // On Balance Volume
  obv(close: number[], volume: number[]): number[] {
    const result: number[] = [volume[0]];
    
    for (let i = 1; i < close.length; i++) {
      if (close[i] > close[i - 1]) {
        result.push(result[i - 1] + volume[i]);
      } else if (close[i] < close[i - 1]) {
        result.push(result[i - 1] - volume[i]);
      } else {
        result.push(result[i - 1]);
      }
    }
    
    return result;
  }

  // Money Flow Index
  mfi(high: number[], low: number[], close: number[], volume: number[], period: number = 14): (number | null)[] {
    const result: (number | null)[] = [];
    const typicalPrice = high.map((h, i) => (h + low[i] + close[i]) / 3);
    const rawMoneyFlow = typicalPrice.map((tp, i) => tp * volume[i]);
    
    for (let i = 0; i < typicalPrice.length; i++) {
      if (i < period) {
        result.push(null);
      } else {
        let positiveFlow = 0;
        let negativeFlow = 0;
        
        for (let j = i - period + 1; j <= i; j++) {
          if (typicalPrice[j] > typicalPrice[j - 1]) {
            positiveFlow += rawMoneyFlow[j];
          } else if (typicalPrice[j] < typicalPrice[j - 1]) {
            negativeFlow += rawMoneyFlow[j];
          }
        }
        
        const moneyRatio = positiveFlow / negativeFlow;
        const mfi = 100 - (100 / (1 + moneyRatio));
        result.push(mfi);
      }
    }
    
    return result;
  }

  // Parabolic SAR
  sar(high: number[], low: number[], af: number = 0.02, maxAf: number = 0.2): (number | null)[] {
    const result: (number | null)[] = [];
    let isLong = true;
    let ep = high[0];
    let sar = low[0];
    let acceleration = af;
    
    result.push(sar);
    
    for (let i = 1; i < high.length; i++) {
      const prevSar = sar;
      
      if (isLong) {
        sar = prevSar + acceleration * (ep - prevSar);
        sar = Math.min(sar, low[i - 1], i > 1 ? low[i - 2] : low[i - 1]);
        
        if (high[i] > ep) {
          ep = high[i];
          acceleration = Math.min(acceleration + af, maxAf);
        }
        
        if (low[i] <= sar) {
          isLong = false;
          sar = ep;
          ep = low[i];
          acceleration = af;
        }
      } else {
        sar = prevSar + acceleration * (ep - prevSar);
        sar = Math.max(sar, high[i - 1], i > 1 ? high[i - 2] : high[i - 1]);
        
        if (low[i] < ep) {
          ep = low[i];
          acceleration = Math.min(acceleration + af, maxAf);
        }
        
        if (high[i] >= sar) {
          isLong = true;
          sar = ep;
          ep = high[i];
          acceleration = af;
        }
      }
      
      result.push(sar);
    }
    
    return result;
  }

  // Generic indicator calculator
  calculate(indicatorId: string, data: any[], params: Record<string, any>): any {
    const closes = data.map(d => d.close);
    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    const volumes = data.map(d => d.volume);
    
    switch(indicatorId) {
      case 'sma':
        return this.sma(closes, params.period || 20);
      case 'ema':
        return this.ema(closes, params.period || 20);
      case 'rsi':
        return this.rsi(closes, params.period || 14);
      case 'bollinger_bands':
        return this.bollingerBands(closes, params.period || 20, params.std_dev || 2);
      case 'macd':
        return this.macd(closes, params.fast_period || 12, params.slow_period || 26, params.signal_period || 9);
      case 'stoch':
      case 'stochastic':
        return this.stochastic(highs, lows, closes, params.k_period || 14, params.d_period || 3);
      case 'atr':
        return this.atr(highs, lows, closes, params.period || 14);
      case 'willr':
        return this.willr(highs, lows, closes, params.period || 14);
      case 'cci':
        return this.cci(highs, lows, closes, params.period || 20);
      case 'obv':
        return this.obv(closes, volumes);
      case 'mfi':
        return this.mfi(highs, lows, closes, volumes, params.period || 14);
      case 'sar':
        return this.sar(highs, lows, params.af || 0.02, params.max_af || 0.2);
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