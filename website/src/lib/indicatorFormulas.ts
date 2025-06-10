// Indicator formulas and usage examples
import { getCompleteIndicatorFormula } from './indicatorFormulasComplete';

export function getIndicatorFormula(indicatorId: string): string {
	// First try to get the complete formula
	const completeFormula = getCompleteIndicatorFormula(indicatorId);
	if (completeFormula && !completeFormula.includes('Mathematical definition for this indicator is being updated')) {
		return completeFormula;
	}
	
	// Fallback to existing formulas
	switch (indicatorId) {
		// Moving Averages
		case 'sma':
			return `SMA = (P₁ + P₂ + ... + Pₙ) / n

Where:
- P = Price for each period
- n = Number of periods`;

		case 'ema':
			return `EMA = (Price × α) + (Previous EMA × (1 - α))
α = 2 / (n + 1)

Where:
- α = Smoothing factor
- n = Number of periods`;

		case 'wma':
			return `WMA = (P₁×n + P₂×(n-1) + ... + Pₙ×1) / (n + (n-1) + ... + 1)

Where:
- P = Price for each period
- n = Number of periods`;

		case 'hma':
			return `HMA = WMA(2 × WMA(n/2) - WMA(n), √n)

Where:
- WMA = Weighted Moving Average
- n = Period
- √n = Square root of period`;

		case 'dema':
			return `DEMA = 2 × EMA - EMA(EMA)

Where:
- EMA = Exponential Moving Average of the same period`;

		case 'tema':
			return `TEMA = 3 × EMA - 3 × EMA(EMA) + EMA(EMA(EMA))

Triple smoothed exponential moving average`;

		case 'vwma':
			return `VWMA = Σ(Price × Volume) / Σ(Volume)

Where:
- Σ = Sum over the specified period
- Price and Volume for each period`;

		case 'vwap':
			return `VWAP = Σ(Typical Price × Volume) / Σ(Volume)
Typical Price = (High + Low + Close) / 3

Calculated from session start`;

		case 'alma':
			return `ALMA = Σ(Price × Weight) / Σ(Weight)
Weight = e^(-(i - m)² / (2 × s²))
m = offset × (n - 1)
s = n / sigma

Where:
- e = Euler's number
- i = Index (0 to n-1)
- offset = Phase offset (0.85 default)
- sigma = Smoothness factor (6 default)
- n = Period`;

		case 'kama':
			return `KAMA = Previous KAMA + SC × (Price - Previous KAMA)
SC = (ER × (fastest SC - slowest SC) + slowest SC)²
ER = |Change| / Volatility
Change = |Close - Close[n periods ago]|
Volatility = Σ|Close - Previous Close| over n periods

Where:
- SC = Smoothing Constant
- ER = Efficiency Ratio
- fastest SC = 2/(2+1) = 0.6667
- slowest SC = 2/(30+1) = 0.0645`;

		case 'jma':
			return `JMA uses Jurik's proprietary algorithm:
1. Calculate relative volatility
2. Apply adaptive smoothing based on volatility
3. Phase adjustment for reduced lag

Parameters:
- Length: Smoothing period
- Phase: -100 to +100 (lag adjustment)`;

		case 'frama':
			return `FRAMA = α × Price + (1 - α) × Previous FRAMA
α = e^(-4.6 × (D - 1))
D = (log(N₁ + N₂) - log(N₃)) / log(2)

Where:
- N₁ = (Max - Min) over first half of period
- N₂ = (Max - Min) over second half of period  
- N₃ = (Max - Min) over entire period
- D = Fractal dimension`;

		case 'mama':
			return `MAMA = α × Price + (1 - α) × Previous MAMA
FAMA = α/2 × MAMA + (1 - α/2) × Previous FAMA

α = 0.0962 / (Re[1] + 0.5778) if Re[1] > 0.5778
α = 0.5 if Re[1] ≤ 0.5778

Where Re[1] is the real part of the dominant cycle measurement`;

		case 'trima':
			return `TRIMA = SMA(SMA(Price, n), n)

For odd n: Period₁ = Period₂ = (n + 1) / 2
For even n: Period₁ = n / 2, Period₂ = (n / 2) + 1

Double smoothed moving average`;

		case 'zlema':
			return `ZLEMA = EMA(Price + (Price - Price[lag]), n)
lag = (n - 1) / 2

Where:
- EMA = Exponential Moving Average
- lag = Lag compensation factor`;

		case 'wilders':
			return `Wilder's Smoothing = (Previous Value × (n - 1) + Current Value) / n

Where:
- n = Smoothing period
- Similar to EMA with α = 1/n`;

		case 'sinwma':
			return `SINWMA = Σ(Price × sin(π × i / (n + 1))) / Σ(sin(π × i / (n + 1)))

Where:
- i = Index from 1 to n
- π = Pi (3.14159...)
- n = Period`;

		case 'linreg':
			return `LinReg = a + b × n
b = (n × Σ(xy) - Σ(x) × Σ(y)) / (n × Σ(x²) - (Σ(x))²)
a = (Σ(y) - b × Σ(x)) / n

Where:
- x = Period index (1, 2, ..., n)
- y = Price values
- Linear regression forecast`;

		case 'hwma':
			return `HWMA uses Holt-Winters exponential smoothing:
Level[t] = α × Price[t] + (1 - α) × (Level[t-1] + Trend[t-1])
Trend[t] = β × (Level[t] - Level[t-1]) + (1 - β) × Trend[t-1]

Where:
- α = Level smoothing parameter
- β = Trend smoothing parameter`;

		case 'pwma':
			return `PWMA = Σ(Price × Pascal Weight) / Σ(Pascal Weight)

Pascal Weights from Pascal's Triangle:
Row n: C(n,0), C(n,1), ..., C(n,n)
Where C(n,k) = n! / (k!(n-k)!)`;

		case 'swma':
			return `SWMA = Σ(Price × Weight) / Σ(Weight)

Symmetric weights: w[i] = w[n-1-i]
Typically using triangular or parabolic weighting`;

		case 'supersmoother':
			return `SuperSmoother = c₁ × (Price + Price[1]) + c₂ × SS[1] + c₃ × SS[2]

c₁ = a²/4
c₂ = 2a × cos(1.738 × 180/Period)
c₃ = -a²
a = e^(-1.414 × π / Period)

Where SS[1] = previous SuperSmoother value`;

		case 'supersmoother_3_pole':
			return `3-Pole SuperSmoother applies additional smoothing:
SS3 = α × SuperSmoother + (1 - α) × Previous SS3

Where α is calculated from the period parameter`;

		case 'gaussian':
			return `Gaussian Filter = Σ(Price × e^(-0.5 × ((i - μ)/σ)²))

Where:
- μ = (n - 1) / 2 (center of window)
- σ = n / (2 × poles)
- i = Index from 0 to n-1`;

		case 'highpass':
			return `HighPass = α × (Price - Previous Price) + α × Previous HighPass
α = (cos(1.414 × π / Period) + sin(1.414 × π / Period) - 1) / cos(1.414 × π / Period)

Removes low-frequency components below cutoff period`;

		case 'highpass_2_pole':
			return `2-Pole HighPass = c₁ × (Price - 2 × Price[1] + Price[2]) + c₂ × HP[1] + c₃ × HP[2]

c₁ = (1 - α/2)²
c₂ = 2 × (1 - α)
c₃ = -(1 - α)²
α = (cos(1.414 × π / Period) + sin(1.414 × π / Period) - 1) / cos(1.414 × π / Period)`;

		case 'reflex':
			return `Reflex = (Price - 2 × Price[2] + Price[4]) / 4

Simple 3-point difference filter for trend detection`;

		case 'trendflex':
			return `TrendFlex = (Price + 2 × Price[1] + Price[2]) / 4

Opposite of Reflex - emphasizes trend components`;

		case 'ehlers_itrend':
			return `ITrend = (α - α²/4) × Price + 0.5 × α² × Price[1] - (α - 0.75 × α²) × Price[2] + 2 × (1 - α) × ITrend[1] - (1 - α)² × ITrend[2]

Where α = 2 / (Period + 1)`;

		case 'vpwma':
			return `VPWMA = Σ(Price × Volume × Weight) / Σ(Volume × Weight)

Variable period weights adjusted by volume and time`;

		case 'cwma':
			return `CWMA = WMA applied symmetrically around center point

Centered Weighted Moving Average for better smoothing`;

		case 'sqwma':
			return `SQWMA = Σ(Price × i²) / Σ(i²)

Where:
- i = Index from 1 to n
- Quadratic weighting scheme`;

		case 'fwma':
			return `FWMA = Σ(Price × F[i]) / Σ(F[i])

Where F[i] are Fibonacci numbers:
F[0]=0, F[1]=1, F[n]=F[n-1]+F[n-2]`;

		case 'maaq':
			return `MAAQ = Previous MAAQ + α × (Price - Previous MAAQ)
α = ER × (fast_sc - slow_sc) + slow_sc

Where:
- ER = Efficiency Ratio (similar to KAMA)
- Adaptive Q factor adjustment`;

		case 'epma':
			return `EPMA = α × Price + (1 - α) × Previous EPMA
α = 2 / (1 + Period)

End Point Moving Average with exponential weighting`;

		case 'edcf':
			return `EDCF uses Ehlers Distance Coefficient:
Distance = √(Σ((Price[i] - Mean)²))
Coefficient = Distance / MaxDistance

Adaptive filter based on price distance from mean`;

		case 'jsa':
			return `JSA applies Jurik smoothing algorithm:
Similar to JMA but with different phase and smoothing parameters`;

		case 'mwdx':
			return `MWDX uses MESA Window Discrete Transform:
Applies windowed DFT for cycle analysis and smoothing`;

		case 'nma':
			return `NMA = e^(Σ(ln(Price)) / n)

Natural (logarithmic) moving average using geometric mean`;

		case 'srwma':
			return `SRWMA = Σ(Price × √i) / Σ(√i)

Where:
- i = Index from 1 to n
- Square root weighting scheme`;

		// Momentum Indicators
		case 'rsi':
			return `RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss

Where:
- Average Gain = EMA of gains over n periods
- Average Loss = EMA of losses over n periods
- n = Period (typically 14)`;

		case 'macd':
			return `MACD = EMA₁₂ - EMA₂₆
Signal = EMA₉(MACD)
Histogram = MACD - Signal

Where:
- EMA₁₂ = 12-period EMA
- EMA₂₆ = 26-period EMA
- EMA₉ = 9-period EMA of MACD`;

		case 'stoch':
			return `%K = ((C - L₁₄) / (H₁₄ - L₁₄)) × 100
%D = SMA₃(%K)

Where:
- C = Current close
- L₁₄ = Lowest low over 14 periods
- H₁₄ = Highest high over 14 periods
- SMA₃ = 3-period simple moving average`;

		case 'cci':
			return `CCI = (TP - SMA(TP)) / (0.015 × Mean Deviation)
TP = (High + Low + Close) / 3

Where:
- TP = Typical Price
- SMA = Simple Moving Average
- Mean Deviation = Average of absolute deviations`;

		case 'willr':
			return `%R = ((Highest High - Close) / (Highest High - Lowest Low)) × -100

Where:
- Highest High = Highest high over n periods
- Lowest Low = Lowest low over n periods
- Typically n = 14`;

		case 'roc':
			return `ROC = ((Price - Price₍ₙ₎) / Price₍ₙ₎) × 100

Where:
- Price = Current price
- Price₍ₙ₎ = Price n periods ago
- n = Number of periods`;

		case 'mom':
			return `Momentum = Price - Price₍ₙ₎

Where:
- Price = Current price
- Price₍ₙ₎ = Price n periods ago
- n = Number of periods`;

		// Volatility Indicators
		case 'atr':
			return `TR = max(|H - L|, |H - C₋₁|, |L - C₋₁|)
ATR = EMA(TR, n)

Where:
- H = Current High
- L = Current Low
- C₋₁ = Previous Close
- TR = True Range
- n = Period (typically 14)`;

		case 'bollinger_bands':
			return `Middle Band = SMA(n)
Upper Band = SMA(n) + (k × σ)
Lower Band = SMA(n) - (k × σ)

Where:
- SMA = Simple Moving Average
- n = Period (typically 20)
- k = Standard deviation multiplier (typically 2)
- σ = Standard deviation of price`;

		case 'stddev':
			return `σ = √(Σ(x - μ)² / n)

Where:
- σ = Standard deviation
- x = Individual price values
- μ = Mean of price values
- n = Number of periods`;

		case 'donchian':
			return `Upper Channel = Highest High over n periods
Lower Channel = Lowest Low over n periods
Middle Channel = (Upper + Lower) / 2

Where:
- n = Number of periods (typically 20)`;

		// Volume Indicators
		case 'obv':
			return `If Close > Previous Close: OBV = Previous OBV + Volume
If Close < Previous Close: OBV = Previous OBV - Volume
If Close = Previous Close: OBV = Previous OBV

On-Balance Volume accumulates volume based on price direction`;

		case 'mfi':
			return `Typical Price = (High + Low + Close) / 3
Money Flow = Typical Price × Volume
MFI = 100 - (100 / (1 + Money Flow Ratio))

Where Money Flow Ratio = Positive Money Flow / Negative Money Flow`;

		case 'ad':
			return `CLV = ((Close - Low) - (High - Close)) / (High - Low)
A/D = Previous A/D + (CLV × Volume)

Where:
- CLV = Close Location Value
- A/D = Accumulation/Distribution Line`;

		// Trend Indicators
		case 'adx':
			return `+DM = max(H - H₋₁, 0) if H - H₋₁ > L₋₁ - L, else 0
-DM = max(L₋₁ - L, 0) if L₋₁ - L > H - H₋₁, else 0
TR = max(H - L, |H - C₋₁|, |L - C₋₁|)
+DI = 100 × EMA(+DM) / EMA(TR)
-DI = 100 × EMA(-DM) / EMA(TR)
DX = 100 × |+DI - -DI| / (+DI + -DI)
ADX = EMA(DX)

Where H=High, L=Low, C=Close, subscript indicates periods ago`;

		case 'sar':
			return `SAR = Previous SAR + AF × (EP - Previous SAR)

Where:
- AF = Acceleration Factor (starts at 0.02)
- EP = Extreme Point (highest high or lowest low)
- AF increases by 0.02 each period up to maximum 0.20`;

		case 'supertrend':
			return `HL2 = (High + Low) / 2
Basic Upper Band = HL2 + (Multiplier × ATR)
Basic Lower Band = HL2 - (Multiplier × ATR)

SuperTrend follows the appropriate band based on closing price`;

		case 'aroon':
			return `Aroon Up = ((n - Periods Since Highest High) / n) × 100
Aroon Down = ((n - Periods Since Lowest Low) / n) × 100

Where:
- n = Number of periods (typically 14)`;

		case 'trix':
			return `Single Smoothed = EMA(Close)
Double Smoothed = EMA(Single Smoothed)
Triple Smoothed = EMA(Double Smoothed)
TRIX = (Triple Smoothed - Previous Triple Smoothed) / Previous Triple Smoothed × 10000`;

		// Statistical Indicators
		case 'avgprice':
			return `Average Price = (Open + High + Low + Close) / 4

Simple average of all four OHLC values`;

		case 'medprice':
			return `Median Price = (High + Low) / 2

Midpoint of the high-low range`;

		case 'wclprice':
			return `Weighted Close = (High + Low + 2×Close) / 4

Weighted average giving double weight to closing price`;

		default:
			return `${indicatorId.toUpperCase()} Formula

Mathematical definition and calculation method for this indicator will be displayed here.

Please refer to the documentation for specific implementation details.`;
	}
}

export function getUsageExample(indicatorId: string): string {
	const functionName = `calculate_${indicatorId}`;
	
	switch (indicatorId) {
		case 'rsi':
			return `// Import the WASM module
import init, { ${functionName} } from './pkg/rust_backtester.js';

async function calculateRSI() {
    // Initialize WASM module
    await init();
    
    // Sample price data (closing prices)
    const prices = [
        44.94, 44.95, 45.15, 45.29, 45.41,
        45.23, 45.08, 45.15, 45.57, 45.32,
        45.15, 45.39, 45.83, 45.85, 46.08
    ];
    
    // Calculate RSI with period of 14
    const rsiValues = ${functionName}(new Float64Array(prices), 14);
    
    console.log('RSI Values:', Array.from(rsiValues));
}`;

		case 'sma':
			return `// Import the WASM module
import init, { ${functionName} } from './pkg/rust_backtester.js';

async function calculateSMA() {
    await init();
    
    const prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    const period = 5;
    
    const smaValues = ${functionName}(new Float64Array(prices), period);
    console.log('SMA Values:', Array.from(smaValues));
}`;

		case 'ema':
			return `// Import the WASM module
import init, { ${functionName} } from './pkg/rust_backtester.js';

async function calculateEMA() {
    await init();
    
    const prices = [22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29];
    const period = 10;
    
    const emaValues = ${functionName}(new Float64Array(prices), period);
    console.log('EMA Values:', Array.from(emaValues));
}`;

		case 'macd':
			return `// Import the WASM module
import init, { ${functionName} } from './pkg/rust_backtester.js';

async function calculateMACD() {
    await init();
    
    const prices = [/* your price data */];
    const fastPeriod = 12;
    const slowPeriod = 26;
    const signalPeriod = 9;
    
    const result = ${functionName}(
        new Float64Array(prices), 
        fastPeriod, 
        slowPeriod, 
        signalPeriod
    );
    
    console.log('MACD Line:', Array.from(result.macd));
    console.log('Signal Line:', Array.from(result.signal));
    console.log('Histogram:', Array.from(result.histogram));
}`;

		case 'bollinger_bands':
			return `// Import the WASM module
import init, { ${functionName} } from './pkg/rust_backtester.js';

async function calculateBollingerBands() {
    await init();
    
    const prices = [/* your price data */];
    const period = 20;
    const stdDev = 2.0;
    
    const result = ${functionName}(
        new Float64Array(prices), 
        period, 
        stdDev
    );
    
    console.log('Upper Band:', Array.from(result.upper));
    console.log('Middle Band:', Array.from(result.middle));
    console.log('Lower Band:', Array.from(result.lower));
}`;

		case 'atr':
			return `// Import the WASM module
import init, { ${functionName} } from './pkg/rust_backtester.js';

async function calculateATR() {
    await init();
    
    const highs = [/* high prices */];
    const lows = [/* low prices */];
    const closes = [/* close prices */];
    const period = 14;
    
    const atrValues = ${functionName}(
        new Float64Array(highs),
        new Float64Array(lows),
        new Float64Array(closes),
        period
    );
    
    console.log('ATR Values:', Array.from(atrValues));
}`;

		case 'stoch':
			return `// Import the WASM module
import init, { ${functionName} } from './pkg/rust_backtester.js';

async function calculateStochastic() {
    await init();
    
    const highs = [/* high prices */];
    const lows = [/* low prices */];
    const closes = [/* close prices */];
    const kPeriod = 14;
    const dPeriod = 3;
    
    const result = ${functionName}(
        new Float64Array(highs),
        new Float64Array(lows),
        new Float64Array(closes),
        kPeriod,
        dPeriod
    );
    
    console.log('%K Values:', Array.from(result.k));
    console.log('%D Values:', Array.from(result.d));
}`;

		default:
			return `// Import the WASM module
import init, { ${functionName} } from './pkg/rust_backtester.js';

async function calculate${indicatorId.toUpperCase()}() {
    // Initialize WASM module
    await init();
    
    // Prepare your market data
    const data = [/* your market data */];
    
    // Calculate ${indicatorId.toUpperCase()} with appropriate parameters
    const result = ${functionName}(data, /* parameters */);
    
    console.log('${indicatorId.toUpperCase()} Values:', result);
}

// Example usage
calculate${indicatorId.toUpperCase()}().then(() => {
    console.log('Calculation complete');
}).catch(err => {
    console.error('Error:', err);
});`;
	}
}

export function isOscillator(indicatorId: string): boolean {
	const oscillators = [
		'rsi', 'stoch', 'stochf', 'cci', 'willr', 'mfi', 'macd', 'ppo', 'apo',
		'roc', 'rocp', 'rocr', 'mom', 'srsi', 'cmo', 'aroonosc', 'fosc', 'tsi',
		'ultosc', 'kdj', 'stc', 'squeeze_momentum', 'lrsi', 'rsx', 'ift_rsi',
		'cg', 'ao', 'gatorosc', 'fisher', 'chande', 'dti', 'er', 'eri', 'kst',
		'pfe', 'qstick', 'rsmk', 'coppock', 'wavetrend', 'msw', 'dec_osc',
		'acosc', 'bop', 'cfo', 'correlation_cycle', 'voss', 'ttm_trend', 'cksp'
	];
	return oscillators.includes(indicatorId);
}

export function isMovingAverage(indicatorId: string): boolean {
	const movingAverages = [
		'sma', 'ema', 'wma', 'dema', 'tema', 'hma', 'alma', 'vwma', 'vwap',
		'smma', 'kama', 'jma', 'tilson', 'frama', 'mama', 'trima', 'zlema',
		'wilders', 'sinwma', 'linreg', 'hwma', 'pwma', 'swma', 'supersmoother',
		'supersmoother_3_pole', 'gaussian', 'highpass', 'highpass_2_pole',
		'reflex', 'trendflex', 'ehlers_itrend', 'vpwma', 'cwma', 'sqwma',
		'fwma', 'maaq', 'epma', 'edcf', 'jsa', 'mwdx', 'nma', 'srwma'
	];
	return movingAverages.includes(indicatorId);
}

export function isVolatilityIndicator(indicatorId: string): boolean {
	const volatilityIndicators = [
		'atr', 'natr', 'bollinger_bands', 'bollinger_bands_width', 'keltner',
		'donchian', 'stddev', 'var', 'chop', 'ui', 'damiani_volatmeter',
		'mass', 'rvi', 'kurtosis', 'cvi', 'deviation', 'bandpass', 'decycler'
	];
	return volatilityIndicators.includes(indicatorId);
}

export function isVolumeIndicator(indicatorId: string): boolean {
	const volumeIndicators = [
		'obv', 'mfi', 'ad', 'adosc', 'emv', 'vpt', 'nvi', 'pvi', 'efi',
		'kvo', 'marketefi', 'vosc', 'vwmacd', 'wad', 'vpci', 'vlma'
	];
	return volumeIndicators.includes(indicatorId);
}

export function isTrendIndicator(indicatorId: string): boolean {
	const trendIndicators = [
		'adx', 'adxr', 'dx', 'di', 'dm', 'aroon', 'sar', 'supertrend', 'trix',
		'dpo', 'pma', 'linearreg_slope', 'linearreg_angle', 'linearreg_intercept',
		'tsf', 'alligator', 'vi', 'vidya', 'mab', 'devstop', 'safezonestop',
		'kaufmanstop', 'ht_trendline', 'ht_trendmode', 'ht_dcperiod',
		'ht_dcphase', 'ht_phasor', 'ht_sine', 'pivot', 'correl_hl', 'emd',
		'heikin_ashi_candles', 'zscore'
	];
	return trendIndicators.includes(indicatorId);
}

export function isStatisticalIndicator(indicatorId: string): boolean {
	const statisticalIndicators = [
		'avgprice', 'medprice', 'wclprice', 'midpoint', 'midprice', 'minmax',
		'mean_ad', 'medium_ad'
	];
	return statisticalIndicators.includes(indicatorId);
}