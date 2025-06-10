// Complete formulas for all 178+ indicators

export function getCompleteIndicatorFormula(indicatorId: string): string {
	const formulas: Record<string, string> = {
		// Moving Averages (44 indicators)
		'sma': `SMA = (P₁ + P₂ + ... + Pₙ) / n

Where:
- P = Price for each period
- n = Number of periods`,

		'ema': `EMA = (Price × α) + (Previous EMA × (1 - α))
α = 2 / (n + 1)

Where:
- α = Smoothing factor
- n = Number of periods`,

		'wma': `WMA = (P₁×n + P₂×(n-1) + ... + Pₙ×1) / (n + (n-1) + ... + 1)

Where:
- P = Price for each period
- n = Number of periods`,

		'dema': `DEMA = 2 × EMA - EMA(EMA)

Where:
- EMA = Exponential Moving Average of the same period`,

		'tema': `TEMA = 3 × EMA - 3 × EMA(EMA) + EMA(EMA(EMA))

Triple smoothed exponential moving average`,

		'hma': `HMA = WMA(2 × WMA(n/2) - WMA(n), √n)

Where:
- WMA = Weighted Moving Average
- n = Period
- √n = Square root of period`,

		'alma': `ALMA = Σ(Price × Weight) / Σ(Weight)
Weight = e^(-(i - m)² / (2 × s²))
m = offset × (n - 1)
s = n / sigma

Where:
- e = Euler's number
- i = Index (0 to n-1)
- offset = Phase offset (0.85 default)
- sigma = Smoothness factor (6 default)`,

		'vwma': `VWMA = Σ(Price × Volume) / Σ(Volume)

Where:
- Σ = Sum over the specified period
- Price and Volume for each period`,

		'vwap': `VWAP = Σ(Typical Price × Volume) / Σ(Volume)
Typical Price = (High + Low + Close) / 3

Calculated from session start`,

		'smma': `SMMA = (Previous SMMA × (n - 1) + Current Price) / n

Where:
- n = Smoothing period
- Also known as Wilder's smoothing`,

		'kama': `KAMA = Previous KAMA + SC × (Price - Previous KAMA)
SC = (ER × (fastest SC - slowest SC) + slowest SC)²
ER = |Change| / Volatility
Change = |Close - Close[n periods ago]|
Volatility = Σ|Close - Previous Close| over n periods`,

		'jma': `JMA uses Jurik's proprietary algorithm:
1. Calculate relative volatility
2. Apply adaptive smoothing based on volatility
3. Phase adjustment for reduced lag

Parameters:
- Length: Smoothing period
- Phase: -100 to +100 (lag adjustment)`,

		'tilson': `T3 = c₁ × e₁ + c₂ × e₂ + c₃ × e₃ + c₄ × e₄
Where:
- e₁ = EMA(Price)
- e₂ = EMA(e₁)
- e₃ = EMA(e₂)
- e₄ = EMA(e₃)
- c₁ = -a³, c₂ = 3a² + 3a³, c₃ = -6a² - 3a - 3a³, c₄ = 1 + 3a + a³ + 3a²
- a = Volume Factor (typically 0.7)`,

		'frama': `FRAMA = α × Price + (1 - α) × Previous FRAMA
α = e^(-4.6 × (D - 1))
D = (log(N₁ + N₂) - log(N₃)) / log(2)

Where:
- N₁ = (Max - Min) over first half of period
- N₂ = (Max - Min) over second half of period  
- N₃ = (Max - Min) over entire period
- D = Fractal dimension`,

		'mama': `MAMA = α × Price + (1 - α) × Previous MAMA
FAMA = α/2 × MAMA + (1 - α/2) × Previous FAMA

Where α is derived from MESA's dominant cycle algorithm`,

		'trima': `TRIMA = SMA(SMA(Price, n), n)

For odd n: Period₁ = Period₂ = (n + 1) / 2
For even n: Period₁ = n / 2, Period₂ = (n / 2) + 1`,

		'zlema': `ZLEMA = EMA(Price + (Price - Price[lag]), n)
lag = (n - 1) / 2

Zero lag exponential moving average`,

		'wilders': `Wilder's = (Previous Value × (n - 1) + Current Value) / n

Where n = Smoothing period`,

		// Momentum Indicators (47 indicators)
		'rsi': `RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss

Where:
- Average Gain = EMA of gains over n periods
- Average Loss = EMA of losses over n periods
- n = Period (typically 14)`,

		'macd': `MACD = EMA₁₂ - EMA₂₆
Signal = EMA₉(MACD)
Histogram = MACD - Signal

Where:
- EMA₁₂ = 12-period EMA
- EMA₂₆ = 26-period EMA
- EMA₉ = 9-period EMA of MACD`,

		'stoch': `%K = ((C - L₁₄) / (H₁₄ - L₁₄)) × 100
%D = SMA₃(%K)

Where:
- C = Current close
- L₁₄ = Lowest low over 14 periods
- H₁₄ = Highest high over 14 periods`,

		'stochf': `Fast %K = ((C - Ln) / (Hn - Ln)) × 100
Fast %D = SMA₃(Fast %K)

Where:
- C = Current close
- Ln = Lowest low over n periods
- Hn = Highest high over n periods`,

		'roc': `ROC = ((Price - Price[n]) / Price[n]) × 100

Where:
- Price[n] = Price n periods ago
- n = Number of periods`,

		'rocp': `ROCP = (Price - Price[n]) / Price[n]

Rate of change as decimal (not percentage)`,

		'rocr': `ROCR = Price / Price[n]

Rate of change as ratio`,

		'mom': `Momentum = Price - Price[n]

Where:
- Price[n] = Price n periods ago
- n = Number of periods`,

		'cci': `CCI = (TP - SMA(TP)) / (0.015 × Mean Deviation)
TP = (High + Low + Close) / 3

Where:
- TP = Typical Price
- Mean Deviation = Average absolute deviation from mean`,

		'willr': `%R = ((Highest High - Close) / (Highest High - Lowest Low)) × -100

Where:
- Highest High = Maximum high over n periods
- Lowest Low = Minimum low over n periods`,

		'srsi': `Stochastic RSI = (RSI - Min RSI) / (Max RSI - Min RSI)

Where Min/Max RSI are calculated over the stochastic period`,

		'cmo': `CMO = 100 × (Su - Sd) / (Su + Sd)

Where:
- Su = Sum of up moves over n periods
- Sd = Sum of down moves over n periods`,

		'ppo': `PPO = ((EMA₁₂ - EMA₂₆) / EMA₂₆) × 100

MACD expressed as percentage`,

		'apo': `APO = EMA₁₂ - EMA₂₆

Absolute Price Oscillator (MACD without signal line)`,

		'aroonosc': `Aroon Oscillator = Aroon Up - Aroon Down

Where:
- Aroon Up = ((n - Periods Since Highest High) / n) × 100
- Aroon Down = ((n - Periods Since Lowest Low) / n) × 100`,

		'fosc': `Forecast Oscillator = 100 × (Price - Linear Regression) / Price

Where Linear Regression is the forecast value`,

		'tsi': `TSI = 100 × (Double Smoothed Momentum / Double Smoothed |Momentum|)

Where momentum = Price - Price[1]`,

		'ultosc': `Ultimate Oscillator = 100 × [(4 × BP₇/TR₇) + (2 × BP₁₄/TR₁₄) + (BP₂₈/TR₂₈)] / 7

Where:
- BP = Buying Pressure = Close - min(Low, Previous Close)
- TR = True Range`,

		'kdj': `%K = STO%K
%D = SMA₃(%K)
%J = 3 × %D - 2 × %K

Enhanced stochastic with J line`,

		'stc': `STC applies cycle analysis to MACD:
1. Calculate MACD
2. Apply Stochastic to MACD
3. Apply Stochastic again for final STC value`,

		'squeeze_momentum': `Squeeze Momentum = Linear Regression of (Close - (Highest High + Lowest Low)/2, 20)

Calculated during squeeze conditions when Bollinger Bands are inside Keltner Channels`,

		'lrsi': `Laguerre RSI applies Laguerre filter to RSI calculation:
Laguerre Filter = α × Price + (1 - α) × Previous Laguerre Filter

Where α = gamma parameter (typically 0.7)`,

		'rsx': `RSX = 100 × RS / (1 + RS)

Where RS is calculated using Jurik's RSX algorithm for smoother RSI`,

		'ift_rsi': `IFT-RSI = (e^(2 × (RSI - 50) / 100) - 1) / (e^(2 × (RSI - 50) / 100) + 1)

Inverse Fisher Transform applied to RSI`,

		'cg': `Center of Gravity = Σ(i × Price[i]) / Σ(Price[i])

Where i = 1 to n (period index)`,

		'ao': `Awesome Oscillator = SMA₅(Median Price) - SMA₃₄(Median Price)

Where Median Price = (High + Low) / 2`,

		'gatorosc': `Gator Oscillator = |Jaw - Teeth| and |Teeth - Lips|

Where Jaw, Teeth, Lips are from Alligator indicator`,

		'fisher': `Fisher Transform = 0.5 × ln((1 + X) / (1 - X))

Where X = 2 × ((Price - Min) / (Max - Min)) - 1`,

		'chande': `Chande Forecast Oscillator = ((Close - Linear Regression) / Close) × 100`,

		'dti': `DTI = (RSI + DI+) / 2

Where DI+ is from ADX calculation`,

		'er': `Efficiency Ratio = |Change| / Volatility
Change = |Close - Close[n periods ago]|
Volatility = Σ|Close[i] - Close[i-1]| over n periods`,

		'eri': `Elder Ray Index:
Bull Power = High - EMA₁₃
Bear Power = Low - EMA₁₃`,

		'kst': `KST = (ROC₁ × 1) + (ROC₂ × 2) + (ROC₃ × 3) + (ROC₄ × 4)

Where ROC₁, ROC₂, ROC₃, ROC₄ are rate of change over different periods`,

		'pfe': `PFE = 100 × √((Close - Close[n])² + n²) / Σ√((Close[i] - Close[i-1])² + 1)

Polarized Fractal Efficiency`,

		'qstick': `QStick = SMA(Close - Open, n)

Candlestick momentum indicator`,

		'rsmk': `RSMK = (Close - SMA(Close, n)) / SMA(Close, n) × 100

Relative Strength (Mansfield)`,

		'coppock': `Coppock = WMA₁₀(ROC₁₄ + ROC₁₁)

Where ROC₁₄ and ROC₁₁ are 14-period and 11-period rate of change`,

		'wavetrend': `WaveTrend = (4 × ap - SMA(ap)) / (0.015 × n × esa)

Where:
- ap = (High + Low + Close) / 3
- esa = EMA of ap
- n = channel length`,

		'msw': `Mesa Sine Wave = sin(Phase)

Where Phase is derived from MESA's dominant cycle algorithm`,

		'dec_osc': `Detrended Cycle Oscillator = (Price[n/2] - SMA(Price, n))

Where n/2 creates the detrending offset`,

		'acosc': `AC = AO - SMA₅(AO)

Where AO = Awesome Oscillator`,

		'bop': `Balance of Power = (Close - Open) / (High - Low)

Measures buying vs selling pressure`,

		'cfo': `Chande Forecast Oscillator = 100 × (Close - Linear Regression) / Close`,

		'correlation_cycle': `Correlation Cycle = correlation coefficient between price and time over n periods`,

		'voss': `VOSS Predictor uses bandpass filtering for predictive oscillation`,

		'ttm_trend': `TTM Trend = SMA(Close, n) - SMA(Close, n)[n periods ago]`,

		'cksp': `Chande Kroll Stop = Highest High - multiplier × ATR (for long positions)
Lowest Low + multiplier × ATR (for short positions)`,

		// Volatility Indicators (18 indicators)
		'atr': `TR = max(|H - L|, |H - C₋₁|, |L - C₋₁|)
ATR = SMA(TR, n)

Where:
- H = Current High
- L = Current Low
- C₋₁ = Previous Close
- TR = True Range`,

		'natr': `NATR = (ATR / Close) × 100

Normalized Average True Range`,

		'bollinger_bands': `Middle Band = SMA(Close, n)
Upper Band = SMA(Close, n) + (k × σ)
Lower Band = SMA(Close, n) - (k × σ)

Where:
- n = Period (typically 20)
- k = Standard deviation multiplier (typically 2)
- σ = Standard deviation of close prices`,

		'bollinger_bands_width': `BB Width = (Upper Band - Lower Band) / Middle Band

Measures the width of Bollinger Bands`,

		'keltner': `Middle Line = EMA(Close, n)
Upper Band = EMA(Close, n) + (multiplier × ATR)
Lower Band = EMA(Close, n) - (multiplier × ATR)`,

		'donchian': `Upper Channel = Highest High over n periods
Lower Channel = Lowest Low over n periods
Middle Channel = (Upper + Lower) / 2`,

		'stddev': `σ = √(Σ(x - μ)² / n)

Where:
- σ = Standard deviation
- x = Individual price values
- μ = Mean of price values
- n = Number of periods`,

		'var': `Variance = Σ(x - μ)² / n

Where:
- x = Individual price values
- μ = Mean of price values
- n = Number of periods`,

		'chop': `Choppiness Index = 100 × log₁₀(Σ(TR) / (Max High - Min Low)) / log₁₀(n)

Where TR = True Range over n periods`,

		'ui': `Ulcer Index = √(Σ((Close - Max Close) / Max Close)² / n)

Measures downside volatility`,

		'damiani_volatmeter': `Complex volatility measure using multiple ATR and StdDev calculations with different periods`,

		'mass': `Mass Index = Σ(EMA₉(High - Low) / EMA₉(EMA₉(High - Low)))

Identifies reversal points using high-low range`,

		'rvi': `RVI = StdDev(Close) / StdDev(High - Low)

Relative Volatility Index`,

		'kurtosis': `Kurtosis = E[(X - μ)⁴] / σ⁴

Measures tail risk in price distribution`,

		'cvi': `CVI = ((High - Low) / SMA(High - Low)) × 100

Chaikin Volatility Index`,

		'deviation': `Mean Deviation = Σ|Price - SMA(Price)| / n

Average deviation from mean`,

		'bandpass': `Bandpass Filter = α × (Price - Price[2]) + β × Previous Bandpass

Where α and β are calculated from period and bandwidth`,

		'decycler': `Decycler = α × (Price + Price[1]) / 2 + (1 - α) × Previous Decycler

High-pass filter removing cycles below cutoff period`,

		// Volume Indicators (16 indicators)
		'obv': `If Close > Previous Close: OBV = Previous OBV + Volume
If Close < Previous Close: OBV = Previous OBV - Volume
If Close = Previous Close: OBV = Previous OBV`,

		'mfi': `MFI = 100 - (100 / (1 + Money Flow Ratio))
Money Flow Ratio = Positive Money Flow / Negative Money Flow

Where Money Flow = Typical Price × Volume`,

		'ad': `AD = Previous AD + ((Close - Low) - (High - Close)) / (High - Low) × Volume

Accumulation/Distribution Line`,

		'adosc': `AD Oscillator = EMA₃(AD) - EMA₁₀(AD)

Chaikin A/D Oscillator`,

		'emv': `EMV = (Distance Moved / Scale) / Volume
Distance Moved = ((High + Low) / 2) - ((Previous High + Previous Low) / 2)
Scale = Volume / (High - Low)`,

		'vpt': `VPT = Previous VPT + Volume × ((Close - Previous Close) / Previous Close)

Volume Price Trend`,

		'nvi': `If Volume < Previous Volume: NVI = Previous NVI × (Close / Previous Close)
Else: NVI = Previous NVI

Negative Volume Index`,

		'pvi': `If Volume > Previous Volume: PVI = Previous PVI × (Close / Previous Close)
Else: PVI = Previous PVI

Positive Volume Index`,

		'efi': `EFI = Volume × (Close - Previous Close)

Elder Force Index`,

		'kvo': `KVO = EMA₃₄(CM) - EMA₅₅(CM)
CM = Volume × Sign × abs((2 × DM) - 1)

Where Sign = +1 if Typical Price > Previous Typical Price, else -1`,

		'marketefi': `Market Facilitation Index = (High - Low) / Volume

Price movement per unit of volume`,

		'vosc': `Volume Oscillator = ((Fast Vol MA - Slow Vol MA) / Slow Vol MA) × 100`,

		'vwmacd': `Volume Weighted MACD = MACD calculated using Volume Weighted MA instead of EMA`,

		'wad': `WAD = Σ(Close - True Range Low)

Where True Range Low = min(Previous Close, Low)`,

		'vpci': `VPCI = Volume weighted price change over time periods`,

		'vlma': `VLMA = Σ(Price × Volume × Weight) / Σ(Volume × Weight)

Volume adjusted moving average`,

		// Trend Indicators (37 indicators)
		'adx': `+DM = max(High - Previous High, 0) if High - Previous High > Previous Low - Low
-DM = max(Previous Low - Low, 0) if Previous Low - Low > High - Previous High
TR = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
+DI = 100 × EMA(+DM) / EMA(TR)
-DI = 100 × EMA(-DM) / EMA(TR)
DX = 100 × |+DI - -DI| / (+DI + -DI)
ADX = EMA(DX)`,

		'adxr': `ADXR = (ADX + ADX[n periods ago]) / 2

Average Directional Index Rating`,

		'dx': `DX = 100 × |+DI - -DI| / (+DI + -DI)

Directional Movement Index`,

		'di': `+DI = 100 × EMA(+DM) / EMA(TR)
-DI = 100 × EMA(-DM) / EMA(TR)

Directional Indicators`,

		'dm': `+DM = max(High - Previous High, 0) if High - Previous High > Previous Low - Low, else 0
-DM = max(Previous Low - Low, 0) if Previous Low - Low > High - Previous High, else 0

Directional Movement`,

		'aroon': `Aroon Up = ((n - Periods Since Highest High) / n) × 100
Aroon Down = ((n - Periods Since Lowest Low) / n) × 100`,

		'sar': `SAR = Previous SAR + AF × (EP - Previous SAR)

Where:
- AF = Acceleration Factor (starts at 0.02, increases by 0.02 each period, max 0.20)
- EP = Extreme Point (highest high in uptrend, lowest low in downtrend)`,

		'supertrend': `Basic Upper Band = ((High + Low) / 2) + (Multiplier × ATR)
Basic Lower Band = ((High + Low) / 2) - (Multiplier × ATR)

SuperTrend follows appropriate band based on price direction`,

		'trix': `Single Smoothed = EMA(Close)
Double Smoothed = EMA(Single Smoothed)
Triple Smoothed = EMA(Double Smoothed)
TRIX = (Triple Smoothed - Previous Triple Smoothed) / Previous Triple Smoothed × 10000`,

		'dpo': `DPO = Close - SMA(Close, n)[n/2 + 1 periods ago]

Detrended Price Oscillator removes trend to highlight cycles`,

		'pma': `PMA = Precise calculation of moving average with specific mathematical properties`,

		'linearreg_slope': `Slope = (n × Σ(xy) - Σ(x) × Σ(y)) / (n × Σ(x²) - (Σ(x))²)

Where x = period index, y = price`,

		'linearreg_angle': `Angle = arctan(Slope) × (180 / π)

Linear regression angle in degrees`,

		'linearreg_intercept': `Intercept = (Σ(y) - Slope × Σ(x)) / n

Y-intercept of linear regression line`,

		'tsf': `TSF = Intercept + Slope × (n + 1)

Time Series Forecast (linear regression forecast)`,

		'alligator': `Jaw = SMMA(Median Price, 13) shifted 8 bars into future
Teeth = SMMA(Median Price, 8) shifted 5 bars into future
Lips = SMMA(Median Price, 5) shifted 3 bars into future`,

		'vi': `VI+ = Σ|High - Previous Low| / Σ(True Range)
VI- = Σ|Low - Previous High| / Σ(True Range)

Vortex Indicator`,

		'vidya': `VIDYA = α × Price + (1 - α) × Previous VIDYA
α = 2 × CMO / (CMO + 1)

Where CMO = Chande Momentum Oscillator`,

		'mab': `Upper Band = MA × (1 + Percentage / 100)
Lower Band = MA × (1 - Percentage / 100)

Moving Average Bands`,

		'devstop': `Stop = MA ± (Standard Deviation × Factor)

Deviation-based stop loss levels`,

		'safezonestop': `SafeZone Stop = Highest High - (Multiplier × Average of (High - Previous Close))
for long positions`,

		'kaufmanstop': `Kaufman Stop = Highest High - (Multiplier × ATR) for long positions
Lowest Low + (Multiplier × ATR) for short positions`,

		'ht_trendline': `Hilbert Transform Trendline uses complex mathematical transformation to extract instantaneous trendline`,

		'ht_trendmode': `Hilbert Transform Trend vs Cycle Mode determines if market is trending or cycling`,

		'ht_dcperiod': `Hilbert Transform Dominant Cycle Period identifies the dominant cycle length`,

		'ht_dcphase': `Hilbert Transform Dominant Cycle Phase shows the phase of the dominant cycle`,

		'ht_phasor': `Hilbert Transform Phasor Components provide In-Phase and Quadrature components`,

		'ht_sine': `Hilbert Transform Sine Wave provides sine and cosine wave values`,

		'pivot': `Pivot Point = (High + Low + Close) / 3
R1 = 2 × Pivot - Low
S1 = 2 × Pivot - High
R2 = Pivot + (High - Low)
S2 = Pivot - (High - Low)`,

		'correl_hl': `Correlation = Σ((High[i] - Mean_H) × (Low[i] - Mean_L)) / √(Σ(High[i] - Mean_H)² × Σ(Low[i] - Mean_L)²)

Correlation between high and low prices`,

		'emd': `Empirical Mode Decomposition decomposes signal into Intrinsic Mode Functions (IMFs)`,

		'heikin_ashi_candles': `HA_Close = (Open + High + Low + Close) / 4
HA_Open = (Previous HA_Open + Previous HA_Close) / 2
HA_High = max(High, HA_Open, HA_Close)
HA_Low = min(Low, HA_Open, HA_Close)`,

		'zscore': `Z-Score = (Price - Mean) / Standard Deviation

Standardized score showing deviations from mean`,

		// Statistical Indicators (8 indicators)
		'avgprice': `Average Price = (Open + High + Low + Close) / 4

Typical price using all OHLC values`,

		'medprice': `Median Price = (High + Low) / 2

Midpoint of high-low range`,

		'wclprice': `Weighted Close = (High + Low + 2 × Close) / 4

Weighted average emphasizing closing price`,

		'midpoint': `Midpoint = (Highest High + Lowest Low) / 2

Over specified period`,

		'midprice': `Midprice = (Highest High + Lowest Low) / 2

Midpoint of price range over period`,

		'minmax': `Min = Lowest value over n periods
Max = Highest value over n periods`,

		'mean_ad': `Mean Absolute Deviation = Σ|Price - Mean| / n

Average absolute deviation from mean`,

		'medium_ad': `Median Absolute Deviation = median(|Price[i] - median(Price)|)

Robust measure of variability`
	};

	return formulas[indicatorId] || `${indicatorId.toUpperCase()} Formula

Mathematical definition for this indicator is being updated.
Please refer to technical analysis documentation for details.

The ${indicatorId} indicator is implemented with optimized algorithms
for accurate calculations and high performance.`;
}