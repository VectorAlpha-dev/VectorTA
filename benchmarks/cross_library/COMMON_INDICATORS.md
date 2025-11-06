# Common Indicators Between Rust-TA and Tulip

## Currently Benchmarked (10)
1. SMA - Simple Moving Average
2. EMA - Exponential Moving Average  
3. RSI - Relative Strength Index
4. Bollinger Bands
5. MACD
6. ATR - Average True Range
7. Stochastic
8. Aroon
9. ADX - Average Directional Index
10. CCI - Commodity Channel Index

## Additional Common Indicators Available (50+)

### Overlap Studies (Moving Averages)
- DEMA - Double Exponential Moving Average
- TEMA - Triple Exponential Moving Average
- KAMA - Kaufman Adaptive Moving Average
- WMA - Weighted Moving Average
- VWMA - Volume Weighted Moving Average
- TRIMA - Triangular Moving Average
- HMA - Hull Moving Average
- VIDYA - Variable Index Dynamic Average
- ZLEMA - Zero Lag Exponential Moving Average

### Momentum Indicators
- APO - Absolute Price Oscillator
- CMO - Chande Momentum Oscillator
- DPO - Detrended Price Oscillator
- MOM - Momentum
- PPO - Percentage Price Oscillator
- ROC - Rate of Change
- ROCR - Rate of Change Ratio
- STOCHRSI - Stochastic RSI
- TSI - True Strength Index (as TRIX in Tulip)
- WILLR - Williams %R
- AROONOSC - Aroon Oscillator

### Volatility Indicators
- NATR - Normalized ATR
- Keltner Channels (as KELTNER)

### Volume Indicators
- AD - Accumulation/Distribution
- ADOSC - AD Oscillator
- EMV - Ease of Movement
- MFI - Money Flow Index
- NVI - Negative Volume Index
- OBV - On Balance Volume
- PVI - Positive Volume Index
- VWAP (as VWMA in Tulip)
- WAD - Williams Accumulation/Distribution

### Trend Indicators
- ADXR - ADX Rating
- DI - Directional Indicator
- DM - Directional Movement
- DX - Directional Index
- PSAR - Parabolic SAR
- TSF - Time Series Forecast

### Statistical Functions
- LINEARREG - Linear Regression
- LINEARREG_INTERCEPT
- LINEARREG_SLOPE
- STDDEV - Standard Deviation
- VAR - Variance
- CORREL - Correlation

### Price Transforms
- AVGPRICE - Average Price
- MEDPRICE - Median Price
- TYPPRICE - Typical Price
- WCPRICE - Weighted Close Price

### Other Indicators
- AO - Awesome Oscillator
- BOP - Balance of Power
- CVI - Chaikin Volatility Index
- FISHER - Fisher Transform
- FOSC - Forecast Oscillator
- MASS - Mass Index
- MSW - Mesa Sine Wave
- QSTICK
- ULTOSC - Ultimate Oscillator
- VHF - Vertical Horizontal Filter
- VOSC - Volume Oscillator

## Indicators Only in Rust-TA (Not in Tulip)
- Alligator
- AlphaTrend
- Bollinger Bands Width
- Chandelier Exit
- CHOP - Choppiness Index
- Coppock Curve
- CoraWave
- Damiani Volatmeter
- Donchian Channels
- Gator Oscillator
- IFT_RSI
- Kaufman Stop
- KDJ
- Keltner Channels (full implementation)
- KST - Know Sure Thing
- Kurtosis
- LRSI - Laguerre RSI
- MAB - Moving Average Breakout
- Market Efficiency
- Pivot Points
- RSX
- RVI - Relative Vigor Index
- SafeZone Stop
- Squeeze Momentum
- SRSI - Stochastic RSI (different from STOCHRSI)
- STC - Schaff Trend Cycle
- Supertrend
- TTM Trend
- UI - Ulcer Index
- VOSS
- VPCI
- VPT - Volume Price Trend
- VWMACD
- Wavetrend
- Z-Score

## Indicators Only in Tulip (Not in Rust-TA)
- Mathematical operations (abs, acos, add, asin, atan, ceil, cos, cosh, div, exp, floor, ln, log10, max, min, mul, round, sin, sinh, sqrt, sub, sum, tan, tanh, todeg, torad, trunc)
- CROSSANY, CROSSOVER - Crossover detection
- DECAY, EDECAY - Exponential Decay
- LAG - Lag indicator
- MD - Mean Deviation
- STDERR - Standard Error