// Comprehensive indicator-specific descriptions and trading applications

export function getIndicatorOverview(indicatorId: string): string {
	const overviews: Record<string, string> = {
		// Moving Averages
		'sma': 'The Simple Moving Average (SMA) is the most basic form of moving average, calculated by taking the arithmetic mean of prices over a specified number of periods. It provides equal weight to all periods in the calculation, making it slower to react to price changes but highly reliable for identifying long-term trends. SMA is often used as a baseline for other technical indicators and serves as a foundation for more complex moving average systems.',
		
		'ema': 'The Exponential Moving Average (EMA) applies greater weight to recent prices, making it more responsive to current market conditions than the Simple Moving Average. The exponential smoothing factor ensures that recent price action has more influence on the average, allowing traders to identify trend changes more quickly. EMA is particularly effective in trending markets and forms the basis for many advanced technical indicators like MACD.',
		
		'wma': 'The Weighted Moving Average (WMA) assigns linearly decreasing weights to older prices, with the most recent price receiving the highest weight. This creates a more responsive average than SMA while maintaining smoother signals than EMA. WMA is particularly useful for short-term trading strategies where responsiveness to recent price action is crucial while still filtering out some market noise.',
		
		'dema': 'The Double Exponential Moving Average (DEMA) was developed by Patrick Mulloy to reduce the lag inherent in traditional moving averages. By applying exponential smoothing twice and then calculating the difference, DEMA achieves faster signal generation while maintaining smoothness. This indicator is particularly valuable for traders who need quick trend identification without excessive false signals.',
		
		'tema': 'The Triple Exponential Moving Average (TEMA) extends the DEMA concept by applying exponential smoothing three times, further reducing lag while maintaining signal quality. TEMA responds even more quickly to price changes than DEMA, making it ideal for short-term trading and scalping strategies. However, the increased responsiveness comes with slightly more sensitivity to market noise.',
		
		'hma': 'The Hull Moving Average (HMA), developed by Alan Hull, combines the benefits of weighted moving averages with square root periods to achieve minimal lag and superior smoothness. HMA virtually eliminates lag while maintaining curve smoothness, making it excellent for trend identification and generating timely entry/exit signals. The indicator is particularly effective in trending markets.',
		
		'alma': 'The Arnaud Legoux Moving Average (ALMA) uses a Gaussian filter and customizable parameters to achieve an optimal balance between responsiveness and smoothness. Unlike traditional moving averages, ALMA allows traders to adjust both the phase (lag) and bandwidth (smoothness) to match their trading style and market conditions. This flexibility makes ALMA suitable for various trading strategies and timeframes.',
		
		'vwma': 'The Volume Weighted Moving Average (VWMA) incorporates volume data into the moving average calculation, giving more weight to periods with higher volume. This approach reflects the market\'s true price interest, as high-volume periods are considered more significant. VWMA is particularly useful for confirming price movements and identifying when volume supports or contradicts price action.',
		
		'vwap': 'Volume Weighted Average Price (VWAP) is a trading benchmark that represents the average price at which a security has traded throughout the day, weighted by volume. Institutional traders often use VWAP as an execution benchmark, and retail traders use it to identify intraday support/resistance levels and gauge whether they\'re buying or selling at fair value.',
		
		// Momentum Indicators  
		'rsi': 'The Relative Strength Index (RSI), developed by J. Welles Wilder Jr., is one of the most widely used momentum oscillators in technical analysis. RSI measures the speed and magnitude of price changes, oscillating between 0 and 100. It helps identify overbought conditions (typically above 70) and oversold conditions (typically below 30), while also revealing momentum divergences that often precede price reversals.',
		
		'macd': 'The Moving Average Convergence Divergence (MACD), created by Gerald Appel, is a trend-following momentum indicator that shows the relationship between two moving averages of prices. MACD consists of three components: the MACD line (12-EMA minus 26-EMA), the signal line (9-EMA of MACD), and the histogram (MACD minus signal line). This versatile indicator provides signals for trend direction, momentum changes, and potential entry/exit points.',
		
		'stoch': 'The Stochastic Oscillator, developed by George Lane, compares a security\'s closing price to its price range over a specific period. This momentum indicator operates on the premise that closing prices tend to close near the high in uptrends and near the low in downtrends. The Stochastic consists of %K (fast stochastic) and %D (slow stochastic), helping identify overbought/oversold conditions and potential reversal points.',
		
		'stochf': 'Fast Stochastic is the original version of the Stochastic Oscillator without additional smoothing. It provides more sensitive signals than the regular Stochastic, responding quickly to price changes but generating more false signals. Fast Stochastic is preferred by traders who prioritize early signal generation and are comfortable managing the increased noise in their analysis.',
		
		'cci': 'The Commodity Channel Index (CCI), developed by Donald Lambert, measures the current price level relative to an average price level over a given period. Originally designed for commodity trading, CCI is now widely used across all markets. It identifies cyclical turns in commodities and can signal when an instrument is overbought (typically above +100) or oversold (typically below -100).',
		
		'willr': 'Williams %R, developed by Larry Williams, is a momentum indicator that measures overbought and oversold levels. Similar to Stochastic but inverted, Williams %R oscillates between 0 and -100, with readings above -20 considered overbought and below -80 considered oversold. This fast-moving oscillator is particularly effective for timing short-term entries and exits.',
		
		'roc': 'Rate of Change (ROC) is a momentum oscillator that measures the percentage change in price from one period to the next. ROC oscillates around zero, with positive values indicating upward price momentum and negative values indicating downward momentum. This simple yet effective indicator helps identify acceleration and deceleration in price trends.',
		
		'mom': 'Momentum is one of the simplest momentum indicators, calculated as the difference between the current price and the price n periods ago. Unlike ROC which shows percentage change, Momentum shows absolute change. It oscillates around zero and helps identify the strength of price movement and potential trend changes when momentum diverges from price.',
		
		// Volatility Indicators
		'atr': 'Average True Range (ATR), developed by J. Welles Wilder Jr., measures market volatility by calculating the average of true ranges over a specified period. ATR doesn\'t predict price direction but quantifies volatility, making it invaluable for position sizing, setting stop-losses, and identifying periods of high or low volatility. Higher ATR values indicate more volatile conditions.',
		
		'bollinger_bands': 'Bollinger Bands, created by John Bollinger, consist of a middle line (usually a 20-period SMA) and two outer bands placed at standard deviations above and below the middle line. These bands expand and contract based on market volatility, providing dynamic support and resistance levels. Bollinger Bands help identify overbought/oversold conditions and potential breakout scenarios.',
		
		'donchian': 'Donchian Channels, developed by Richard Donchian, consist of upper and lower bands based on the highest high and lowest low over a specified period. The middle line represents the average of these extremes. Donchian Channels help identify breakouts, support/resistance levels, and trend direction. Many trend-following systems use Donchian Channel breakouts as entry signals.',
		
		'keltner': 'Keltner Channels, developed by Chester Keltner and later modified by Linda Raschke, consist of an exponential moving average in the center with upper and lower bands based on Average True Range (ATR). These channels adapt to volatility changes and provide dynamic support and resistance levels, making them excellent for trend identification and breakout trading.',
		
		'stddev': 'Standard Deviation measures the dispersion of prices from their moving average, quantifying volatility. Higher standard deviation values indicate greater price volatility, while lower values suggest more stable price action. This statistical measure is fundamental to many technical indicators, including Bollinger Bands, and helps traders assess risk and market conditions.',
		
		// Volume Indicators
		'obv': 'On-Balance Volume (OBV), developed by Joe Granville, is a momentum indicator that uses volume flow to predict changes in stock price. OBV adds volume on up days and subtracts volume on down days, creating a running total that should confirm price trends. Divergences between OBV and price often signal potential reversals, making it a powerful confirmation tool.',
		
		'mfi': 'The Money Flow Index (MFI), sometimes called the "volume-weighted RSI," combines price and volume to measure buying and selling pressure. MFI oscillates between 0 and 100, with readings above 80 indicating overbought conditions and below 20 indicating oversold conditions. The incorporation of volume makes MFI particularly effective for confirming price movements.',
		
		'ad': 'The Accumulation/Distribution Line (A/D Line) measures the cumulative flow of money into and out of a security. It considers both price and volume, with the calculation based on where the closing price falls within the high-low range. The A/D Line helps confirm trends and identify potential reversals through divergences with price action.',
		
		// Trend Indicators
		'adx': 'The Average Directional Index (ADX), developed by J. Welles Wilder Jr., measures the strength of a trend without indicating its direction. ADX values range from 0 to 100, with readings above 25 typically indicating a strong trend and below 20 suggesting a weak or sideways market. ADX is often used with the Directional Movement Indicators (+DI and -DI) for complete trend analysis.',
		
		'sar': 'The Parabolic SAR (Stop and Reverse), developed by J. Welles Wilder Jr., is a trend-following indicator that provides potential entry and exit points. The SAR appears as dots placed above or below price bars, with the position indicating the current trend direction. When price crosses the SAR, it signals a potential trend reversal and provides new stop-loss levels.',
		
		'supertrend': 'SuperTrend is a trend-following indicator that uses Average True Range (ATR) to calculate dynamic support and resistance levels. The indicator plots above price in downtrends (bearish) and below price in uptrends (bullish). SuperTrend provides clear trend direction signals and dynamic stop-loss levels, making it popular among trend traders.',
		
		'aroon': 'The Aroon indicator, developed by Tushar Chande, identifies trend changes and measures trend strength. It consists of Aroon Up and Aroon Down lines that measure how long it has been since the highest high and lowest low within a specified period. Aroon helps determine if a market is trending or trading sideways and can signal trend reversals.',
		
		// Default for indicators without specific descriptions
		'default': 'This technical indicator provides valuable insights for market analysis through mathematical calculations applied to price and/or volume data. Each indicator serves specific purposes in technical analysis, helping traders and analysts identify market conditions, trends, and potential trading opportunities based on historical price patterns and market behavior.'
	};

	return overviews[indicatorId] || overviews['default'];
}

export function getIndicatorApplications(indicatorId: string): string {
	const applications: Record<string, string> = {
		// Moving Averages
		'sma': 'SMA serves multiple purposes in trading strategies: (1) Trend identification - prices above SMA indicate uptrend, below indicate downtrend; (2) Support/resistance levels - SMA often acts as dynamic support in uptrends and resistance in downtrends; (3) Moving average crossovers - shorter period SMA crossing above longer period signals potential uptrend; (4) Price mean reversion - when price deviates significantly from SMA, it often returns to the average; (5) Filtering market noise in ranging markets.',
		
		'ema': 'EMA applications include: (1) Early trend identification due to reduced lag compared to SMA; (2) Dynamic support and resistance in trending markets; (3) Entry signals when price bounces off EMA in established trends; (4) Exit signals when price breaks through EMA against the trend; (5) Multiple EMA systems (8, 21, 55) for trend confirmation; (6) Base calculation for MACD and other advanced indicators; (7) Adaptive stop-loss placement in trending positions.',
		
		'wma': 'WMA trading applications: (1) Short-term trend identification with faster response than SMA; (2) Breakout confirmation when price moves decisively above/below WMA; (3) Momentum gauge - steep WMA slope indicates strong momentum; (4) Support/resistance in active markets; (5) Entry timing when price retraces to WMA in trends; (6) Filtering false signals in choppy markets; (7) Crossover systems with different periods for trend changes.',
		
		'dema': 'DEMA strategies focus on: (1) Early trend change detection with minimal lag; (2) Momentum trading when price breaks above/below DEMA; (3) Pullback entries in strong trends when price touches DEMA; (4) Dynamic stop-loss placement for trend-following positions; (5) Crossover signals with other moving averages; (6) Trend strength assessment through DEMA slope analysis; (7) Support/resistance identification in volatile markets.',
		
		'tema': 'TEMA applications include: (1) Ultra-responsive trend following for short-term traders; (2) Scalping strategies using TEMA crossovers; (3) Momentum breakout confirmation; (4) Quick reversal detection in fast-moving markets; (5) Dynamic support/resistance for day trading; (6) Trend continuation signals when price stays above/below TEMA; (7) Exit strategies for protecting profits in trending moves.',
		
		'hma': 'HMA trading uses: (1) Precise trend direction identification with minimal lag; (2) Color-coded trend signals (rising HMA = bullish, falling = bearish); (3) Support/resistance levels that adapt quickly to price changes; (4) Entry signals on HMA direction changes; (5) Momentum confirmation through HMA slope steepness; (6) Swing trading strategies using HMA for trend timing; (7) Stop-loss placement for optimal risk management.',
		
		'alma': 'ALMA strategies leverage: (1) Customizable responsiveness through phase and sigma adjustments; (2) Trend following with reduced noise through Gaussian smoothing; (3) Support/resistance levels adapted to market volatility; (4) Entry signals when price crosses ALMA with momentum; (5) Trend strength measurement through ALMA angle; (6) Multi-timeframe analysis using different ALMA settings; (7) Risk management through ALMA-based stop levels.',
		
		'vwma': 'VWMA applications: (1) Volume-confirmed trend identification - VWMA movement supported by volume is more reliable; (2) Breakout validation using volume-weighted price levels; (3) Support/resistance levels that reflect volume-based price acceptance; (4) Divergence analysis between price and volume-weighted movements; (5) Institutional level identification where high volume occurred; (6) Trend continuation confirmation through volume-price alignment.',
		
		'vwap': 'VWAP trading strategies: (1) Intraday fair value assessment - prices above VWAP suggest institutional buying interest; (2) Mean reversion trades when price deviates significantly from VWAP; (3) Support/resistance identification, especially during institutional trading hours; (4) Execution benchmarking for large orders; (5) Trend bias determination - sustained trading above/below VWAP indicates institutional positioning; (6) End-of-day settlement reference for institutions.',
		
		// Momentum Indicators
		'rsi': 'RSI trading applications: (1) Overbought/oversold identification - RSI > 70 suggests overbought conditions, RSI < 30 suggests oversold; (2) Divergence analysis - price making new highs/lows while RSI doesn\'t confirms potential reversals; (3) Trend strength assessment - RSI remaining above 40 in uptrends, below 60 in downtrends; (4) Support/resistance levels at key RSI levels (30, 50, 70); (5) Momentum confirmation for breakouts; (6) Counter-trend entries in ranging markets; (7) Trend continuation signals when RSI bounces from 40/60 levels.',
		
		'macd': 'MACD strategies include: (1) Signal line crossovers - MACD crossing above signal line suggests bullish momentum; (2) Zero line crossovers for trend direction changes; (3) Histogram analysis for momentum acceleration/deceleration; (4) Divergence identification between MACD and price for reversal signals; (5) Trend following using MACD direction; (6) Entry timing when MACD confirms price breakouts; (7) Exit signals when MACD momentum weakens.',
		
		'stoch': 'Stochastic applications: (1) Overbought (>80) and oversold (<20) condition identification; (2) %K and %D crossovers for entry/exit signals; (3) Divergence analysis for trend reversal confirmation; (4) Ranging market oscillations between support/resistance; (5) Trend following when Stochastic remains in upper/lower half; (6) Momentum confirmation for breakout trades; (7) Multiple timeframe analysis for signal confirmation.',
		
		'stochf': 'Fast Stochastic uses: (1) Quick overbought/oversold signals for active trading; (2) Early reversal warnings in trending markets; (3) Scalping strategies using rapid %K movements; (4) Momentum confirmation for short-term breakouts; (5) Counter-trend entries with tight stop-losses; (6) High-frequency trading signal generation; (7) Market turning point identification with faster response.',
		
		'cci': 'CCI trading strategies: (1) Overbought (+100) and oversold (-100) level trading; (2) Zero line crossovers for trend direction changes; (3) Extreme readings (±200) for high-probability reversal setups; (4) Divergence analysis for trend weakness identification; (5) Trend following when CCI remains above/below 100; (6) Cycle identification in commodity markets; (7) Momentum confirmation for breakout trades.',
		
		'willr': 'Williams %R applications: (1) Overbought (-20 to 0) and oversold (-80 to -100) identification; (2) Fast-moving signals for short-term trading; (3) Reversal confirmation at extreme levels; (4) Momentum assessment for trend strength; (5) Entry timing in established trends; (6) Exit signals when momentum shifts; (7) Multiple timeframe confirmation for trade entries.',
		
		'roc': 'ROC trading uses: (1) Momentum direction assessment - positive ROC indicates upward momentum; (2) Zero line crossovers for trend change signals; (3) Divergence analysis with price for reversal identification; (4) Momentum extremes for overbought/oversold conditions; (5) Trend acceleration/deceleration measurement; (6) Breakout momentum confirmation; (7) Comparative strength analysis between securities.',
		
		'mom': 'Momentum indicator applications: (1) Trend strength measurement through momentum magnitude; (2) Zero line crossovers for directional changes; (3) Momentum divergences with price for reversal signals; (4) Acceleration/deceleration identification in trends; (5) Breakout validation through momentum confirmation; (6) Support/resistance at momentum levels; (7) Market phase identification (accumulation, markup, distribution).',
		
		// Volatility Indicators
		'atr': 'ATR applications: (1) Position sizing based on volatility - higher ATR suggests smaller position sizes; (2) Stop-loss placement using ATR multiples (1.5-3x ATR); (3) Profit target setting using volatility-based expectations; (4) Breakout filtering - significant moves should exceed average volatility; (5) Market condition assessment - high ATR indicates increased uncertainty; (6) Entry timing - entering during lower volatility periods; (7) Risk management through volatility-adjusted strategies.',
		
		'bollinger_bands': 'Bollinger Bands strategies: (1) Mean reversion trading when price touches outer bands; (2) Breakout trading when price closes outside bands with volume; (3) Squeeze identification when bands contract, preceding major moves; (4) Band walk patterns in strong trends (price hugging upper/lower band); (5) Support/resistance from middle band (20 SMA); (6) Volatility assessment through band width; (7) Divergence analysis between price and band position.',
		
		'donchian': 'Donchian Channel applications: (1) Breakout trading - price breaking above upper channel signals bullish breakout; (2) Trend following using channel direction; (3) Support/resistance identification from channel boundaries; (4) Range trading between upper and lower channels; (5) Volatility measurement through channel width; (6) Trend strength assessment via channel slope; (7) Risk management using channel-based stops.',
		
		'keltner': 'Keltner Channel strategies: (1) Trend identification through price position relative to channels; (2) Breakout confirmation using volatility-adjusted levels; (3) Mean reversion trades from channel extremes; (4) Trend following when price remains near one channel; (5) Support/resistance from adaptive channel levels; (6) Volatility assessment through channel width changes; (7) Entry timing using channel bounces.',
		
		'stddev': 'Standard Deviation uses: (1) Volatility measurement for risk assessment; (2) Position sizing adjustments based on price dispersion; (3) Breakout filtering - moves should exceed normal volatility; (4) Market regime identification (high/low volatility periods); (5) Option trading strategies based on volatility levels; (6) Stop-loss placement using volatility multiples; (7) Trend strength assessment through volatility patterns.',
		
		// Volume Indicators  
		'obv': 'OBV trading applications: (1) Trend confirmation - OBV should move with price in healthy trends; (2) Divergence identification - OBV diverging from price suggests trend weakness; (3) Breakout validation using volume accumulation; (4) Support/resistance levels where OBV previously reversed; (5) Accumulation/distribution phase identification; (6) Institutional activity detection through volume patterns; (7) Trend continuation confirmation through OBV momentum.',
		
		'mfi': 'MFI strategies include: (1) Overbought (>80) and oversold (<20) identification with volume confirmation; (2) Divergence analysis combining price and volume momentum; (3) Volume-weighted momentum assessment; (4) Reversal signals at extreme MFI levels; (5) Trend strength measurement through sustained MFI direction; (6) Breakout confirmation using money flow acceleration; (7) Market sentiment gauge through volume-price relationship.',
		
		'ad': 'Accumulation/Distribution applications: (1) Trend confirmation through volume-weighted price action; (2) Distribution detection when A/D Line diverges from rising prices; (3) Accumulation identification during price consolidation with rising A/D; (4) Support/resistance levels where A/D Line previously turned; (5) Breakout validation using volume accumulation patterns; (6) Institutional activity tracking through sustained A/D movements; (7) Market phase identification (accumulation, markup, distribution, decline).',
		
		// Trend Indicators
		'adx': 'ADX trading strategies: (1) Trend strength identification - ADX > 25 indicates strong trend; (2) Ranging market detection when ADX < 20; (3) Trend development monitoring through rising ADX; (4) Trend exhaustion signals when ADX peaks and declines; (5) Directional bias using +DI and -DI crossovers; (6) Breakout confirmation through ADX acceleration; (7) Market condition assessment for strategy selection.',
		
		'sar': 'Parabolic SAR applications: (1) Trend direction identification through SAR position; (2) Stop-loss placement using SAR levels; (3) Entry signals when price crosses SAR; (4) Trend following with automatic stop adjustments; (5) Exit timing for trend-following positions; (6) Trend acceleration measurement through SAR step increases; (7) Risk management through trailing stops.',
		
		'supertrend': 'SuperTrend strategies: (1) Clear trend direction signals through color changes; (2) Support/resistance identification from SuperTrend levels; (3) Entry signals when price breaks above/below SuperTrend; (4) Stop-loss placement using SuperTrend line; (5) Trend continuation confirmation; (6) Exit signals when SuperTrend changes direction; (7) Multiple timeframe trend alignment.',
		
		'aroon': 'Aroon applications: (1) Trend identification when Aroon Up > Aroon Down (bullish) or vice versa; (2) Trending vs. ranging market detection through Aroon level analysis; (3) Trend strength measurement using Aroon extremes (near 100); (4) Consolidation identification when both Aroons remain low; (5) Breakout anticipation when Aroons begin rising; (6) Trend reversal signals through Aroon crossovers; (7) Market condition assessment for strategy selection.',
		
		// Default
		'default': 'This indicator can be applied in various trading contexts including trend identification, momentum analysis, support and resistance detection, entry and exit timing, risk management, and market condition assessment. Specific applications depend on the indicator\'s mathematical properties and the trader\'s strategy framework.'
	};

	return applications[indicatorId] || applications['default'];
}

export function getIndicatorCharacteristics(indicatorId: string, isOscillator: boolean, category: string): Array<{label: string, value: string}> {
	const characteristics: Record<string, Array<{label: string, value: string}>> = {
		// Moving Averages
		'sma': [
			{label: 'Type', value: 'Trend-following moving average'},
			{label: 'Calculation', value: 'Arithmetic mean of closing prices'},
			{label: 'Responsiveness', value: 'Slow (equal weight to all periods)'},
			{label: 'Smoothness', value: 'High (filters noise effectively)'},
			{label: 'Lag', value: 'High (n/2 periods)'},
			{label: 'Best for', value: 'Long-term trend identification, support/resistance'},
			{label: 'Timeframes', value: 'Daily, weekly, monthly charts'},
			{label: 'Common periods', value: '20, 50, 100, 200'}
		],
		
		'ema': [
			{label: 'Type', value: 'Exponentially weighted moving average'},
			{label: 'Calculation', value: 'Exponential smoothing with decay factor'},
			{label: 'Responsiveness', value: 'Medium-High (recent prices weighted more)'},
			{label: 'Smoothness', value: 'Medium (good balance)'},
			{label: 'Lag', value: 'Medium (approximately (n-1)/2 periods)'},
			{label: 'Best for', value: 'Trend following, dynamic support/resistance'},
			{label: 'Timeframes', value: 'All timeframes, especially intraday'},
			{label: 'Common periods', value: '8, 12, 21, 26, 50'}
		],
		
		'rsi': [
			{label: 'Type', value: 'Bounded momentum oscillator'},
			{label: 'Range', value: '0 to 100'},
			{label: 'Overbought threshold', value: '70 (adjustable to 80 in strong trends)'},
			{label: 'Oversold threshold', value: '30 (adjustable to 20 in strong trends)'},
			{label: 'Calculation period', value: '14 periods (standard)'},
			{label: 'Smoothing', value: 'Wilder\'s smoothing method'},
			{label: 'Best for', value: 'Mean reversion, divergence analysis'},
			{label: 'Market conditions', value: 'Range-bound and trending markets'}
		],
		
		'macd': [
			{label: 'Type', value: 'Trend-following momentum indicator'},
			{label: 'Components', value: 'MACD line, Signal line, Histogram'},
			{label: 'Default periods', value: '12, 26, 9 (Fast EMA, Slow EMA, Signal EMA)'},
			{label: 'Range', value: 'Unbounded (oscillates around zero)'},
			{label: 'Signal types', value: 'Line crossovers, zero crossovers, divergences'},
			{label: 'Lag', value: 'Medium (due to moving average basis)'},
			{label: 'Best for', value: 'Trend identification, momentum changes'},
			{label: 'Market conditions', value: 'Trending markets, trend reversals'}
		],
		
		'atr': [
			{label: 'Type', value: 'Volatility measurement indicator'},
			{label: 'Range', value: 'Always positive (measures magnitude)'},
			{label: 'Calculation', value: 'Average of True Range over n periods'},
			{label: 'Unit', value: 'Same as underlying instrument (points, pips, etc.)'},
			{label: 'Smoothing', value: 'Wilder\'s smoothing method'},
			{label: 'Standard period', value: '14 periods'},
			{label: 'Best for', value: 'Position sizing, stop-loss placement, breakout filtering'},
			{label: 'Interpretation', value: 'Higher values = higher volatility'}
		],
		
		'bollinger_bands': [
			{label: 'Type', value: 'Volatility-based channel indicator'},
			{label: 'Components', value: 'Middle band (SMA), Upper band (+2σ), Lower band (-2σ)'},
			{label: 'Standard settings', value: '20-period SMA, 2 standard deviations'},
			{label: 'Adaptation', value: 'Bands expand/contract with volatility'},
			{label: 'Statistical basis', value: '~95% of price action occurs within bands'},
			{label: 'Signals', value: 'Band touches, squeezes, breakouts, walks'},
			{label: 'Best for', value: 'Mean reversion, volatility assessment, breakouts'},
			{label: 'Market conditions', value: 'All conditions, especially ranging markets'}
		],
		
		'stoch': [
			{label: 'Type', value: 'Bounded momentum oscillator'},
			{label: 'Range', value: '0 to 100'},
			{label: 'Components', value: '%K (fast line), %D (slow line)'},
			{label: 'Overbought threshold', value: '80'},
			{label: 'Oversold threshold', value: '20'},
			{label: 'Standard periods', value: '14 for %K, 3 for %D smoothing'},
			{label: 'Principle', value: 'Closing price position within recent range'},
			{label: 'Best for', value: 'Overbought/oversold, momentum shifts, divergences'}
		],
		
		'adx': [
			{label: 'Type', value: 'Trend strength indicator'},
			{label: 'Range', value: '0 to 100'},
			{label: 'Components', value: 'ADX, +DI (bullish DI), -DI (bearish DI)'},
			{label: 'Strong trend threshold', value: 'ADX > 25'},
			{label: 'Weak trend threshold', value: 'ADX < 20'},
			{label: 'Direction indication', value: 'None (only strength)'},
			{label: 'Standard period', value: '14 periods'},
			{label: 'Best for', value: 'Trend strength assessment, market condition identification'}
		],
		
		'obv': [
			{label: 'Type', value: 'Volume-momentum indicator'},
			{label: 'Range', value: 'Unbounded cumulative line'},
			{label: 'Calculation', value: 'Cumulative volume based on price direction'},
			{label: 'Volume treatment', value: 'Added on up days, subtracted on down days'},
			{label: 'Trend confirmation', value: 'Should move with price in healthy trends'},
			{label: 'Divergence signals', value: 'Price vs. OBV divergences signal reversals'},
			{label: 'Best for', value: 'Trend confirmation, accumulation/distribution analysis'},
			{label: 'Time sensitivity', value: 'Works on all timeframes'}
		]
	};

	// Default characteristics based on category
	const defaultCharacteristics: Record<string, Array<{label: string, value: string}>> = {
		'moving-averages': [
			{label: 'Type', value: 'Trend-following indicator'},
			{label: 'Calculation', value: 'Price-based moving average'},
			{label: 'Signals', value: 'Trend direction, support/resistance'},
			{label: 'Best for', value: 'Trend identification and following'}
		],
		'momentum': [
			{label: 'Type', value: 'Momentum oscillator'},
			{label: 'Signals', value: 'Overbought/oversold, divergences'},
			{label: 'Range', value: isOscillator ? 'Bounded oscillator' : 'Unbounded momentum'},
			{label: 'Best for', value: 'Momentum analysis and timing'}
		],
		'volatility': [
			{label: 'Type', value: 'Volatility measurement'},
			{label: 'Values', value: 'Always positive (measures magnitude)'},
			{label: 'Signals', value: 'Market volatility, breakout potential'},
			{label: 'Best for', value: 'Risk assessment and position sizing'}
		],
		'volume': [
			{label: 'Type', value: 'Volume-based analysis'},
			{label: 'Data', value: 'Combines price and volume'},
			{label: 'Signals', value: 'Money flow, accumulation/distribution'},
			{label: 'Best for', value: 'Confirming price movements'}
		],
		'trend': [
			{label: 'Type', value: 'Trend analysis tool'},
			{label: 'Signals', value: 'Trend direction and strength'},
			{label: 'Application', value: 'Trend identification and confirmation'},
			{label: 'Best for', value: 'Determining market direction'}
		],
		'statistical': [
			{label: 'Type', value: 'Statistical measure'},
			{label: 'Calculation', value: 'Mathematical analysis of price data'},
			{label: 'Application', value: 'Statistical analysis and normalization'},
			{label: 'Best for', value: 'Quantitative analysis and modeling'}
		]
	};

	return characteristics[indicatorId] || defaultCharacteristics[category] || defaultCharacteristics['momentum'];
}