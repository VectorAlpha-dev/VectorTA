<script lang="ts">
	import { page } from '$app/stores';
	import { indicatorCategories, generateSampleData, type IndicatorInfo } from '$lib/stores';
	import SimpleChart from '$lib/components/SimpleChart.svelte';
	import { onMount } from 'svelte';
	import type { LineData } from 'lightweight-charts';
	import { 
		getIndicatorFormula, 
		getUsageExample, 
		isOscillator, 
		isMovingAverage, 
		isVolatilityIndicator, 
		isVolumeIndicator,
		isTrendIndicator,
		isStatisticalIndicator
	} from '$lib/indicatorFormulas';
	import { getPythonExample, getPythonInstallationExample, getPythonPandasExample } from '$lib/pythonExamples';
	import { getIndicatorOverview, getIndicatorApplications, getIndicatorCharacteristics } from '$lib/indicatorDescriptions';
	
	$: slug = $page.params.slug;
	$: indicator = findIndicator(slug);
	$: sampleData = generateSampleData(100);
	
	let indicatorData: LineData[] = [];
	let parameters: Record<string, any> = {};
	
	function findIndicator(slug: string): IndicatorInfo | null {
		for (const indicators of Object.values($indicatorCategories)) {
			const found = indicators.find(ind => ind.id === slug);
			if (found) return found;
		}
		return null;
	}
	
	function generateMockIndicatorData(indicatorId: string): LineData[] {
		const data: LineData[] = [];
		const startIndex = getStartIndex(indicatorId);
		
		for (let i = startIndex; i < sampleData.length; i++) {
			let value: number;
			
			// Generate realistic values based on indicator type
			if (isOscillator(indicatorId)) {
				// Oscillators have specific ranges
				if (['rsi', 'stoch', 'stochf', 'mfi', 'srsi', 'ift_rsi', 'rsx', 'lrsi'].includes(indicatorId)) {
					value = Math.random() * 100; // 0-100 range
				} else if (indicatorId === 'willr') {
					value = -Math.random() * 100; // -100 to 0
				} else if (indicatorId === 'cci') {
					value = (Math.random() - 0.5) * 400; // -200 to +200
				} else {
					value = (Math.random() - 0.5) * 4; // Centered around 0
				}
			} else if (isMovingAverage(indicatorId)) {
				// Moving averages close to price
				value = sampleData[i].close * (0.98 + Math.random() * 0.04);
			} else if (isVolatilityIndicator(indicatorId)) {
				// Volatility indicators - positive values
				if (indicatorId === 'bollinger_bands') {
					value = sampleData[i].close; // Middle band
				} else if (['atr', 'natr', 'stddev'].includes(indicatorId)) {
					value = Math.random() * 1000 + 100;
				} else {
					value = Math.random() * 50 + 10;
				}
			} else if (isVolumeIndicator(indicatorId)) {
				// Volume indicators
				if (['obv', 'ad', 'vpt'].includes(indicatorId)) {
					value = (Math.random() - 0.5) * 10000000;
				} else if (['mfi'].includes(indicatorId)) {
					value = Math.random() * 100;
				} else {
					value = (Math.random() - 0.5) * 1000;
				}
			} else if (isTrendIndicator(indicatorId)) {
				// Trend indicators
				if (['adx', 'adxr', 'dx'].includes(indicatorId)) {
					value = Math.random() * 60 + 10; // 10-70 range
				} else if (['sar', 'supertrend'].includes(indicatorId)) {
					value = sampleData[i].close * (0.95 + Math.random() * 0.1);
				} else {
					value = sampleData[i].close * (0.98 + Math.random() * 0.04);
				}
			} else if (isStatisticalIndicator(indicatorId)) {
				// Statistical indicators
				value = sampleData[i].close * (0.99 + Math.random() * 0.02);
			} else {
				// Default case
				value = Math.random() * 100;
			}
			
			data.push({
				time: sampleData[i].time,
				value: parseFloat(value.toFixed(2))
			});
		}
		
		return data;
	}
	
	function getStartIndex(indicatorId: string): number {
		// Different indicators need different warm-up periods
		switch (indicatorId) {
			case 'rsi': return 14;
			case 'macd': return 26;
			case 'sma': case 'ema': case 'wma': return parameters.period || 20;
			case 'bollinger_bands': return 20;
			case 'atr': return 14;
			case 'stoch': return 14;
			case 'adx': return 14;
			default: return 1;
		}
	}
	
	function initializeParameters() {
		if (indicator) {
			parameters = {};
			indicator.parameters.forEach(param => {
				parameters[param.name] = param.default;
			});
		}
	}
	
	function updateIndicatorData() {
		if (indicator) {
			indicatorData = generateMockIndicatorData(indicator.id);
		}
	}
	
	
	onMount(() => {
		initializeParameters();
		updateIndicatorData();
	});
	
	$: if (indicator) {
		initializeParameters();
		updateIndicatorData();
	}
	
	$: isOscillatorType = indicator && isOscillator(indicator.id);
	$: showPrice = indicator && !isOscillatorType;
</script>

<svelte:head>
	{#if indicator}
		<title>{indicator.name} | Rust-Backtester</title>
		<meta name="description" content="{indicator.description}" />
	{:else}
		<title>Indicator Not Found | Rust-Backtester</title>
	{/if}
</svelte:head>

{#if indicator}
	<header class="header">
		<div class="header-content">
			<h1>{indicator.name}</h1>
			<p class="subtitle">{indicator.description}</p>
		</div>
	</header>

	<section class="intro">
		<div class="description-content">
			<h2>Overview</h2>
			<p class="description-text">{getIndicatorOverview(indicator.id)}</p>
			
			<h3>Trading Applications</h3>
			<p class="description-text">{getIndicatorApplications(indicator.id)}</p>

			<h3>Key Characteristics</h3>
			<ul class="characteristics-list">
				{#each getIndicatorCharacteristics(indicator.id, isOscillatorType, indicator.category) as characteristic}
					<li><strong>{characteristic.label}:</strong> {characteristic.value}</li>
				{/each}
			</ul>
		</div>
	</section>

	<section class="getting-started">
		<h2>Formula</h2>
		<div class="code-block">
			<pre>{getIndicatorFormula(indicator.id)}</pre>
		</div>

		{#if indicator.parameters.length > 0}
			<h2>Parameters</h2>
			<table class="param-table">
				<thead>
					<tr>
						<th>Parameter</th>
						<th>Type</th>
						<th>Default</th>
						<th>Description</th>
					</tr>
				</thead>
				<tbody>
					{#each indicator.parameters as param}
						<tr>
							<td>{param.name}</td>
							<td>{param.type}</td>
							<td>{param.default}</td>
							<td>{param.description}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{/if}

		<h2>WASM Usage Example</h2>
		<div class="code-block">
			<pre>{getUsageExample(indicator.id)}</pre>
		</div>

		<h2>Python Bindings</h2>
		<p>The Rust-Backtester library provides high-performance Python bindings for seamless integration with data science workflows. These bindings offer the same performance as the native Rust implementation while providing a familiar Python interface.</p>
		
		<h3>Installation</h3>
		<div class="code-block">
			<pre>{getPythonInstallationExample()}</pre>
		</div>

		<h3>Basic Usage</h3>
		<div class="code-block">
			<pre>{getPythonExample(indicator.id)}</pre>
		</div>

		<h3>Integration with Pandas</h3>
		<div class="code-block">
			<pre>{getPythonPandasExample(indicator.id)}</pre>
		</div>

		<div class="python-features">
			<h3>Python Binding Features</h3>
			<ul>
				<li><strong>High Performance:</strong> Near-native Rust performance with Python convenience</li>
				<li><strong>NumPy Integration:</strong> Seamless compatibility with NumPy arrays</li>
				<li><strong>Pandas Support:</strong> Easy integration with pandas DataFrames</li>
				<li><strong>Type Safety:</strong> Comprehensive type hints for better IDE support</li>
				<li><strong>Memory Efficiency:</strong> Zero-copy operations where possible</li>
				<li><strong>Error Handling:</strong> Descriptive error messages for debugging</li>
			</ul>
		</div>
	</section>

	<section class="categories">
		<h2>Interactive Charts</h2>
		<p class="chart-description">
			Interactive chart visualization for {indicator.name} with clean, professional interface. 
			The chart displays sample data to demonstrate the indicator's behavior and characteristics.
		</p>
		
		<div class="chart-wrapper">
			<SimpleChart 
				indicatorId={indicator.id}
				height={500}
			/>
		</div>
		
		<div class="feature-grid" style="margin-top: 2rem;">
			{#if indicator.id === 'rsi'}
				<div class="feature-card">
					<h3>🔴 Overbought Zone</h3>
					<p>RSI values above 70 typically indicate overbought conditions. Consider taking profits or looking for short opportunities.</p>
				</div>
				
				<div class="feature-card">
					<h3>🟢 Oversold Zone</h3>
					<p>RSI values below 30 typically indicate oversold conditions. Consider buying opportunities or covering short positions.</p>
				</div>
				
				<div class="feature-card">
					<h3>📊 Neutral Zone</h3>
					<p>RSI values between 30-70 indicate normal market conditions. Look for trend continuation or other confirmation signals.</p>
				</div>
				
				<div class="feature-card">
					<h3>⚠️ Divergences</h3>
					<p>When price makes new highs/lows but RSI doesn't confirm, it may signal potential trend reversal.</p>
				</div>
			{:else if indicator.id === 'macd'}
				<div class="feature-card">
					<h3>📈 Signal Line Crossover</h3>
					<p>When MACD line crosses above signal line, it may indicate bullish momentum. Below indicates bearish momentum.</p>
				</div>
				
				<div class="feature-card">
					<h3>📊 Zero Line Cross</h3>
					<p>MACD crossing above zero suggests upward momentum, while crossing below suggests downward momentum.</p>
				</div>
				
				<div class="feature-card">
					<h3>📉 Histogram</h3>
					<p>The histogram shows the difference between MACD and signal lines, indicating momentum strength.</p>
				</div>
				
				<div class="feature-card">
					<h3>⚠️ Divergences</h3>
					<p>Divergences between price and MACD can signal potential trend reversals or momentum shifts.</p>
				</div>
			{:else}
				<div class="feature-card">
					<h3>📊 Interpretation</h3>
					<p>Analyze the indicator values in conjunction with price action for optimal trading signals.</p>
				</div>
				
				<div class="feature-card">
					<h3>📈 Trend Analysis</h3>
					<p>Use this indicator to confirm trend direction and identify potential entry/exit points.</p>
				</div>
				
				<div class="feature-card">
					<h3>⚠️ Risk Management</h3>
					<p>Always combine indicator signals with proper risk management and position sizing strategies.</p>
				</div>
				
				<div class="feature-card">
					<h3>🎯 Best Practices</h3>
					<p>Use multiple timeframes and confirmation signals for more reliable trading decisions.</p>
				</div>
			{/if}
		</div>
	</section>
{:else}
	<header class="header">
		<h1>Indicator Not Found</h1>
		<p class="subtitle">The requested indicator "{slug}" could not be found.</p>
	</header>

	<section class="intro">
		<div class="feature-card">
			<h3>Available Indicators</h3>
			<p>Please check the sidebar for a list of available indicators or return to the home page.</p>
			<div class="cta-buttons" style="margin-top: 1rem;">
				<a href="/" class="btn btn-primary">Go Home</a>
			</div>
		</div>
	</section>
{/if}