<script lang="ts">
	import { indicatorCategories } from '$lib/stores';
	
	$: categories = $indicatorCategories;
	
	// Count total indicators
	$: totalIndicators = Object.values(categories).reduce((total, indicators) => total + indicators.length, 0);
</script>

<svelte:head>
	<title>Rust-Backtester - Technical Analysis Library</title>
	<meta name="description" content="High-performance technical indicators implemented in Rust with WASM bindings for real-time financial analysis" />
</svelte:head>

<header class="header">
	<div class="header-content">
		<h1>Rust-Backtester Technical Analysis Library</h1>
		<p class="subtitle">High-performance technical indicators implemented in Rust with WASM bindings for modern financial analysis</p>
	</div>
</header>

<section class="intro">
	<div class="feature-grid">
		<div class="feature-card">
			<h3>
				<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
					<path d="M13 2L3 14H12L11 22L21 10H12L13 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
				</svg>
				High Performance
			</h3>
			<p>Built in Rust for maximum performance and memory safety. Optimized algorithms deliver blazing-fast calculations for real-time financial analysis.</p>
		</div>
		
		<div class="feature-card">
			<h3>
				<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
					<path d="M9 19C9 20.1046 7.65685 21 6 21C4.34315 21 3 20.1046 3 19C3 17.8954 4.34315 17 6 17C7.65685 17 9 17.8954 9 19Z" stroke="currentColor" stroke-width="2"/>
					<path d="M21 8C21 9.10457 19.6569 10 18 10C16.3431 10 15 9.10457 15 8C15 6.89543 16.3431 6 18 6C19.6569 6 21 6.89543 21 8Z" stroke="currentColor" stroke-width="2"/>
					<path d="M15 14L9 14" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
					<path d="M8 14L21 3" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
				</svg>
				{totalIndicators}+ Indicators
			</h3>
			<p>Comprehensive collection of technical indicators covering momentum, trend, volatility, and volume analysis with mathematical precision.</p>
		</div>
		
		<div class="feature-card">
			<h3>
				<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
					<path d="M21.54 15.15L20.24 15.89L18.94 16.63C18.1816 17.0302 17.3113 17.1793 16.46 17.06L12 16.24L7.54 17.06C6.6887 17.1793 5.8184 17.0302 5.06 16.63L3.76 15.89L2.46 15.15" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
					<path d="M21.54 8.85L20.24 9.59L18.94 10.33C18.1816 10.7302 17.3113 10.8793 16.46 10.76L12 9.94L7.54 10.76C6.6887 10.8793 5.8184 10.7302 5.06 10.33L3.76 9.59L2.46 8.85" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
					<path d="M21.54 2.55L20.24 3.29L18.94 4.03C18.1816 4.43024 17.3113 4.57929 16.46 4.46L12 3.64L7.54 4.46C6.6887 4.57929 5.8184 4.43024 5.06 4.03L3.76 3.29L2.46 2.55" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
				</svg>
				WASM Ready
			</h3>
			<p>WebAssembly bindings enable seamless integration in web applications and JavaScript environments with near-native performance.</p>
		</div>
		
		<div class="feature-card">
			<h3>
				<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
					<path d="M18 20V10M12 20V4M6 20V14" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
				</svg>
				Interactive Charts
			</h3>
			<p>Live demonstrations using TradingView Lightweight Charts with real market data examples and interactive parameter adjustment.</p>
		</div>
	</div>
</section>

<section class="getting-started">
	<h2>Getting Started</h2>
	<p>Explore the comprehensive collection of technical indicators available in this library. Each indicator page includes:</p>
	<ul>
		<li><strong>Description</strong> - Mathematical definition and trading applications</li>
		<li><strong>Parameters</strong> - Configurable inputs and their effects</li>
		<li><strong>Usage Examples</strong> - Code snippets for WASM integration</li>
		<li><strong>Interactive Chart</strong> - Live visualization with sample data</li>
	</ul>
	
	<div class="cta-buttons">
		<a href="/indicators/rsi" class="btn btn-primary">Start with RSI</a>
		<a href="/indicators/macd" class="btn btn-secondary">Explore MACD</a>
	</div>
</section>

<section class="categories">
	<h2>Indicator Categories</h2>
	<div class="category-grid">
		{#each Object.entries(categories) as [categoryKey, indicators]}
			<div class="category-card">
				<h4>
					{categoryKey === 'moving-averages' ? 'Moving Averages' : 
					 categoryKey === 'momentum' ? 'Momentum Indicators' :
					 categoryKey === 'volatility' ? 'Volatility Indicators' :
					 categoryKey === 'volume' ? 'Volume Indicators' :
					 categoryKey === 'trend' ? 'Trend Indicators' : categoryKey}
				</h4>
				<p>
					{categoryKey === 'moving-averages' ? 'Various moving average algorithms including SMA, EMA, and adaptive variants' :
					 categoryKey === 'momentum' ? 'Oscillators and momentum-based indicators for trend strength analysis' :
					 categoryKey === 'volatility' ? 'Volatility measures including Bollinger Bands, ATR, and channel indicators' :
					 categoryKey === 'volume' ? 'Volume-based analysis tools for market participation insights' :
					 categoryKey === 'trend' ? 'Directional indicators for identifying and confirming market trends' : 'Advanced analysis tools'}
				</p>
				<span class="indicator-count">{indicators.length} indicators</span>
			</div>
		{/each}
	</div>
</section>
