<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	
	export let indicatorId: string = '';
	export let height: number = 500;

	let chartContainer: HTMLDivElement;
	let canvas: HTMLCanvasElement;
	let loading = true;
	let error = '';

	// Chart theme
	let chartTheme: 'light' | 'dark' = 'dark';

	// Chart data
	let chartData: Array<{time: string, value: number}> = [];

	function generateSampleData() {
		const data = [];
		const startDate = new Date();
		startDate.setDate(startDate.getDate() - 100);
		
		let baseValue = 50000;
		if (indicatorId === 'rsi') baseValue = 50;
		else if (indicatorId === 'macd') baseValue = 0;
		else if (indicatorId === 'atr') baseValue = 1000;
		
		for (let i = 0; i < 100; i++) {
			const date = new Date(startDate);
			date.setDate(date.getDate() + i);
			
			let value;
			if (indicatorId === 'rsi') {
				value = 30 + Math.sin(i * 0.1) * 20 + Math.random() * 40;
			} else if (indicatorId === 'macd') {
				value = Math.sin(i * 0.1) * 500 + Math.random() * 200;
			} else if (indicatorId === 'atr') {
				value = 500 + Math.random() * 1000 + Math.sin(i * 0.05) * 300;
			} else {
				value = baseValue + Math.sin(i * 0.1) * 5000 + Math.random() * 2000;
			}
			
			data.push({
				time: date.toISOString().split('T')[0],
				value: Math.round(value * 100) / 100
			});
		}
		return data;
	}

	function drawChart() {
		if (!canvas || !chartData.length) return;
		
		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		const dpr = window.devicePixelRatio || 1;
		const rect = canvas.getBoundingClientRect();
		
		canvas.width = rect.width * dpr;
		canvas.height = rect.height * dpr;
		ctx.scale(dpr, dpr);
		
		// Set canvas size
		canvas.style.width = rect.width + 'px';
		canvas.style.height = rect.height + 'px';

		// Clear canvas
		const bgColor = chartTheme === 'dark' ? '#1e1e1e' : '#ffffff';
		const textColor = chartTheme === 'dark' ? '#d1d4dc' : '#191919';
		const gridColor = chartTheme === 'dark' ? '#363738' : '#e1e3e6';
		const lineColor = '#2962FF';

		ctx.fillStyle = bgColor;
		ctx.fillRect(0, 0, rect.width, rect.height);

		// Chart margins
		const margin = { top: 20, right: 60, bottom: 40, left: 60 };
		const chartWidth = rect.width - margin.left - margin.right;
		const chartHeight = rect.height - margin.top - margin.bottom;

		// Find min/max values
		const values = chartData.map(d => d.value);
		const minValue = Math.min(...values);
		const maxValue = Math.max(...values);
		const valueRange = maxValue - minValue;
		const padding = valueRange * 0.1;

		// Draw grid lines
		ctx.strokeStyle = gridColor;
		ctx.lineWidth = 1;
		
		// Horizontal grid lines
		for (let i = 0; i <= 5; i++) {
			const y = margin.top + (chartHeight / 5) * i;
			ctx.beginPath();
			ctx.moveTo(margin.left, y);
			ctx.lineTo(margin.left + chartWidth, y);
			ctx.stroke();
		}

		// Vertical grid lines
		for (let i = 0; i <= 10; i++) {
			const x = margin.left + (chartWidth / 10) * i;
			ctx.beginPath();
			ctx.moveTo(x, margin.top);
			ctx.lineTo(x, margin.top + chartHeight);
			ctx.stroke();
		}

		// Draw price line
		ctx.strokeStyle = lineColor;
		ctx.lineWidth = 2;
		ctx.beginPath();

		chartData.forEach((point, index) => {
			const x = margin.left + (index / (chartData.length - 1)) * chartWidth;
			const y = margin.top + chartHeight - ((point.value - minValue + padding) / (valueRange + 2 * padding)) * chartHeight;
			
			if (index === 0) {
				ctx.moveTo(x, y);
			} else {
				ctx.lineTo(x, y);
			}
		});

		ctx.stroke();

		// Draw labels
		ctx.fillStyle = textColor;
		ctx.font = '12px Inter, sans-serif';
		ctx.textAlign = 'right';

		// Y-axis labels
		for (let i = 0; i <= 5; i++) {
			const value = maxValue + padding - (i / 5) * (valueRange + 2 * padding);
			const y = margin.top + (chartHeight / 5) * i + 4;
			ctx.fillText(value.toFixed(2), margin.left - 10, y);
		}

		// X-axis labels
		ctx.textAlign = 'center';
		const labelCount = 5;
		for (let i = 0; i <= labelCount; i++) {
			const dataIndex = Math.floor((i / labelCount) * (chartData.length - 1));
			const point = chartData[dataIndex];
			const x = margin.left + (dataIndex / (chartData.length - 1)) * chartWidth;
			const y = margin.top + chartHeight + 20;
			ctx.fillText(point.time, x, y);
		}

		// Draw chart title
		ctx.textAlign = 'left';
		ctx.font = 'bold 14px Inter, sans-serif';
		ctx.fillText(indicatorId ? `${indicatorId.toUpperCase()} Chart` : 'Price Chart', margin.left, 15);
	}

	function resizeCanvas() {
		if (canvas) {
			drawChart();
		}
	}

	onMount(async () => {
		try {
			loading = true;
			error = '';

			// Generate sample data
			chartData = generateSampleData();

			// Wait a bit for DOM to be ready
			await new Promise(resolve => setTimeout(resolve, 100));

			if (!canvas) {
				throw new Error('Canvas not found');
			}

			// Initial draw
			drawChart();

			// Add resize listener
			window.addEventListener('resize', resizeCanvas);

			loading = false;

		} catch (err) {
			console.error('Chart error:', err);
			error = err instanceof Error ? err.message : 'Failed to create chart';
			loading = false;
		}
	});

	onDestroy(() => {
		window.removeEventListener('resize', resizeCanvas);
	});

	function switchTheme() {
		chartTheme = chartTheme === 'dark' ? 'light' : 'dark';
		drawChart();
	}

	function resetZoom() {
		drawChart();
	}
</script>

<div class="simple-chart">
	<!-- Simple Toolbar -->
	<div class="chart-toolbar">
		<div class="toolbar-section">
			<h3>
				{#if indicatorId}
					{indicatorId.toUpperCase()} Chart
				{:else}
					Price Chart
				{/if}
			</h3>
		</div>
		
		<div class="toolbar-section">
			<button 
				class="control-btn"
				on:click={resetZoom}
				title="Reset Zoom"
				aria-label="Reset Zoom"
			>
				↻
			</button>
			<button 
				class="control-btn"
				on:click={switchTheme}
				title="Switch Theme"
				aria-label="Switch Theme"
			>
				{chartTheme === 'dark' ? '☀️' : '🌙'}
			</button>
		</div>
	</div>

	<!-- Chart Container -->
	<div 
		class="chart-container" 
		class:loading 
		class:error={!!error}
		bind:this={chartContainer}
		style="height: {height}px"
	>
		{#if loading}
			<div class="chart-loading">
				<div class="loading-spinner"></div>
				<p>Loading chart...</p>
			</div>
		{:else if error}
			<div class="chart-error">
				<h3>Chart Error</h3>
				<p>{error}</p>
				<p class="error-help">
					Try refreshing the page or check the browser console for more details.
				</p>
			</div>
		{:else}
			<canvas 
				bind:this={canvas}
				class="chart-canvas"
				style="width: 100%; height: 100%;"
			></canvas>
		{/if}
	</div>
</div>

<style>
	.simple-chart {
		width: 100%;
		background: var(--surface);
		border-radius: var(--radius-lg);
		border: 1px solid var(--border);
		overflow: hidden;
		box-shadow: var(--shadow-md);
	}

	.chart-toolbar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: var(--space-4);
		background: var(--surface-elevated);
		border-bottom: 1px solid var(--border);
	}

	.toolbar-section {
		display: flex;
		align-items: center;
		gap: var(--space-2);
	}

	.toolbar-section h3 {
		margin: 0;
		color: var(--text-primary);
		font-size: 1.125rem;
		font-weight: 600;
	}

	.control-btn {
		padding: var(--space-2);
		border: 1px solid var(--border);
		background: var(--surface);
		color: var(--text-secondary);
		border-radius: var(--radius);
		cursor: pointer;
		transition: var(--transition);
		display: flex;
		align-items: center;
		justify-content: center;
		width: 36px;
		height: 36px;
		font-size: 1rem;
	}

	.control-btn:hover {
		background: var(--surface-elevated);
		border-color: var(--primary-color);
		transform: translateY(-1px);
	}

	.chart-container {
		position: relative;
		width: 100%;
		background: var(--surface);
	}

	.chart-container.loading,
	.chart-container.error {
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.chart-canvas {
		display: block;
		width: 100%;
		height: 100%;
	}

	.chart-loading {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: var(--space-4);
		color: var(--text-secondary);
	}

	.loading-spinner {
		width: 40px;
		height: 40px;
		border: 3px solid var(--border);
		border-top: 3px solid var(--primary-color);
		border-radius: 50%;
		animation: spin 1s linear infinite;
	}

	@keyframes spin {
		0% { transform: rotate(0deg); }
		100% { transform: rotate(360deg); }
	}

	.chart-error {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: var(--space-3);
		color: var(--text-secondary);
		text-align: center;
		padding: var(--space-8);
	}

	.chart-error h3 {
		margin: 0;
		color: var(--text-primary);
		font-size: 1.125rem;
		font-weight: 600;
	}

	.chart-error p {
		margin: 0;
		font-size: 0.875rem;
	}

	.error-help {
		opacity: 0.7;
		font-size: 0.8rem !important;
	}
</style>