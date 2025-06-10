<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { createChart, type IChartApi, type ISeriesApi, ColorType, LineStyle, type UTCTimestamp } from 'lightweight-charts';
	import { parseCsvData, convertToCandlestickData, convertToVolumeData, getIndicatorData, IndicatorCalculator, type RawCandleData, type IndicatorData } from '$lib/chartData';

	export let indicatorId: string = '';
	export let height: number = 600;

	let chartContainer: HTMLDivElement;
	let chart: IChartApi;
	let candlestickSeries: ISeriesApi<'Candlestick'>;
	let volumeSeries: ISeriesApi<'Histogram'>;
	let indicatorSeries: ISeriesApi<'Line'>[] = [];
	let rawData: RawCandleData[] = [];
	let loading = true;
	let error = '';

	// Chart settings
	let showVolume = true;
	let chartTheme: 'light' | 'dark' = 'dark';

	onMount(async () => {
		// Add a small delay to ensure DOM is fully ready
		await new Promise(resolve => setTimeout(resolve, 100));
		await initializeChart();
		await loadData();
	});

	onDestroy(() => {
		if (chart) {
			chart.remove();
		}
	});

	async function initializeChart() {
		try {
			if (!chartContainer) {
				throw new Error('Chart container not found');
			}

			// Ensure createChart is available
			if (typeof createChart !== 'function') {
				throw new Error('TradingView createChart function not available');
			}

			const chartOptions = {
				width: chartContainer.clientWidth,
				height: height,
				layout: {
					background: { 
						type: ColorType.Solid, 
						color: chartTheme === 'dark' ? '#1e1e1e' : '#ffffff' 
					},
					textColor: chartTheme === 'dark' ? '#d1d4dc' : '#191919',
					fontSize: 12,
					fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif'
				},
				grid: {
					vertLines: { 
						color: chartTheme === 'dark' ? '#2a2e39' : '#f0f3fa',
						visible: true
					},
					horzLines: { 
						color: chartTheme === 'dark' ? '#2a2e39' : '#f0f3fa',
						visible: true
					}
				},
				crosshair: {
					mode: 1,
					vertLine: {
						color: chartTheme === 'dark' ? '#6a7179' : '#9598a1',
						width: 1,
						style: LineStyle.Dashed
					},
					horzLine: {
						color: chartTheme === 'dark' ? '#6a7179' : '#9598a1',
						width: 1,
						style: LineStyle.Dashed
					}
				},
				rightPriceScale: {
					borderColor: chartTheme === 'dark' ? '#2a2e39' : '#d6d8db',
					scaleMargins: {
						top: 0.1,
						bottom: showVolume ? 0.4 : 0.1
					}
				},
				timeScale: {
					borderColor: chartTheme === 'dark' ? '#2a2e39' : '#d6d8db',
					timeVisible: true,
					secondsVisible: false,
					fixLeftEdge: true,
					fixRightEdge: true
				},
				handleScroll: {
					mouseWheel: true,
					pressedMouseMove: true,
					horzTouchDrag: true,
					vertTouchDrag: true
				},
				handleScale: {
					axisPressedMouseMove: true,
					mouseWheel: true,
					pinch: true
				}
			};

			chart = createChart(chartContainer, chartOptions);

			if (!chart) {
				throw new Error('Failed to create chart instance');
			}

			// Verify chart has required methods
			if (typeof chart.addCandlestickSeries !== 'function') {
				throw new Error('Chart instance missing addCandlestickSeries method');
			}

			// Add candlestick series
			candlestickSeries = chart.addCandlestickSeries({
				upColor: '#26a69a',
				downColor: '#ef5350',
				borderVisible: false,
				wickUpColor: '#26a69a',
				wickDownColor: '#ef5350'
			});

			// Add volume series if enabled
			if (showVolume && typeof chart.addHistogramSeries === 'function') {
				volumeSeries = chart.addHistogramSeries({
					color: '#26a69a',
					priceFormat: {
						type: 'volume'
					},
					priceScaleId: 'volume',
					scaleMargins: {
						top: 0.7,
						bottom: 0
					}
				});

				// Configure volume price scale
				if (typeof chart.priceScale === 'function') {
					chart.priceScale('volume').applyOptions({
						scaleMargins: {
							top: 0.7,
							bottom: 0
						}
					});
				}
			}

			// Handle resize
			const resizeObserver = new ResizeObserver(entries => {
				if (entries.length === 0 || entries[0].target !== chartContainer) return;
				const { width, height } = entries[0].contentRect;
				if (chart && typeof chart.applyOptions === 'function') {
					chart.applyOptions({ width, height });
				}
			});

			resizeObserver.observe(chartContainer);

		} catch (err) {
			console.error('Failed to initialize chart:', err);
			error = err instanceof Error ? err.message : 'Failed to initialize chart';
			loading = false;
		}
	}

	async function loadData() {
		try {
			loading = true;
			error = '';
			
			// Ensure chart is initialized
			if (!chart) {
				throw new Error('Chart not initialized');
			}

			rawData = await parseCsvData();
			
			if (!rawData || rawData.length === 0) {
				throw new Error('No data loaded');
			}

			// Convert and set candlestick data
			const candlestickData = convertToCandlestickData(rawData);
			
			if (!candlestickSeries) {
				throw new Error('Candlestick series not initialized');
			}

			if (typeof candlestickSeries.setData !== 'function') {
				throw new Error('Candlestick series missing setData method');
			}

			candlestickSeries.setData(candlestickData);

			// Set volume data if volume series exists
			if (volumeSeries && showVolume && typeof volumeSeries.setData === 'function') {
				const volumeData = convertToVolumeData(rawData);
				volumeSeries.setData(volumeData);
			}

			// Load specific indicator if provided
			if (indicatorId) {
				await loadIndicator(indicatorId);
			}

			// Fit content to show all data
			if (typeof chart.timeScale === 'function') {
				const timeScale = chart.timeScale();
				if (typeof timeScale.fitContent === 'function') {
					timeScale.fitContent();
				}
			}

		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load chart data';
			console.error('Chart data loading error:', err);
		} finally {
			loading = false;
		}
	}

	async function loadIndicator(id: string) {
		if (!rawData.length) return;

		try {
			const indicators = getIndicatorData(id, rawData);
			
			for (const indicator of indicators) {
				const series = chart.addLineSeries({
					color: indicator.color,
					lineWidth: indicator.lineWidth || 2,
					lineStyle: indicator.lineStyle || LineStyle.Solid,
					priceScaleId: indicator.priceScaleId || 'right',
					title: indicator.name
				});

				series.setData(indicator.data);
				indicatorSeries.push(series);

				// Configure price scale for specific indicators
				if (indicator.priceScaleId && indicator.priceScaleId !== 'right') {
					chart.priceScale(indicator.priceScaleId).applyOptions({
						scaleMargins: {
							top: 0.1,
							bottom: 0.1
						}
					});
				}
			}
		} catch (err) {
			console.error(`Failed to load indicator ${id}:`, err);
		}
	}



	function switchTheme() {
		chartTheme = chartTheme === 'dark' ? 'light' : 'dark';
		
		const backgroundColor = chartTheme === 'dark' ? '#1e1e1e' : '#ffffff';
		const textColor = chartTheme === 'dark' ? '#d1d4dc' : '#191919';
		const gridColor = chartTheme === 'dark' ? '#2a2e39' : '#f0f3fa';
		const borderColor = chartTheme === 'dark' ? '#2a2e39' : '#d6d8db';
		const crosshairColor = chartTheme === 'dark' ? '#6a7179' : '#9598a1';

		chart.applyOptions({
			layout: {
				background: { type: ColorType.Solid, color: backgroundColor },
				textColor: textColor
			},
			grid: {
				vertLines: { color: gridColor },
				horzLines: { color: gridColor }
			},
			rightPriceScale: { borderColor },
			timeScale: { borderColor },
			crosshair: {
				vertLine: { color: crosshairColor },
				horzLine: { color: crosshairColor }
			}
		});
	}

	function resetChart() {
		chart.timeScale().fitContent();
	}

</script>

<div class="trading-chart">
	<!-- Chart Toolbar -->
	<div class="chart-toolbar">
		<div class="toolbar-section">
			<div class="chart-controls">
				<button 
					class="control-btn"
					on:click={switchTheme}
					title="Switch Theme"
					aria-label="Switch Theme"
				>
					<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
						<circle cx="12" cy="12" r="5"/>
						<line x1="12" y1="1" x2="12" y2="3"/>
						<line x1="12" y1="21" x2="12" y2="23"/>
						<line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
						<line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
						<line x1="1" y1="12" x2="3" y2="12"/>
						<line x1="21" y1="12" x2="23" y2="12"/>
						<line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
						<line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
					</svg>
				</button>
				<button 
					class="control-btn"
					on:click={resetChart}
					title="Reset Zoom"
					aria-label="Reset Zoom"
				>
					<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
						<polyline points="1,4 1,10 7,10"/>
						<path d="M3.51,15a9,9,0,0,0,14.85-3.36"/>
						<polyline points="23,20 23,14 17,14"/>
						<path d="M20.49,9A9,9,0,0,0,5.64,5.64L1,10"/>
					</svg>
				</button>
			</div>
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
				<p>Loading chart data...</p>
			</div>
		{:else if error}
			<div class="chart-error">
				<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
					<circle cx="12" cy="12" r="10"/>
					<line x1="15" y1="9" x2="9" y2="15"/>
					<line x1="9" y1="9" x2="15" y2="15"/>
				</svg>
				<h3>Failed to Load Chart</h3>
				<p>{error}</p>
				<button class="retry-btn" on:click={loadData}>Retry</button>
			</div>
		{/if}
	</div>

	<!-- Chart Info -->
	{#if !loading && !error && rawData.length > 0}
		<div class="chart-info">
			<div class="info-item">
				<span class="info-label">Data Points:</span>
				<span class="info-value">{rawData.length.toLocaleString()}</span>
			</div>
			<div class="info-item">
				<span class="info-label">Timeframe:</span>
				<span class="info-value">4H</span>
			</div>
			<div class="info-item">
				<span class="info-label">Indicator:</span>
				<span class="info-value">{indicatorId ? indicatorId.toUpperCase() : 'Price Chart'}</span>
			</div>
		</div>
	{/if}
</div>

<style>
	.trading-chart {
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
		gap: var(--space-4);
		flex-wrap: wrap;
	}

	.toolbar-section {
		display: flex;
		align-items: center;
		gap: var(--space-2);
	}



	.chart-controls {
		display: flex;
		gap: var(--space-1);
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
	}

	.control-btn:hover {
		background: var(--surface-elevated);
		border-color: var(--primary-color);
		color: var(--primary-color);
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

	.chart-error svg {
		color: var(--error-color);
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
		opacity: 0.8;
	}

	.retry-btn {
		padding: var(--space-2) var(--space-4);
		background: var(--primary-color);
		color: white;
		border: none;
		border-radius: var(--radius);
		font-weight: 500;
		cursor: pointer;
		transition: var(--transition);
	}

	.retry-btn:hover {
		background: var(--primary-dark);
		transform: translateY(-1px);
	}

	.chart-info {
		display: flex;
		align-items: center;
		gap: var(--space-6);
		padding: var(--space-3) var(--space-4);
		background: var(--surface-elevated);
		border-top: 1px solid var(--border);
		font-size: 0.875rem;
	}

	.info-item {
		display: flex;
		align-items: center;
		gap: var(--space-2);
	}

	.info-label {
		color: var(--text-secondary);
		font-weight: 500;
	}

	.info-value {
		color: var(--text-primary);
		font-weight: 600;
	}

	@media (max-width: 768px) {
		.chart-toolbar {
			flex-direction: column;
			align-items: stretch;
			gap: var(--space-3);
		}

		.toolbar-section {
			justify-content: center;
		}

		.chart-info {
			flex-direction: column;
			align-items: stretch;
			gap: var(--space-2);
		}

		.info-item {
			justify-content: space-between;
		}
	}

	@media (max-width: 480px) {
		.chart-controls {
			flex-wrap: wrap;
		}

		.chart-toolbar {
			padding: var(--space-3);
		}
	}
</style>