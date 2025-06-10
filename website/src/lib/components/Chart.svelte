<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { createChart, type IChartApi, type ISeriesApi, type CandlestickData, type LineData } from 'lightweight-charts';
	import type { CandleData } from '$lib/stores';
	
	export let data: CandleData[] = [];
	export let indicatorData: LineData[] = [];
	export let height: number = 400;
	export let type: 'candlestick' | 'line' = 'candlestick';
	export let title: string = '';
	export let showOverbought: boolean = false;
	export let showOversold: boolean = false;
	export let overboughtLevel: number = 70;
	export let oversoldLevel: number = 30;
	
	let chartContainer: HTMLDivElement;
	let chart: IChartApi;
	let mainSeries: ISeriesApi<any>;
	let indicatorSeries: ISeriesApi<'Line'>;
	let overboughtSeries: ISeriesApi<'Line'>;
	let oversoldSeries: ISeriesApi<'Line'>;
	
	onMount(() => {
		if (chartContainer) {
			initChart();
		}
	});
	
	onDestroy(() => {
		if (chart) {
			chart.remove();
		}
	});
	
	function initChart() {
		chart = createChart(chartContainer, {
			width: chartContainer.offsetWidth,
			height: height,
			layout: {
				backgroundColor: '#ffffff',
				textColor: '#333',
			},
			grid: {
				vertLines: { color: '#f0f0f0' },
				horzLines: { color: '#f0f0f0' },
			},
			rightPriceScale: {
				borderVisible: false,
				scaleMargins: {
					top: 0.1,
					bottom: 0.1,
				},
			},
			timeScale: {
				borderVisible: false,
				timeVisible: true,
				secondsVisible: false,
			},
		});
		
		if (type === 'candlestick') {
			mainSeries = chart.addCandlestickSeries({
				upColor: '#26a69a',
				downColor: '#ef5350',
				borderVisible: false,
				wickUpColor: '#26a69a',
				wickDownColor: '#ef5350',
			});
			
			// Convert data format for candlestick
			const candleData: CandlestickData[] = data.map(item => ({
				time: item.time,
				open: item.open,
				high: item.high,
				low: item.low,
				close: item.close,
			}));
			
			mainSeries.setData(candleData);
		} else {
			mainSeries = chart.addLineSeries({
				color: '#2962FF',
				lineWidth: 2,
			});
			
			if (indicatorData.length > 0) {
				mainSeries.setData(indicatorData);
			}
		}
		
		// Add indicator line if provided
		if (indicatorData.length > 0 && type === 'candlestick') {
			indicatorSeries = chart.addLineSeries({
				color: '#FF6B35',
				lineWidth: 2,
				priceScaleId: 'right',
			});
			indicatorSeries.setData(indicatorData);
		}
		
		// Add overbought/oversold lines for oscillators
		if (showOverbought) {
			const overboughtData = indicatorData.map(item => ({
				time: item.time,
				value: overboughtLevel
			}));
			
			overboughtSeries = chart.addLineSeries({
				color: '#FF6B6B',
				lineWidth: 1,
				lineStyle: 2, // dashed
				priceScaleId: 'right',
			});
			overboughtSeries.setData(overboughtData);
		}
		
		if (showOversold) {
			const oversoldData = indicatorData.map(item => ({
				time: item.time,
				value: oversoldLevel
			}));
			
			oversoldSeries = chart.addLineSeries({
				color: '#4ECDC4',
				lineWidth: 1,
				lineStyle: 2, // dashed
				priceScaleId: 'right',
			});
			oversoldSeries.setData(oversoldData);
		}
		
		// Handle resize
		const resizeObserver = new ResizeObserver(() => {
			if (chartContainer && chart) {
				chart.applyOptions({ width: chartContainer.offsetWidth });
			}
		});
		
		resizeObserver.observe(chartContainer);
		
		return () => {
			resizeObserver.disconnect();
		};
	}
	
	// Reactive updates
	$: if (chart && mainSeries && data.length > 0) {
		if (type === 'candlestick') {
			const candleData: CandlestickData[] = data.map(item => ({
				time: item.time,
				open: item.open,
				high: item.high,
				low: item.low,
				close: item.close,
			}));
			mainSeries.setData(candleData);
		}
	}
	
	$: if (chart && indicatorSeries && indicatorData.length > 0) {
		indicatorSeries.setData(indicatorData);
	}
</script>

<div class="chart-wrapper">
	{#if title}
		<h3 class="chart-title">{title}</h3>
	{/if}
	<div bind:this={chartContainer} class="chart-container"></div>
</div>

<style>
	.chart-wrapper {
		background: var(--surface-color, #ffffff);
		border: 1px solid var(--border-color, #e2e8f0);
		border-radius: 8px;
		padding: 1rem;
		margin: 2rem 0;
	}
	
	.chart-title {
		margin-bottom: 1rem;
		color: var(--text-primary, #1e293b);
		font-size: 1.125rem;
		font-weight: 600;
	}
	
	.chart-container {
		width: 100%;
		position: relative;
	}
</style>