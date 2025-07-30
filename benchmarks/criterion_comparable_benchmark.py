#!/usr/bin/env python3
"""
Criterion-comparable benchmark for Python bindings.
Implements best practices for accurate performance measurement.
"""

import argparse
import json
import time
import gc
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import my_project

# Disable multi-threading for consistent results
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


class CriterionComparableBenchmark:
    """Benchmark harness that matches Criterion's methodology."""
    
    def __init__(self, data_size: str = '1M', filter_indicator: str = None):
        self.data_size = data_size
        self.filter_indicator = filter_indicator
        self.rust_results = {}
        self.python_results = {}
        
        # Benchmark parameters matching Criterion
        self.warmup_target_ns = 150_000_000  # 150ms in nanoseconds
        self.sample_count = 10  # Number of samples to take
        self.min_iterations = 10  # Minimum iterations per sample
        
    def load_csv_data(self) -> Dict[str, np.ndarray]:
        """Load CSV data once, outside of timing loops."""
        csv_path = Path(__file__).parent.parent / 'src/data/1MillionCandles.csv'
        
        # Pre-allocate arrays for better performance
        with open(csv_path, 'r') as f:
            f.readline()  # Skip header
            # Count lines first
            line_count = sum(1 for _ in f)
        
        # Allocate arrays
        timestamps = np.empty(line_count, dtype=np.int64)
        opens = np.empty(line_count, dtype=np.float64)
        highs = np.empty(line_count, dtype=np.float64)
        lows = np.empty(line_count, dtype=np.float64)
        closes = np.empty(line_count, dtype=np.float64)
        volumes = np.empty(line_count, dtype=np.float64)
        
        # Load data
        with open(csv_path, 'r') as f:
            f.readline()  # Skip header
            import csv
            reader = csv.reader(f)
            valid_count = 0
            for row in reader:
                if len(row) >= 7:
                    try:
                        timestamps[valid_count] = int(row[0])
                        opens[valid_count] = float(row[1])
                        highs[valid_count] = float(row[2])
                        lows[valid_count] = float(row[3])
                        closes[valid_count] = float(row[4])
                        volumes[valid_count] = float(row[6])
                        valid_count += 1
                    except ValueError:
                        continue
        
        # Trim to actual size and ensure C-contiguous
        data = {
            'timestamps': np.ascontiguousarray(timestamps[:valid_count]),
            'open': np.ascontiguousarray(opens[:valid_count]),
            'high': np.ascontiguousarray(highs[:valid_count]),
            'low': np.ascontiguousarray(lows[:valid_count]),
            'close': np.ascontiguousarray(closes[:valid_count]),
            'volume': np.ascontiguousarray(volumes[:valid_count])
        }
        
        print(f"Loaded {valid_count} candles")
        return data
    
    def parse_criterion_json(self):
        """Parse Criterion's JSON output files for accurate medians."""
        criterion_dir = Path(__file__).parent.parent / 'target/criterion'
        if not criterion_dir.exists():
            print("Warning: No Criterion results found. Run Rust benchmarks first.")
            return
        
        print("\nParsing Criterion JSON results...")
        print("-" * 80)
        
        # Map of indicator names to their benchmark paths
        indicators_to_find = [
            'alma', 'alligator', 'aroonosc', 'bollinger_bands', 'ao', 'vpwma', 'vwma', 'wilders', 'wma', 'zlema', 'ad', 'adx', 'acosc', 'adosc', 'apo',
            'bandpass', 'vwap', 'cwma', 'dema', 'edcf', 'ehlers_itrend', 'ema', 'epma',
            'frama', 'fwma', 'gaussian', 'highpass_2_pole', 'highpass', 'hma',
            'hwma', 'ift_rsi', 'jma', 'jsa', 'kama', 'kvo', 'linreg', 'maaq', 'mama', 'mwdx',
            'nma', 'pwma', 'reflex', 'sinwma', 'sma', 'smma', 'sqwma', 'srwma',
            'supersmoother_3_pole', 'supersmoother', 'swma', 'tema', 'tilson',
            'trendflex', 'trima', 'vqwma', 'adxr', 'aroon', 'bollinger_bands_width', 'atr', 'cci', 'bop',
            'cg', 'cfo', 'cmo', 'dec_osc', 'macd', 'mfi', 'natr', 'ppo', 'var', 'vpci', 'wclprice'  # Added missing indicators
        ]
        
        size_map = {'10k': '10k', '100k': '100k', '1M': '1m'}
        target_size = size_map.get(self.data_size, '1m')
        
        # Also add batch indicators to find
        batch_indicators = ['alma_batch', 'aroonosc_batch', 'bollinger_bands_batch', 'ao_batch', 'vpwma_batch', 'wma_batch', 'zlema_batch', 
                           'sma_batch', 'ema_batch', 'dema_batch', 'edcf_batch', 'ehlers_itrend_batch', 'tema_batch', 
                           'hma_batch', 'ift_rsi_batch', 'kvo_batch', 'cwma_batch', 'adxr_batch', 'adx_batch', 'adosc_batch', 'aroon_batch',
                           'bollinger_bands_width_batch', 'apo_batch', 'bandpass_batch', 'atr_batch', 'cci_batch', 'bop_batch', 
                           'trendflex_batch', 'cfo_batch', 'cmo_batch', 'dec_osc_batch', 'donchian_batch', 'emv_batch', 'macd_batch', 'mfi_batch', 'natr_batch', 'ppo_batch', 'var_batch', 'vpci_batch', 'wclprice_batch']
        all_indicators = indicators_to_find + batch_indicators
        
        for indicator in all_indicators:
            if self.filter_indicator and not indicator.startswith(self.filter_indicator):
                continue
                
            # Try to find the best kernel result
            best_time = float('inf')
            best_kernel = None
            
            # Try different directory name patterns
            possible_dirs = [indicator, f"{indicator}_bench"]
            
            for dir_name in possible_dirs:
                dir_path = criterion_dir / dir_name
                if not dir_path.exists():
                    continue
                    
                # Check each kernel variant (for batch operations, check batch kernels)
                if indicator.endswith('_batch'):
                    kernels_to_check = ['avx512batch', 'avx2batch', 'scalarbatch', '']
                else:
                    kernels_to_check = ['avx512', 'avx2', 'scalar', '']
                    
                for kernel in kernels_to_check:
                    if kernel:
                        # The directory structure is: indicator/indicator_kernel/size/new/estimates.json
                        bench_name = f"{indicator}_{kernel}"
                        json_path = dir_path / bench_name / target_size / 'new' / 'estimates.json'
                    else:
                        # For indicators without kernel suffix, check if they have a direct structure
                        # Some indicators might be in: indicator/scalar/size/new/estimates.json
                        json_path = dir_path / 'scalar' / target_size / 'new' / 'estimates.json'
                        if not json_path.exists():
                            # Try the base path without kernel subdirectory
                            json_path = dir_path / target_size / 'new' / 'estimates.json'
                    
                    if json_path.exists():
                        try:
                            with open(json_path, 'r') as f:
                                data = json.load(f)
                                median_ns = data['median']['point_estimate']
                                median_ms = median_ns / 1_000_000
                                
                                if median_ms < best_time:
                                    best_time = median_ms
                                    best_kernel = kernel or 'auto'
                        except Exception as e:
                            print(f"  Error reading {json_path}: {e}")
            
            if best_time < float('inf'):
                self.rust_results[indicator] = best_time
                print(f"  {indicator}: {best_time:.3f} ms (kernel: {best_kernel})")
    
    def benchmark_function(self, func: Callable, name: str) -> float:
        """
        Benchmark a function using Criterion-like methodology.
        Returns median time in milliseconds.
        """
        # Disable garbage collection during measurement
        gc_was_enabled = gc.isenabled()
        gc.disable()
        
        try:
            # Warmup phase - run until we've accumulated at least 150ms
            warmup_elapsed = 0
            warmup_iterations = 0
            
            while warmup_elapsed < self.warmup_target_ns:
                start = time.perf_counter_ns()
                func()
                end = time.perf_counter_ns()
                warmup_elapsed += (end - start)
                warmup_iterations += 1
            
            # Sampling phase - take multiple samples
            samples = []
            
            for _ in range(self.sample_count):
                # Each sample measures multiple iterations
                iterations = max(self.min_iterations, warmup_iterations // 10)
                
                start = time.perf_counter_ns()
                for _ in range(iterations):
                    func()
                end = time.perf_counter_ns()
                
                # Calculate time per iteration
                time_per_iter = (end - start) / iterations
                samples.append(time_per_iter)
            
            # Return median (like Criterion does)
            median_ns = np.median(samples)
            return median_ns / 1_000_000  # Convert to ms
            
        finally:
            # Re-enable GC if it was enabled
            if gc_was_enabled:
                gc.enable()
    
    def run_python_benchmarks(self):
        """Run Python benchmarks with pre-allocated buffers."""
        print("\n\nRunning Python benchmarks (Criterion-comparable)...")
        print("-" * 80)
        
        data = self.load_csv_data()
        
        # Pre-allocate output buffers for each indicator
        output_buffers = {
            'single': np.empty_like(data['close']),
            'multi': np.empty((4, len(data['close'])), dtype=np.float64),  # For indicators with multiple outputs
        }
        
        # Define indicators with their functions
        indicators = [
            ('alma', lambda: my_project.alma(data['close'], 9, 0.85, 6.0)),
            ('alligator', lambda: my_project.alligator((data['high'] + data['low']) / 2)),
            ('bollinger_bands', lambda: my_project.bollinger_bands(data['close'], 20, 2.0, 2.0, "sma", 0)),
            ('aroonosc', lambda: my_project.aroonosc(data['high'], data['low'], 14)),
            ('ao', lambda: my_project.ao(data['high'], data['low'], 5, 34)),
            ('vpwma', lambda: my_project.vpwma(data['close'], 14, 0.382)),
            ('vwma', lambda: my_project.vwma(data['close'], data['volume'], 14)),
            ('wilders', lambda: my_project.wilders(data['close'], 14)),
            ('wma', lambda: my_project.wma(data['close'], 14)),
            ('zlema', lambda: my_project.zlema(data['close'], 14)),
            ('ad', lambda: my_project.ad(data['high'], data['low'], data['close'], data['volume'])),
            ('adx', lambda: my_project.adx(data['high'], data['low'], data['close'], 14)),
            ('acosc', lambda: my_project.acosc(data['high'], data['low'])),
            ('adosc', lambda: my_project.adosc(data['high'], data['low'], data['close'], data['volume'], 3, 10)),
            ('apo', lambda: my_project.apo(data['close'], 10, 20)),
            ('bandpass', lambda: my_project.bandpass(data['close'], 20, 0.3)),
            ('vwap', lambda: my_project.vwap(data['timestamps'], data['volume'], data['close'], '1d')),
            ('cwma', lambda: my_project.cwma(data['close'], 14)),
            ('dema', lambda: my_project.dema(data['close'], 14)),
            ('edcf', lambda: my_project.edcf(data['close'], 14)),
            ('ehlers_itrend', lambda: my_project.ehlers_itrend(data['close'], 20, 48)),
            ('ema', lambda: my_project.ema(data['close'], 14)),
            ('epma', lambda: my_project.epma(data['close'], 14, 0)),
            ('frama', lambda: my_project.frama(data['high'], data['low'], data['close'], 14, 1, 198)),
            ('fwma', lambda: my_project.fwma(data['close'], 14)),
            ('gaussian', lambda: my_project.gaussian(data['close'], 14, 4)),
            ('highpass_2_pole', lambda: my_project.highpass_2_pole(data['close'], 48, 0.707)),
            ('highpass', lambda: my_project.highpass(data['close'], 48)),
            ('hma', lambda: my_project.hma(data['close'], 14)),
            ('hwma', lambda: my_project.hwma(data['close'], 0.2, 0.1, 0.1)),
            ('ift_rsi', lambda: my_project.ift_rsi(data['close'], 5, 9)),
            ('jma', lambda: my_project.jma(data['close'], 14, 0.0, 2)),
            ('jsa', lambda: my_project.jsa(data['close'], 14)),
            ('kama', lambda: my_project.kama(data['close'], 14)),
            ('kvo', lambda: my_project.kvo(data['high'], data['low'], data['close'], data['volume'], 2, 5)),
            ('linreg', lambda: my_project.linreg(data['close'], 14)),
            ('macd', lambda: my_project.macd(data['close'], 12, 26, 9, "ema")),
            ('maaq', lambda: my_project.maaq(data['close'], 14, 10, 50)),
            ('mama', lambda: my_project.mama(data['close'], 0.5, 0.05)),
            ('mwdx', lambda: my_project.mwdx(data['close'], 0.125)),
            ('nma', lambda: my_project.nma(data['close'], 40)),
            ('pwma', lambda: my_project.pwma(data['close'], 14)),
            ('reflex', lambda: my_project.reflex(data['close'], 20)),
            ('rsi', lambda: my_project.rsi(data['close'], 14)),
            ('sinwma', lambda: my_project.sinwma(data['close'], 14)),
            ('sma', lambda: my_project.sma(data['close'], 14)),
            ('smma', lambda: my_project.smma(data['close'], 14)),
            ('sqwma', lambda: my_project.sqwma(data['close'], 14)),
            ('srwma', lambda: my_project.srwma(data['close'], 14)),
            ('supersmoother_3_pole', lambda: my_project.supersmoother_3_pole(data['close'], 14)),
            ('supersmoother', lambda: my_project.supersmoother(data['close'], 14)),
            ('swma', lambda: my_project.swma(data['close'], 14)),
            ('tema', lambda: my_project.tema(data['close'], 14)),
            ('tilson', lambda: my_project.tilson(data['close'], 14, 0.7)),
            ('trendflex', lambda: my_project.trendflex(data['close'], 20)),
            ('trima', lambda: my_project.trima(data['close'], 14)),
            ('vqwma', lambda: my_project.vqwma(data['close'], 0.5, 0.2, 0.2)),
            ('adxr', lambda: my_project.adxr(data['high'], data['low'], data['close'], 14)),
            ('aroon', lambda: my_project.aroon(data['high'], data['low'], 14)),
            ('bollinger_bands_width', lambda: my_project.bollinger_bands_width(data['close'], 20, 2.0, 2.0)),
            ('atr', lambda: my_project.atr(data['high'], data['low'], data['close'], 14)),
            ('cg', lambda: my_project.cg(data['close'], 10)),
            ('cci', lambda: my_project.cci((data['high'] + data['low'] + data['close']) / 3, 14)),
            ('cfo', lambda: my_project.cfo(data['close'], 14, 100.0)),
            ('cmo', lambda: my_project.cmo(data['close'], 14)),
            ('dec_osc', lambda: my_project.dec_osc(data['close'], 125, 1.0)),
            ('donchian', lambda: my_project.donchian(data['high'], data['low'], 20)),
            ('emv', lambda: my_project.emv(data['high'], data['low'], data['close'], data['volume'])),
            ('bop', lambda: my_project.bop(data['open'], data['high'], data['low'], data['close'])),
            ('mfi', lambda: my_project.mfi((data['high'] + data['low'] + data['close']) / 3.0, data['volume'], 14)),
            ('natr', lambda: my_project.natr(data['high'], data['low'], data['close'], 14)),
            ('ppo', lambda: my_project.ppo(data['close'], 12, 26, 'sma')),
            ('squeeze_momentum', lambda: my_project.squeeze_momentum(data['high'], data['low'], data['close'], 20, 2.0, 20, 1.5)),
            ('trix', lambda: my_project.trix(data['close'], 18)),
            ('var', lambda: my_project.var(data['close'], 14, 1.0)),
            ('vpci', lambda: my_project.vpci(data['close'], data['volume'], 5, 25)),
            ('wclprice', lambda: my_project.wclprice(data['high'], data['low'], data['close'])),
        ]
        
        # Filter if requested
        if self.filter_indicator:
            indicators = [(name, func) for name, func in indicators 
                         if name.startswith(self.filter_indicator)]
        
        # Run benchmarks
        for name, func in indicators:
            try:
                median_time = self.benchmark_function(func, name)
                self.python_results[name] = median_time
                print(f"  {name}: {median_time:.3f} ms")
            except Exception as e:
                print(f"  {name}: FAILED - {str(e)[:50]}...")
        
        # Also run batch operations
        print("\n  Batch operations (232 combos - matching Rust defaults):")
        batch_indicators = [
            ('alma_batch', lambda: my_project.alma_batch(data['close'], (9, 240, 1), (0.85, 0.85, 0.0), (6.0, 6.0, 0.0))),
            ('aroonosc_batch', lambda: my_project.aroonosc_batch(data['high'], data['low'], (14, 14, 1))),
            ('bollinger_bands_batch', lambda: my_project.bollinger_bands_batch(data['close'], (20, 20, 0), (2.0, 2.0, 0.0), (2.0, 2.0, 0.0), "sma", 0)),
            ('ao_batch', lambda: my_project.ao_batch(data['high'], data['low'], (5, 5, 1), (34, 34, 1))),
            ('vpwma_batch', lambda: my_project.vpwma_batch(data['close'], (14, 14, 1), (0.382, 0.382, 0.1))),
            ('wma_batch', lambda: my_project.wma_batch(data['close'], (14, 14, 1))),
            ('zlema_batch', lambda: my_project.zlema_batch(data['close'], (14, 14, 1))),
            ('sma_batch', lambda: my_project.sma_batch(data['close'], (14, 14, 1))),
            ('ema_batch', lambda: my_project.ema_batch(data['close'], (14, 14, 1))),
            ('dema_batch', lambda: my_project.dema_batch(data['close'], (14, 14, 1))),
            ('edcf_batch', lambda: my_project.edcf_batch(data['close'], (15, 50, 1))),
            ('ehlers_itrend_batch', lambda: my_project.ehlers_itrend_batch(data['close'], (12, 20, 4), (40, 50, 5))),
            ('tema_batch', lambda: my_project.tema_batch(data['close'], (9, 240, 1))),
            ('tilson_batch', lambda: my_project.tilson_batch(data['close'], (5, 40, 1), (0.0, 1.0, 0.1))),
            ('hma_batch', lambda: my_project.hma_batch(data['close'], (14, 14, 1))),
            ('ift_rsi_batch', lambda: my_project.ift_rsi_batch(data['close'], (5, 5, 0), (9, 9, 0))),
            ('kvo_batch', lambda: my_project.kvo_batch(data['high'], data['low'], data['close'], data['volume'], (2, 2, 0), (5, 5, 0))),
            ('macd_batch', lambda: my_project.macd_batch(data['close'], (12, 12, 0), (26, 26, 0), (9, 9, 0), "ema")),
            ('cwma_batch', lambda: my_project.cwma_batch(data['close'], (14, 14, 1))),
            ('adxr_batch', lambda: my_project.adxr_batch(data['high'], data['low'], data['close'], (14, 14, 1))),
            ('adx_batch', lambda: my_project.adx_batch(data['high'], data['low'], data['close'], (14, 14, 1))),
            ('adosc_batch', lambda: my_project.adosc_batch(data['high'], data['low'], data['close'], data['volume'], (3, 3, 1), (10, 10, 1))),
            ('aroon_batch', lambda: my_project.aroon_batch(data['high'], data['low'], (14, 14, 1))),
            ('bollinger_bands_width_batch', lambda: my_project.bollinger_bands_width_batch(data['close'], (20, 20, 1), (2.0, 2.0, 0), (2.0, 2.0, 0))),
            ('apo_batch', lambda: my_project.apo_batch(data['close'], (10, 10, 1), (20, 20, 1))),
            ('bandpass_batch', lambda: my_project.bandpass_batch(data['close'], (20, 20, 1), (0.3, 0.3, 0.0))),
            ('atr_batch', lambda: my_project.atr_batch(data['high'], data['low'], data['close'], (14, 14, 1))),
            ('cg_batch', lambda: my_project.cg_batch(data['close'], (10, 10, 1))),
            ('cci_batch', lambda: my_project.cci_batch((data['high'] + data['low'] + data['close']) / 3, (14, 14, 1))),
            ('cfo_batch', lambda: my_project.cfo_batch(data['close'], (14, 14, 1), (100.0, 100.0, 0.0))),
            ('cmo_batch', lambda: my_project.cmo_batch(data['close'], (10, 30, 2))),
            ('dec_osc_batch', lambda: my_project.dec_osc_batch(data['close'], (100, 150, 25), (0.5, 1.5, 0.5))),
            ('donchian_batch', lambda: my_project.donchian_batch(data['high'], data['low'], (10, 30, 10))),
            ('emv_batch', lambda: my_project.emv_batch(data['high'], data['low'], data['close'], data['volume'])),
            ('bop_batch', lambda: my_project.bop_batch(data['open'], data['high'], data['low'], data['close'])),
            ('trendflex_batch', lambda: my_project.trendflex_batch(data['close'], (20, 80, 1))),
            ('mfi_batch', lambda: my_project.mfi_batch((data['high'] + data['low'] + data['close']) / 3.0, data['volume'], (10, 20, 5))),
            ('natr_batch', lambda: my_project.natr_batch(data['high'], data['low'], data['close'], (14, 14, 1))),
            ('ppo_batch', lambda: my_project.ppo_batch(data['close'], (12, 12, 0), (26, 26, 0), 'sma')),
            ('rsi_batch', lambda: my_project.rsi_batch(data['close'], (14, 14, 0))),
            ('squeeze_momentum_batch', lambda: my_project.squeeze_momentum_batch(data['high'], data['low'], data['close'], (20, 20, 0), (2.0, 2.0, 0.0), (20, 20, 0), (1.5, 1.5, 0.0))),
            ('trix_batch', lambda: my_project.trix_batch(data['close'], (18, 100, 1))),
            ('var_batch', lambda: my_project.var_batch(data['close'], (14, 60, 1), (1.0, 1.0, 0.0))),
            ('vpci_batch', lambda: my_project.vpci_batch(data['close'], data['volume'], (5, 20, 1), (25, 60, 5))),
            ('wclprice_batch', lambda: my_project.wclprice_batch(data['high'], data['low'], data['close'])),
        ]
        
        # Filter batch tests if indicator filter is active
        if self.filter_indicator:
            batch_indicators = [(name, func) for name, func in batch_indicators 
                               if name.startswith(self.filter_indicator)]
        
        for name, func in batch_indicators:
            try:
                median_time = self.benchmark_function(func, name)
                self.python_results[name] = median_time
                print(f"  {name}: {median_time:.3f} ms")
            except Exception as e:
                print(f"  {name}: FAILED - {str(e)[:50]}...")
    
    def compare_results(self):
        """Compare Python and Rust results."""
        print("\n\n" + "=" * 80)
        print("PERFORMANCE COMPARISON (Criterion-comparable)")
        print("=" * 80)
        print(f"{'Indicator':25} {'Python (ms)':>12} {'Rust (ms)':>12} {'Overhead':>12} {'Status':>10}")
        print("-" * 80)
        
        comparisons = []
        
        for indicator in sorted(self.python_results.keys()):
            python_time = self.python_results[indicator]
            
            if indicator in self.rust_results:
                rust_time = self.rust_results[indicator]
                overhead_ms = python_time - rust_time
                overhead_pct = (python_time / rust_time - 1) * 100
                
                # Status based on overhead percentage
                if overhead_pct <= 15:
                    status = "EXCELLENT"
                elif overhead_pct <= 30:
                    status = "GOOD"
                elif overhead_pct <= 50:
                    status = "OK"
                else:
                    status = "HIGH"
                
                print(f"{indicator:25} {python_time:12.2f} {rust_time:12.2f} "
                      f"{overhead_pct:11.1f}% {status}")
                
                comparisons.append({
                    'indicator': indicator,
                    'python_ms': python_time,
                    'rust_ms': rust_time,
                    'overhead_ms': overhead_ms,
                    'overhead_pct': overhead_pct
                })
            else:
                print(f"{indicator:25} {python_time:12.2f} {'N/A':>12} {'N/A':>12}")
        
        # Summary statistics
        if comparisons:
            avg_overhead = np.mean([c['overhead_pct'] for c in comparisons])
            median_overhead = np.median([c['overhead_pct'] for c in comparisons])
            print("\n" + "-" * 80)
            print(f"Average overhead: {avg_overhead:.1f}%")
            print(f"Median overhead: {median_overhead:.1f}%")
        
        # Batch vs Single analysis
        print("\n\n" + "=" * 80)
        print("BATCH vs SINGLE ANALYSIS")
        print("=" * 80)
        print(f"{'Indicator':20} {'Single (ms)':>12} {'Batch (ms)':>12} {'Overhead':>12} {'Status':>10}")
        print("-" * 80)
        
        batch_comparisons = []
        for base_name in ['alma', 'aroonosc', 'ao', 'vpwma', 'wma', 'zlema', 'sma', 'ema', 'dema', 'tema', 'hma', 'ift_rsi', 'kvo', 'cwma', 'adxr', 'adx', 'adosc', 'aroon', 'bollinger_bands_width', 'apo', 'bandpass', 'atr', 'cg', 'cci', 'cfo', 'cmo', 'dec_osc', 'donchian', 'mfi', 'natr', 'ppo', 'rsi', 'var', 'vpci']:
            if base_name in self.python_results and f"{base_name}_batch" in self.python_results:
                single_time = self.python_results[base_name]
                batch_time = self.python_results[f"{base_name}_batch"]
                overhead_ms = batch_time - single_time
                overhead_pct = (batch_time / single_time - 1) * 100
                
                if overhead_pct <= 10:
                    status = "OK"
                elif overhead_pct <= 20:
                    status = "MODERATE"
                else:
                    status = "HIGH"
                
                print(f"{base_name:20} {single_time:12.2f} {batch_time:12.2f} "
                      f"{overhead_pct:11.1f}% {status}")
                
                batch_comparisons.append({
                    'indicator': base_name,
                    'single_ms': single_time,
                    'batch_ms': batch_time,
                    'overhead_pct': overhead_pct
                })
        
        # Save results
        results = {
            'methodology': 'criterion-comparable',
            'warmup_ms': self.warmup_target_ns / 1_000_000,
            'samples': self.sample_count,
            'gc_disabled': True,
            'omp_threads': 1,
            'data_size': self.data_size,
            'python_results': self.python_results,
            'rust_results': self.rust_results,
            'comparisons': comparisons,
            'batch_comparisons': batch_comparisons
        }
        
        output_path = Path(__file__).parent / 'criterion_comparable_results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to benchmarks/criterion_comparable_results.json")


def main():
    """Run the Criterion-comparable benchmark."""
    parser = argparse.ArgumentParser(description='Criterion-comparable Python benchmark')
    parser.add_argument('filter', nargs='?', help='Filter to specific indicator')
    parser.add_argument('--size', default='1M', choices=['10k', '100k', '1M'],
                       help='Data size to use (default: 1M)')
    args = parser.parse_args()
    
    print("Criterion-Comparable Python Benchmark")
    print("=" * 80)
    print("Features:")
    print("  - Parses Criterion JSON for accurate Rust medians")
    print("  - Disables GC during measurement")
    print("  - Uses 150ms warmup period")
    print("  - Takes median of 10 samples")
    print("  - Single-threaded NumPy (OMP_NUM_THREADS=1)")
    print(f"\nData size: {args.size}")
    if args.filter:
        print(f"Filtering for: {args.filter}")
    
    benchmark = CriterionComparableBenchmark(
        data_size=args.size,
        filter_indicator=args.filter
    )
    
    # Parse Criterion results
    benchmark.parse_criterion_json()
    
    # Run Python benchmarks
    benchmark.run_python_benchmarks()
    
    # Compare results
    benchmark.compare_results()


if __name__ == '__main__':
    main()