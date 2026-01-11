#!/usr/bin/env python3
"""Test cases for SuperSmoother 3-Pole indicator Python bindings."""

import unittest
import numpy as np
import my_project as ta_indicators
from test_utils import (
    load_test_data,
    assert_close,
    EXPECTED_OUTPUTS
)
from rust_comparison import compare_with_rust
import pytest
from datetime import datetime
import os
import sys

class TestSuperSmoother3Pole(unittest.TestCase):
    def setUp(self):
        """Set up test data before each test."""
        self.test_data = load_test_data()
        self.close_prices = self.test_data['close']
        
    def test_supersmoother_3_pole_partial_params(self):
        """Test SuperSmoother3Pole with partial parameters - mirrors check_supersmoother_3_pole_partial_params"""
        
        result = ta_indicators.supersmoother_3_pole(self.close_prices, 14)  
        self.assertEqual(len(result), len(self.close_prices))
    
    def test_supersmoother_3_pole_accuracy(self):
        """Test SuperSmoother3Pole matches expected values from Rust tests - mirrors check_supersmoother_3_pole_accuracy"""
        expected = EXPECTED_OUTPUTS['supersmoother_3_pole']
        
        
        result = ta_indicators.supersmoother_3_pole(
            self.close_prices,
            period=expected['default_params']['period']
        )
        
        
        self.assertEqual(len(result), len(self.close_prices))
        
        
        
        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=0,
            atol=1e-8,
            msg="SuperSmoother3Pole last 5 values mismatch"
        )
        
        
        
        compare_with_rust('supersmoother_3_pole', result, 'close', expected['default_params'], rtol=0, atol=1e-8)
    
    def test_supersmoother_3_pole_default_candles(self):
        """Test SuperSmoother3Pole with default parameters - mirrors check_supersmoother_3_pole_default_candles"""
        
        result = ta_indicators.supersmoother_3_pole(self.close_prices, 14)
        self.assertEqual(len(result), len(self.close_prices))
    
    def test_supersmoother_3_pole_zero_period(self):
        """Test SuperSmoother3Pole fails with zero period - mirrors check_supersmoother_3_pole_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.supersmoother_3_pole(input_data, period=0)
    
    def test_supersmoother_3_pole_period_exceeds_length(self):
        """Test SuperSmoother3Pole fails when period exceeds data length - mirrors check_supersmoother_3_pole_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.supersmoother_3_pole(data_small, period=10)
    
    def test_supersmoother_3_pole_very_small_dataset(self):
        """Test SuperSmoother3Pole fails with insufficient data - mirrors check_supersmoother_3_pole_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.supersmoother_3_pole(single_point, period=9)
    
    def test_supersmoother_3_pole_empty_input(self):
        """Test SuperSmoother3Pole fails with empty input - mirrors check_supersmoother_3_pole_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.supersmoother_3_pole(empty, period=14)
    
    def test_supersmoother_3_pole_all_nan_input(self):
        """Test SuperSmoother3Pole with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.supersmoother_3_pole(all_nan, period=14)
    
    def test_supersmoother_3_pole_streaming(self):
        """Test streaming SuperSmoother3Pole calculation."""
        period = 14
        
        
        batch_result = ta_indicators.supersmoother_3_pole(self.close_prices, period=period)
        
        
        stream = ta_indicators.SuperSmoother3PoleStream(period=period)
        stream_results = []
        
        for price in self.close_prices:
            val = stream.update(price)
            stream_results.append(val)
        
        stream_results = np.array(stream_results)
        
        
        valid_mask = ~np.isnan(batch_result) & ~np.isnan(stream_results)
        if np.any(valid_mask):
            max_diff = np.max(np.abs(batch_result[valid_mask] - stream_results[valid_mask]))
            self.assertLess(max_diff, 1e-10, f"Streaming vs batch max difference: {max_diff}")
    
    def test_supersmoother_3_pole_batch_processing(self):
        """Test batch processing for multiple periods."""
        
        periods = ta_indicators.supersmoother_3_pole_batch(
            self.close_prices,
            period_range=(10, 20, 5)  
        )
        
        
        self.assertIn('values', periods)
        self.assertIn('periods', periods)
        
        
        self.assertEqual(periods['values'].shape[0], 3)  
        self.assertEqual(periods['values'].shape[1], len(self.close_prices))
        
        
        np.testing.assert_array_equal(periods['periods'], [10, 15, 20])
        
        
        
        for i, period in enumerate([10, 15, 20]):
            row = periods['values'][i]
            
            for j in range(min(3, len(row))):
                self.assertFalse(np.isnan(row[j]), f"Value at index {j} for period {period} should not be NaN")
    
    def test_supersmoother_3_pole_nan_handling(self):
        """Test SuperSmoother3Pole handles NaN values correctly - mirrors check_supersmoother_3_pole_nan_handling"""
        result = ta_indicators.supersmoother_3_pole(self.close_prices, period=14)
        self.assertEqual(len(result), len(self.close_prices))
        
        
        if len(result) > 240:
            self.assertFalse(np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period")
        
        
        
        for i in range(min(3, len(result))):
            self.assertFalse(np.isnan(result[i]), f"Value at index {i} should not be NaN")
    
    def test_supersmoother_3_pole_with_leading_nans(self):
        """Test SuperSmoother3Pole correctly handles data that starts with NaN values."""
        
        data = np.array([np.nan] * 5 + list(range(1, 16)))  
        period = 3
        
        result = ta_indicators.supersmoother_3_pole(data, period=period)
        
        
        
        
        
        
        
        self.assertTrue(np.all(np.isnan(result[:5])), 
                       "Expected NaN output where input is NaN")
        
        
        np.testing.assert_almost_equal(result[5], 1.0, decimal=10)
        np.testing.assert_almost_equal(result[6], 2.0, decimal=10)
        np.testing.assert_almost_equal(result[7], 3.0, decimal=10)
        
        
        self.assertFalse(np.isnan(result[8]), "Filter should start calculating at index 8")
        self.assertNotEqual(result[8], 4.0, "Index 8 should be filtered, not passed through")
    
    def test_supersmoother_3_pole_kernel_selection(self):
        """Test different kernel options produce consistent results."""
        period = 14
        
        
        result_auto = ta_indicators.supersmoother_3_pole(self.close_prices, period=period, kernel='auto')
        result_scalar = ta_indicators.supersmoother_3_pole(self.close_prices, period=period, kernel='scalar')
        
        
        valid_mask = ~np.isnan(result_auto) & ~np.isnan(result_scalar)
        if np.any(valid_mask):
            max_diff = np.max(np.abs(result_auto[valid_mask] - result_scalar[valid_mask]))
            self.assertLess(max_diff, 1e-10, f"Kernel results differ by {max_diff}")
    
    def test_supersmoother_3_pole_error_handling(self):
        """Test error handling for invalid inputs."""
        
        with pytest.raises(ValueError) as exc_info:
            ta_indicators.supersmoother_3_pole(self.close_prices, period=0)
        self.assertIn("Invalid period", str(exc_info.value))
        
        
        with pytest.raises(ValueError) as exc_info:
            ta_indicators.supersmoother_3_pole(self.close_prices[:5], period=10)
        self.assertIn("Invalid period", str(exc_info.value))
        
        
        with pytest.raises(ValueError) as exc_info:
            ta_indicators.supersmoother_3_pole(np.full(10, np.nan), period=5)
        self.assertIn("All values are NaN", str(exc_info.value))
    
    def test_supersmoother_3_pole_edge_cases(self):
        """Test edge cases for SuperSmoother3Pole."""
        
        result = ta_indicators.supersmoother_3_pole(self.close_prices, period=1)
        self.assertEqual(len(result), len(self.close_prices))
        
        
        small_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ta_indicators.supersmoother_3_pole(small_data, period=2)
        self.assertEqual(len(result), len(small_data))
        
        
        
        np.testing.assert_almost_equal(result[0], 1.0)
        np.testing.assert_almost_equal(result[1], 2.0)
        np.testing.assert_almost_equal(result[2], 3.0)
    
    def test_compare_with_rust(self):
        """Compare Python binding results with direct Rust implementation."""
        period = 14
        
        
        py_result = ta_indicators.supersmoother_3_pole(self.close_prices, period=period)
        
        
        self.assertTrue(
            compare_with_rust(
                'supersmoother_3_pole', 
                py_result, 
                params={'period': period},
                rtol=0,
                atol=1e-8
            )
        )
    
if __name__ == '__main__':
    unittest.main()
