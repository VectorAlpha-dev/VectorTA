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
        # Test with default params (None not applicable for supersmoother_3_pole which only has period)
        result = ta_indicators.supersmoother_3_pole(self.close_prices, 14)  # Using default
        self.assertEqual(len(result), len(self.close_prices))
    
    def test_supersmoother_3_pole_accuracy(self):
        """Test SuperSmoother3Pole matches expected values from Rust tests - mirrors check_supersmoother_3_pole_accuracy"""
        expected = EXPECTED_OUTPUTS['supersmoother_3_pole']
        
        # Calculate SuperSmoother3Pole
        result = ta_indicators.supersmoother_3_pole(
            self.close_prices,
            period=expected['default_params']['period']
        )
        
        # Check output length
        self.assertEqual(len(result), len(self.close_prices))
        
        # Check last 5 values match expected
        # Match Rust test tolerance: absolute <= 1e-8
        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=0,
            atol=1e-8,
            msg="SuperSmoother3Pole last 5 values mismatch"
        )
        
        # Compare full output with Rust
        # Use absolute tolerance matching Rust tests
        compare_with_rust('supersmoother_3_pole', result, 'close', expected['default_params'], rtol=0, atol=1e-8)
    
    def test_supersmoother_3_pole_default_candles(self):
        """Test SuperSmoother3Pole with default parameters - mirrors check_supersmoother_3_pole_default_candles"""
        # Default param: period=14
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
        
        # Batch calculation
        batch_result = ta_indicators.supersmoother_3_pole(self.close_prices, period=period)
        
        # Streaming calculation
        stream = ta_indicators.SuperSmoother3PoleStream(period=period)
        stream_results = []
        
        for price in self.close_prices:
            val = stream.update(price)
            stream_results.append(val)
        
        stream_results = np.array(stream_results)
        
        # Compare results (allowing for small numerical differences)
        valid_mask = ~np.isnan(batch_result) & ~np.isnan(stream_results)
        if np.any(valid_mask):
            max_diff = np.max(np.abs(batch_result[valid_mask] - stream_results[valid_mask]))
            self.assertLess(max_diff, 1e-10, f"Streaming vs batch max difference: {max_diff}")
    
    def test_supersmoother_3_pole_batch_processing(self):
        """Test batch processing for multiple periods."""
        # Test range of periods
        periods = ta_indicators.supersmoother_3_pole_batch(
            self.close_prices,
            period_range=(10, 20, 5)  # periods: 10, 15, 20
        )
        
        # Check return structure
        self.assertIn('values', periods)
        self.assertIn('periods', periods)
        
        # Verify shape
        self.assertEqual(periods['values'].shape[0], 3)  # 3 different periods
        self.assertEqual(periods['values'].shape[1], len(self.close_prices))
        
        # Verify periods array
        np.testing.assert_array_equal(periods['periods'], [10, 15, 20])
        
        # Test that each row has proper initialization
        # 3-pole supersmoother initializes first 3 values to input, no NaN warmup
        for i, period in enumerate([10, 15, 20]):
            row = periods['values'][i]
            # Check that first 3 values are not NaN
            for j in range(min(3, len(row))):
                self.assertFalse(np.isnan(row[j]), f"Value at index {j} for period {period} should not be NaN")
    
    def test_supersmoother_3_pole_nan_handling(self):
        """Test SuperSmoother3Pole handles NaN values correctly - mirrors check_supersmoother_3_pole_nan_handling"""
        result = ta_indicators.supersmoother_3_pole(self.close_prices, period=14)
        self.assertEqual(len(result), len(self.close_prices))
        
        # After warmup period (240), no NaN values should exist
        if len(result) > 240:
            self.assertFalse(np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period")
        
        # SuperSmoother 3-pole initializes first 3 values to input, so no NaN warmup
        # Check that first 3 values are not NaN
        for i in range(min(3, len(result))):
            self.assertFalse(np.isnan(result[i]), f"Value at index {i} should not be NaN")
    
    def test_supersmoother_3_pole_with_leading_nans(self):
        """Test SuperSmoother3Pole correctly handles data that starts with NaN values."""
        # Create data starting with NaNs
        data = np.array([np.nan] * 5 + list(range(1, 16)))  # 5 NaNs followed by 1-15
        period = 3
        
        result = ta_indicators.supersmoother_3_pole(data, period=period)
        
        # SuperSmoother 3-pole behavior with leading NaNs:
        # - NaN values are preserved in output where input is NaN
        # - Once valid data starts, the first 3 values are passed through
        # - Filter calculation begins after the first 3 valid values
        
        # Check that NaN input produces NaN output
        self.assertTrue(np.all(np.isnan(result[:5])), 
                       "Expected NaN output where input is NaN")
        
        # The first 3 valid values (1, 2, 3) are passed through at indices 5, 6, 7
        np.testing.assert_almost_equal(result[5], 1.0, decimal=10)
        np.testing.assert_almost_equal(result[6], 2.0, decimal=10)
        np.testing.assert_almost_equal(result[7], 3.0, decimal=10)
        
        # Filter calculation starts from index 8
        self.assertFalse(np.isnan(result[8]), "Filter should start calculating at index 8")
        self.assertNotEqual(result[8], 4.0, "Index 8 should be filtered, not passed through")
    
    def test_supersmoother_3_pole_kernel_selection(self):
        """Test different kernel options produce consistent results."""
        period = 14
        
        # Test different kernels
        result_auto = ta_indicators.supersmoother_3_pole(self.close_prices, period=period, kernel='auto')
        result_scalar = ta_indicators.supersmoother_3_pole(self.close_prices, period=period, kernel='scalar')
        
        # Results should be very close (within floating point precision)
        valid_mask = ~np.isnan(result_auto) & ~np.isnan(result_scalar)
        if np.any(valid_mask):
            max_diff = np.max(np.abs(result_auto[valid_mask] - result_scalar[valid_mask]))
            self.assertLess(max_diff, 1e-10, f"Kernel results differ by {max_diff}")
    
    def test_supersmoother_3_pole_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with period = 0
        with pytest.raises(ValueError) as exc_info:
            ta_indicators.supersmoother_3_pole(self.close_prices, period=0)
        self.assertIn("Invalid period", str(exc_info.value))
        
        # Test with period > data length
        with pytest.raises(ValueError) as exc_info:
            ta_indicators.supersmoother_3_pole(self.close_prices[:5], period=10)
        self.assertIn("Invalid period", str(exc_info.value))
        
        # Test with all NaN data
        with pytest.raises(ValueError) as exc_info:
            ta_indicators.supersmoother_3_pole(np.full(10, np.nan), period=5)
        self.assertIn("All values are NaN", str(exc_info.value))
    
    def test_supersmoother_3_pole_edge_cases(self):
        """Test edge cases for SuperSmoother3Pole."""
        # Test with minimum period (1)
        result = ta_indicators.supersmoother_3_pole(self.close_prices, period=1)
        self.assertEqual(len(result), len(self.close_prices))
        
        # Test with very small dataset
        small_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ta_indicators.supersmoother_3_pole(small_data, period=2)
        self.assertEqual(len(result), len(small_data))
        
        # Verify first few values match expectations for 3-pole filter
        # The first 3 values should be the input values (initial conditions)
        np.testing.assert_almost_equal(result[0], 1.0)
        np.testing.assert_almost_equal(result[1], 2.0)
        np.testing.assert_almost_equal(result[2], 3.0)
    
    def test_compare_with_rust(self):
        """Compare Python binding results with direct Rust implementation."""
        period = 14
        
        # Get result from Python binding
        py_result = ta_indicators.supersmoother_3_pole(self.close_prices, period=period)
        
        # Compare with Rust implementation
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
