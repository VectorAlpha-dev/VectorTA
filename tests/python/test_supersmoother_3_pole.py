#!/usr/bin/env python3
"""Test cases for SuperSmoother 3-Pole indicator Python bindings."""

import unittest
import numpy as np
import my_project as ta_indicators
from test_utils import (
    load_test_data,
    EXPECTED_SUPERSMOOTHER_3_POLE
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
        
    def test_supersmoother_3_pole_accuracy(self):
        """Test SuperSmoother3Pole calculation accuracy against expected values."""
        # Use a standard period
        period = 14
        
        # Calculate SuperSmoother3Pole
        result = ta_indicators.supersmoother_3_pole(self.close_prices, period=period)
        
        # Check output length
        self.assertEqual(len(result), len(self.close_prices))
        
        # For 3-pole supersmoother, the first 3 values are initialized to input values
        # It doesn't have a traditional NaN warmup period like other indicators
        # Check that first 3 values are not NaN (they should match input)
        for i in range(min(3, len(result))):
            self.assertFalse(np.isnan(result[i]), f"Value at index {i} should not be NaN")
        
        # Test last 5 values against expected
        last_5 = result[-5:]
        expected_last_5 = EXPECTED_SUPERSMOOTHER_3_POLE
        
        for i, (actual, expected) in enumerate(zip(last_5, expected_last_5)):
            self.assertAlmostEqual(
                actual, expected, places=6,
                msg=f"Mismatch at position {i}: expected {expected}, got {actual}"
            )
    
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
    
    def test_supersmoother_3_pole_with_nan_handling(self):
        """Test SuperSmoother3Pole with NaN values in input."""
        # Create data with some NaN values
        data_with_nan = self.close_prices.copy()
        data_with_nan[10:15] = np.nan
        
        # Should still compute without error
        result = ta_indicators.supersmoother_3_pole(data_with_nan, period=14)
        
        # Result should be all NaN after the NaN input values until enough valid data
        self.assertTrue(np.all(np.isnan(result[10:30])))
    
    def test_supersmoother_3_pole_with_leading_nans(self):
        """Test SuperSmoother3Pole correctly handles data that starts with NaN values."""
        # Create data starting with NaNs
        data = np.array([np.nan] * 5 + list(range(1, 16)))  # 5 NaNs followed by 1-15
        period = 3
        
        result = ta_indicators.supersmoother_3_pole(data, period=period)
        
        # For 3-pole supersmoother with leading NaNs:
        # The warmup period is first_non_nan + period
        # With 5 leading NaNs and period=3, warmup = 5 + 3 = 8
        # So all values up to index 8 will be NaN
        
        # Check that NaN input produces NaN output
        self.assertTrue(np.all(np.isnan(result[:5])), 
                       "Expected NaN output where input is NaN")
        
        # Due to warmup calculation, values remain NaN until after warmup
        # In this case, all values should be NaN since warmup extends beyond
        # where the filter would normally start producing values
        self.assertTrue(np.all(np.isnan(result[:8])), 
                       "Expected NaN values through warmup period")
    
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
                rtol=1e-9,
                atol=1e-12
            )
        )
    
    def test_supersmoother_3_pole_reinput_consistency(self):
        """Test that re-inputting output produces valid results."""
        # First pass
        period1 = 14
        result1 = ta_indicators.supersmoother_3_pole(self.close_prices, period=period1)
        
        # Second pass with different period
        period2 = 7
        result2 = ta_indicators.supersmoother_3_pole(result1, period=period2)
        
        # Results should be valid
        self.assertEqual(len(result2), len(self.close_prices))
        # Check that we have some non-NaN values
        self.assertTrue(np.any(~np.isnan(result2)))

if __name__ == '__main__':
    unittest.main()