"""
Python binding tests for AO (Awesome Oscillator) indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""

import pytest
import numpy as np
import my_project
from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


class TestAo:
    def test_ao_accuracy(self, test_data):
        """Test that AO values match expected results from Rust tests."""
        high = test_data['high']
        low = test_data['low']
        expected = EXPECTED_OUTPUTS['ao']
        
        # Calculate with default parameters
        result = my_project.ao(
            high, low,
            expected['default_params']['short_period'],
            expected['default_params']['long_period']
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == high.shape
        
        # Check last 5 values match expected with tolerance
        assert_close(
            result[-5:], 
            expected['last_5_values'], 
            rtol=1e-4,
            msg="AO last 5 values mismatch"
        )

    def test_ao_with_different_params(self, test_data):
        """Test AO with non-default parameters."""
        high = test_data['high']
        low = test_data['low']
        
        # Test with different parameters
        result = my_project.ao(high, low, 3, 10)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == high.shape
        
        # Check warmup period (should be NaN)
        warmup_period = 10  # long_period
        for i in range(warmup_period - 1):
            assert np.isnan(result[i]), f"Expected NaN at index {i} during warmup"
        
        # Check that values after warmup are not NaN
        for i in range(warmup_period, len(result)):
            assert not np.isnan(result[i]), f"Unexpected NaN at index {i}"

    def test_ao_empty_input(self, test_data):
        """Test AO with empty input arrays."""
        empty = np.array([], dtype=np.float64)
        
        with pytest.raises(ValueError, match="empty|no data"):
            my_project.ao(empty, empty, 5, 34)

    def test_ao_mismatched_lengths(self, test_data):
        """Test AO with mismatched high/low array lengths."""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0])  # Different length
        
        with pytest.raises(ValueError, match="same length"):
            my_project.ao(high, low, 5, 34)

    def test_ao_invalid_periods(self, test_data):
        """Test AO with invalid period parameters."""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        
        # Zero short period
        with pytest.raises(ValueError, match="Invalid periods"):
            my_project.ao(high, low, 0, 34)
        
        # Zero long period
        with pytest.raises(ValueError, match="Invalid periods"):
            my_project.ao(high, low, 5, 0)
        
        # Short period >= long period
        with pytest.raises(ValueError, match="Short period must be less than long period"):
            my_project.ao(high, low, 34, 34)
        
        with pytest.raises(ValueError, match="Short period must be less than long period"):
            my_project.ao(high, low, 35, 34)

    def test_ao_period_exceeds_length(self, test_data):
        """Test AO when period exceeds data length."""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([5.0, 15.0, 25.0])
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            my_project.ao(high, low, 5, 10)  # long_period=10 > length=3

    def test_ao_all_nan(self):
        """Test AO with all NaN values."""
        all_nan = np.full(10, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            my_project.ao(all_nan, all_nan, 5, 34)

    def test_ao_with_leading_nans(self, test_data):
        """Test AO with leading NaN values."""
        high = test_data['high'].copy()
        low = test_data['low'].copy()
        
        # Add some leading NaNs
        high[:5] = np.nan
        low[:5] = np.nan
        
        result = my_project.ao(high, low, 5, 34)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == high.shape
        
        # First 5 + warmup period should be NaN
        for i in range(5 + 34 - 1):
            assert np.isnan(result[i])

    def test_ao_kernel_selection(self, test_data):
        """Test AO with different kernel selections."""
        high = test_data['high']
        low = test_data['low']
        
        # Test with explicit scalar kernel
        result_scalar = my_project.ao(high, low, 5, 34, kernel="scalar")
        
        # Test with auto kernel (default)
        result_auto = my_project.ao(high, low, 5, 34)
        
        # Results should be very close regardless of kernel
        assert_close(result_scalar, result_auto, rtol=1e-10)

    def test_ao_streaming(self, test_data):
        """Test AO streaming functionality."""
        high = test_data['high']
        low = test_data['low']
        
        # Batch calculation
        batch_result = my_project.ao(high, low, 5, 34)
        
        # Streaming calculation
        stream = my_project.AoStream(5, 34)
        stream_results = []
        
        for h, l in zip(high, low):
            val = stream.update(h, l)
            stream_results.append(val if val is not None else np.nan)
        
        stream_results = np.array(stream_results)
        
        # Compare results (streaming should match batch)
        assert_close(stream_results, batch_result, rtol=1e-10,
                    msg="Streaming vs batch mismatch")

    def test_ao_re_input(self, test_data):
        """Test using AO output as input to another calculation."""
        high = test_data['high']
        low = test_data['low']
        
        # First AO calculation
        first_result = my_project.ao(high, low, 5, 34)
        
        # Use output as input (treating as both high and low)
        # This tests that the output is properly formatted
        second_result = my_project.ao(first_result, first_result, 3, 10)
        
        assert isinstance(second_result, np.ndarray)
        assert second_result.shape == first_result.shape

    def test_ao_constant_price(self):
        """Test AO with constant prices."""
        length = 50
        constant_price = 100.0
        high = np.full(length, constant_price)
        low = np.full(length, constant_price)
        
        result = my_project.ao(high, low, 5, 34)
        
        # With constant prices, AO should be 0 after warmup
        warmup = 34
        for i in range(warmup, length):
            assert abs(result[i]) < 1e-10, f"Expected 0 at index {i}, got {result[i]}"

    def test_ao_trending_market(self):
        """Test AO in a trending market."""
        length = 100
        # Create uptrending data
        high = np.linspace(100, 200, length) + 5  # High is price + 5
        low = np.linspace(100, 200, length) - 5   # Low is price - 5
        
        result = my_project.ao(high, low, 5, 34)
        
        # In a strong uptrend, AO should be positive after initial period
        # Check last 10 values are positive
        assert all(result[-10:] > 0), "AO should be positive in uptrend"

    def test_ao_batch_single_params(self, test_data):
        """Test AO batch calculation with single parameter set."""
        high = test_data['high']
        low = test_data['low']
        
        # Single parameter set (acts like regular ao)
        result = my_project.ao_batch(
            high, low,
            (5, 5, 0),   # short_period range
            (34, 34, 0)  # long_period range
        )
        
        assert 'values' in result
        assert 'short_periods' in result
        assert 'long_periods' in result
        
        values = result['values']
        assert values.shape == (1, len(high))  # 1 combination
        
        # Should match single calculation
        single_result = my_project.ao(high, low, 5, 34)
        assert_close(values[0], single_result, rtol=1e-10)

    def test_ao_batch_parameter_sweep(self, test_data):
        """Test AO batch calculation with parameter ranges."""
        high = test_data['high'][:100]  # Use smaller subset for speed
        low = test_data['low'][:100]
        
        result = my_project.ao_batch(
            high, low,
            (3, 7, 2),    # short: 3, 5, 7
            (20, 30, 5)   # long: 20, 25, 30
        )
        
        values = result['values']
        short_periods = result['short_periods']
        long_periods = result['long_periods']
        
        # Should have 3 * 3 = 9 valid combinations
        assert values.shape[0] == 9
        assert len(short_periods) == 9
        assert len(long_periods) == 9
        
        # Verify all combinations are present and valid
        expected_combos = [
            (3, 20), (3, 25), (3, 30),
            (5, 20), (5, 25), (5, 30),
            (7, 20), (7, 25), (7, 30)
        ]
        
        actual_combos = list(zip(short_periods, long_periods))
        for combo in expected_combos:
            assert combo in actual_combos

    def test_ao_batch_invalid_combinations(self, test_data):
        """Test AO batch with invalid parameter combinations."""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        
        # Range where some short >= long (should be filtered out)
        result = my_project.ao_batch(
            high, low,
            (5, 15, 5),   # short: 5, 10, 15
            (10, 12, 2)   # long: 10, 12
        )
        
        values = result['values']
        short_periods = result['short_periods']
        long_periods = result['long_periods']
        
        # Only valid combos where short < long
        # Valid: (5,10), (5,12)
        assert values.shape[0] == 2
        assert list(zip(short_periods, long_periods)) == [(5, 10), (5, 12)]

    def test_ao_error_coverage(self, test_data):
        """Test all error enum variants are covered."""
        high = test_data['high']
        low = test_data['low']
        
        # AllValuesNaN
        with pytest.raises(ValueError, match="All values are NaN"):
            my_project.ao(np.full(10, np.nan), np.full(10, np.nan), 5, 34)
        
        # InvalidPeriods (short=0 or long=0)
        with pytest.raises(ValueError, match="Invalid periods"):
            my_project.ao(high[:50], low[:50], 0, 34)
            
        # ShortPeriodNotLess
        with pytest.raises(ValueError, match="Short period must be less than long period"):
            my_project.ao(high[:50], low[:50], 34, 34)
            
        # NoData (empty)
        with pytest.raises(ValueError, match="empty|no data"):
            my_project.ao(np.array([]), np.array([]), 5, 34)
            
        # NotEnoughValidData
        with pytest.raises(ValueError, match="Not enough valid data"):
            my_project.ao(high[:10], low[:10], 5, 34)

    def test_ao_real_world_conditions(self, test_data):
        """Test AO under real-world conditions."""
        high = test_data['high']
        low = test_data['low']
        
        result = my_project.ao(high, low, 5, 34)
        
        # Check warmup period behavior
        warmup = 34  # long_period
        
        # Values before warmup should be NaN
        assert all(np.isnan(result[:warmup-1]))
        
        # Values from warmup onwards should not be NaN
        valid_start = warmup - 1
        assert not any(np.isnan(result[valid_start:]))
        
        # Check output properties
        assert len(result) == len(high)
        assert result.dtype == np.float64
        
        # AO typically oscillates around zero
        # Check that we have both positive and negative values
        valid_values = result[valid_start:]
        assert any(valid_values > 0), "Should have some positive values"
        assert any(valid_values < 0), "Should have some negative values"