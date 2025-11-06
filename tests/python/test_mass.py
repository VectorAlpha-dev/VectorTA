"""
Python binding tests for Mass Index indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

try:
    import my_project as ta_indicators
except ImportError:
    # If not in virtual environment, try to import from installed location
    try:
        import my_project as ta_indicators
    except ImportError:
        pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust


class TestMass:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_mass_partial_params(self, test_data):
        """Test Mass Index with partial parameters - mirrors check_mass_partial_params"""
        high = test_data['high']
        low = test_data['low']
        
        # Test with default period (5)
        result = ta_indicators.mass(high, low, 5)
        assert len(result) == len(high)
    
    def test_mass_accuracy(self, test_data):
        """Test Mass Index matches expected values from Rust tests - mirrors check_mass_accuracy"""
        high = test_data['high']
        low = test_data['low']
        expected = EXPECTED_OUTPUTS['mass']
        
        result = ta_indicators.mass(
            high,
            low,
            period=expected['default_params']['period']
        )
        
        assert len(result) == len(high)
        
        # Check last 5 values match expected
        # Rust test uses absolute tolerance of 1e-7; match or tighten (no looser than Rust)
        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=0.0,
            atol=1e-7,
            msg="Mass Index last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('mass', result, 'high,low', expected['default_params'])
    
    def test_mass_default_candles(self, test_data):
        """Test Mass Index with default parameters - mirrors check_mass_default_candles"""
        high = test_data['high']
        low = test_data['low']
        
        # Default period is 5
        result = ta_indicators.mass(high, low, 5)
        assert len(result) == len(high)
    
    def test_mass_zero_period(self):
        """Test Mass Index fails with zero period - mirrors check_mass_zero_period"""
        high_data = np.array([10.0, 15.0, 20.0])
        low_data = np.array([5.0, 10.0, 12.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.mass(high_data, low_data, period=0)
    
    def test_mass_period_exceeds_length(self):
        """Test Mass Index fails when period exceeds data length - mirrors check_mass_period_exceeds_length"""
        high_data = np.array([10.0, 15.0, 20.0])
        low_data = np.array([5.0, 10.0, 12.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.mass(high_data, low_data, period=10)
    
    def test_mass_very_small_dataset(self):
        """Test Mass Index fails with insufficient data - mirrors check_mass_very_small_data_set"""
        high_data = np.array([10.0])
        low_data = np.array([5.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.mass(high_data, low_data, period=5)
    
    def test_mass_empty_input(self):
        """Test Mass Index fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.mass(empty, empty, period=5)
    
    def test_mass_different_length_hl(self):
        """Test Mass Index fails when high and low have different lengths"""
        high_data = np.array([10.0, 15.0, 20.0])
        low_data = np.array([5.0, 10.0])
        
        with pytest.raises(ValueError, match="High and low"):
            ta_indicators.mass(high_data, low_data, period=2)
    
    def test_mass_reinput(self, test_data):
        """Test Mass Index applied twice (re-input) - mirrors check_mass_reinput"""
        high = test_data['high']
        low = test_data['low']
        
        # First pass
        first_result = ta_indicators.mass(high, low, period=5)
        assert len(first_result) == len(high)
        
        # Second pass - apply Mass to Mass output
        second_result = ta_indicators.mass(first_result, first_result, period=5)
        assert len(second_result) == len(first_result)
    
    def test_mass_nan_handling(self, test_data):
        """Test Mass Index handles NaN values correctly - mirrors check_mass_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        expected = EXPECTED_OUTPUTS['mass']
        
        result = ta_indicators.mass(high, low, period=5)
        assert len(result) == len(high)
        
        # Check warmup period has NaN values (first 20 values for period=5)
        warmup = expected['warmup_period']
        assert np.all(np.isnan(result[:warmup])), f"Expected NaN in first {warmup} values (warmup period)"
        
        # After warmup, values should be valid
        assert not np.isnan(result[warmup]), f"Expected valid value at index {warmup} (first valid index)"
        
        # After index 240, no NaN values should exist
        if len(result) > 240:
            for i in range(240, len(result)):
                assert not np.isnan(result[i]), f"Found unexpected NaN at index {i}"
    
    def test_mass_batch_single_parameter(self, test_data):
        """Test Mass batch with single parameter combination"""
        high = test_data['high']
        low = test_data['low']
        
        # Single period
        batch_result = ta_indicators.mass_batch(
            high, 
            low,
            period_range=(5, 5, 1)
        )
        
        # Should match single calculation
        single_result = ta_indicators.mass(high, low, 5)
        
        assert batch_result['values'].shape == (1, len(high))
        assert_close(
            batch_result['values'][0], 
            single_result,
            rtol=1e-10,
            msg="Batch vs single calculation mismatch"
        )
    
    def test_mass_batch_multiple_periods(self, test_data):
        """Test Mass batch with multiple period values"""
        high = test_data['high'][:100]  # Use smaller dataset for speed
        low = test_data['low'][:100]
        
        # Multiple periods: 5, 10, 15
        batch_result = ta_indicators.mass_batch(
            high,
            low,
            period_range=(5, 15, 5)
        )
        
        # Should have 3 rows * 100 cols
        assert batch_result['values'].shape == (3, 100)
        assert len(batch_result['periods']) == 3
        
        # Verify each row matches individual calculation
        periods = [5, 10, 15]
        for i, period in enumerate(periods):
            single_result = ta_indicators.mass(high, low, period)
            assert_close(
                batch_result['values'][i], 
                single_result, 
                rtol=1e-10,
                msg=f"Period {period} mismatch"
            )
    
    def test_mass_batch_edge_cases(self, test_data):
        """Test edge cases for batch processing"""
        # Mass needs at least 16 + period - 1 data points
        # For period=3, that's 18 points minimum, so use 20 to be safe
        high = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.float64)
        low = high * 0.9
        
        # Single value sweep
        single_batch = ta_indicators.mass_batch(
            high,
            low,
            period_range=(3, 3, 1)
        )
        
        assert single_batch['values'].shape == (1, 20)
        assert len(single_batch['periods']) == 1
        
        # Step larger than range
        large_batch = ta_indicators.mass_batch(
            high,
            low,
            period_range=(3, 5, 10)  # Step larger than range
        )
        
        # Should only have period=3
        assert large_batch['values'].shape == (1, 20)
        assert len(large_batch['periods']) == 1
    
    def test_mass_streaming(self):
        """Test Mass streaming functionality"""
        # Create a streaming instance
        stream = ta_indicators.MassStream(period=5)
        
        # Test data
        high_data = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0,
                     60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0,
                     110.0]
        low_data = [h * 0.9 for h in high_data]
        
        results = []
        for h, l in zip(high_data, low_data):
            result = stream.update(h, l)
            results.append(result)
        
        # First 20 values should be None (warmup period = 16 + period - 1 = 20)
        for i in range(20):
            assert results[i] is None, f"Expected None at index {i}, got {results[i]}"
        
        # After warmup, should have values
        assert results[20] is not None
        assert isinstance(results[20], float)
    
    def test_mass_kernel_parameter(self, test_data):
        """Test Mass with different kernel parameters"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.mass(high, low, 5)
        
        # Test with scalar kernel
        result_scalar = ta_indicators.mass(high, low, 5, kernel='scalar')
        
        # Results should be very close
        assert_close(
            result_auto,
            result_scalar,
            rtol=1e-10,
            msg="Auto vs Scalar kernel mismatch"
        )
    
    def test_mass_all_nan_input(self):
        """Test Mass Index with all NaN values - mirrors ALMA test coverage"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.mass(all_nan, all_nan, period=5)
    
    def test_mass_batch_metadata(self, test_data):
        """Test Mass batch includes correct metadata - mirrors ALMA batch tests"""
        high = test_data['high'][:50]  # Use smaller dataset
        low = test_data['low'][:50]
        
        # Test with multiple periods
        batch_result = ta_indicators.mass_batch(
            high,
            low,
            period_range=(5, 10, 5)  # periods: 5, 10
        )
        
        # Verify metadata structure
        assert 'values' in batch_result
        assert 'periods' in batch_result
        assert batch_result['values'].shape == (2, 50)
        assert len(batch_result['periods']) == 2
        assert list(batch_result['periods']) == [5, 10]
        
        # Verify each row has correct warmup
        for i, period in enumerate([5, 10]):
            row = batch_result['values'][i]
            warmup = 16 + period - 1
            # Check warmup period has NaN
            assert np.all(np.isnan(row[:warmup])), f"Expected NaN in warmup for period {period}"
            # Check after warmup has values
            if warmup < len(row):
                assert not np.isnan(row[warmup]), f"Expected valid value after warmup for period {period}"
    
    def test_mass_batch_parameter_sweep(self, test_data):
        """Test Mass batch with full parameter sweep"""
        high = test_data['high'][:30]  # Small dataset for testing
        low = test_data['low'][:30]
        
        # Test sweep with step
        batch_result = ta_indicators.mass_batch(
            high,
            low,
            period_range=(3, 7, 2)  # periods: 3, 5, 7
        )
        
        assert batch_result['values'].shape == (3, 30)
        assert list(batch_result['periods']) == [3, 5, 7]
        
        # Verify each row matches single calculation
        for i, period in enumerate([3, 5, 7]):
            single = ta_indicators.mass(high, low, period)
            assert_close(
                batch_result['values'][i],
                single,
                rtol=1e-10,
                msg=f"Batch row {i} (period={period}) doesn't match single calculation"
            )
