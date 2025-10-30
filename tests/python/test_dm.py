"""
Python binding tests for DM (Directional Movement) indicator.
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


class TestDm:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_dm_partial_params(self, test_data):
        """Test DM with partial parameters (None values) - mirrors check_dm_partial_params"""
        high = test_data['high']
        low = test_data['low']
        
        # Test with default params (period=14)
        plus, minus = ta_indicators.dm(high, low, 14)
        assert len(plus) == len(high)
        assert len(minus) == len(low)
    
    def test_dm_accuracy(self, test_data):
        """Test DM matches expected values from Rust tests - mirrors check_dm_known_values"""
        high = test_data['high']
        low = test_data['low']
        
        # Expected values from Rust test check_dm_known_values
        expected_plus = [
            1410.819956368491,
            1384.04710234217,
            1285.186595032015,
            1199.3875525297283,
            1113.7170130633192,
        ]
        expected_minus = [
            3602.8631384045057,
            3345.5157713756125,
            3258.5503591344973,
            3025.796762053462,
            3493.668421906786,
        ]
        
        plus, minus = ta_indicators.dm(high, low, period=14)
        
        assert len(plus) == len(high)
        assert len(minus) == len(low)
        
        # Check last 5 values match expected
        # Match Rust tolerance: absolute diff <= 1e-6 (no looser than Rust)
        assert_close(
            plus[-5:], 
            expected_plus,
            rtol=0.0,
            atol=1e-6,
            msg="DM plus last 5 values mismatch"
        )
        assert_close(
            minus[-5:], 
            expected_minus,
            rtol=0.0,
            atol=1e-6,
            msg="DM minus last 5 values mismatch"
        )
    
    def test_dm_default_candles(self, test_data):
        """Test DM with default parameters - mirrors check_dm_default_candles"""
        high = test_data['high']
        low = test_data['low']
        
        # Default params: period=14
        plus, minus = ta_indicators.dm(high, low, 14)
        assert len(plus) == len(high)
        assert len(minus) == len(low)
    
    def test_dm_zero_period(self):
        """Test DM fails with zero period - mirrors check_dm_zero_period"""
        high_data = np.array([100.0, 110.0, 120.0])
        low_data = np.array([90.0, 100.0, 110.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.dm(high_data, low_data, period=0)
    
    def test_dm_period_exceeds_length(self):
        """Test DM fails when period exceeds data length - mirrors check_dm_period_exceeds_length"""
        high_small = np.array([100.0, 110.0, 120.0])
        low_small = np.array([90.0, 100.0, 110.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.dm(high_small, low_small, period=10)
    
    def test_dm_very_small_dataset(self):
        """Test DM fails with insufficient data - mirrors check_dm_period_exceeds_length"""
        single_high = np.array([42.0])
        single_low = np.array([40.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.dm(single_high, single_low, period=14)
    
    def test_dm_empty_input(self):
        """Test DM fails with empty input - mirrors check_dm_all_values_nan"""
        empty_high = np.array([])
        empty_low = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.dm(empty_high, empty_low, period=14)
    
    def test_dm_not_enough_valid_data(self):
        """Test DM fails when not enough valid data - mirrors check_dm_not_enough_valid_data"""
        high_data = np.array([np.nan, np.nan, 100.0, 101.0, 102.0])
        low_data = np.array([np.nan, np.nan, 90.0, 89.0, 88.0])
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.dm(high_data, low_data, period=5)
    
    def test_dm_all_nan_input(self):
        """Test DM with all NaN values - mirrors check_dm_all_values_nan"""
        all_nan_high = np.full(100, np.nan)
        all_nan_low = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.dm(all_nan_high, all_nan_low, period=14)
    
    def test_dm_nan_handling(self, test_data):
        """Test DM handles NaN values correctly - mirrors check_dm_no_poison"""
        high = test_data['high']
        low = test_data['low']
        
        plus, minus = ta_indicators.dm(high, low, period=14)
        assert len(plus) == len(high)
        assert len(minus) == len(low)
        
        # First period-1 values should be NaN (warmup period)
        # For DM, warmup = first_valid_idx + period - 1
        # Since our test data doesn't have leading NaN, first_valid_idx = 0
        # So warmup = 0 + 14 - 1 = 13
        assert np.all(np.isnan(plus[:13])), "Expected NaN in warmup period for plus"
        assert np.all(np.isnan(minus[:13])), "Expected NaN in warmup period for minus"
        
        # After warmup period, no NaN values should exist
        if len(plus) > 240:
            assert not np.any(np.isnan(plus[240:])), "Found unexpected NaN after warmup period in plus"
            assert not np.any(np.isnan(minus[240:])), "Found unexpected NaN after warmup period in minus"
    
    def test_dm_streaming(self, test_data):
        """Test DM streaming matches batch calculation"""
        high = test_data['high'][:100]  # Use smaller dataset for speed
        low = test_data['low'][:100]
        period = 14
        
        # Batch calculation
        batch_plus, batch_minus = ta_indicators.dm(high, low, period=period)
        
        # Streaming calculation
        stream = ta_indicators.DmStream(period=period)
        stream_plus = []
        stream_minus = []
        
        for h, l in zip(high, low):
            result = stream.update(h, l)
            if result is not None:
                stream_plus.append(result[0])
                stream_minus.append(result[1])
            else:
                stream_plus.append(np.nan)
                stream_minus.append(np.nan)
        
        stream_plus = np.array(stream_plus)
        stream_minus = np.array(stream_minus)
        
        # Compare batch vs streaming
        assert len(batch_plus) == len(stream_plus)
        assert len(batch_minus) == len(stream_minus)
        
        # Compare values where both are not NaN
        for i, (bp, sp, bm, sm) in enumerate(zip(batch_plus, stream_plus, batch_minus, stream_minus)):
            if np.isnan(bp) and np.isnan(sp):
                continue
            assert_close(bp, sp, rtol=1e-9, atol=1e-9, 
                        msg=f"DM plus streaming mismatch at index {i}")
            assert_close(bm, sm, rtol=1e-9, atol=1e-9,
                        msg=f"DM minus streaming mismatch at index {i}")
    
    def test_dm_batch_single_params(self, test_data):
        """Test DM batch processing with single parameter - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        
        result = ta_indicators.dm_batch(
            high, low,
            period_range=(14, 14, 0),  # Default period only
        )
        
        assert 'plus' in result
        assert 'minus' in result
        assert 'periods' in result
        
        # Should have 1 combination (default params)
        assert result['plus'].shape[0] == 1
        assert result['plus'].shape[1] == len(high)
        assert result['minus'].shape[0] == 1
        assert result['minus'].shape[1] == len(low)
        
        # Extract the single row
        default_plus = result['plus'][0]
        default_minus = result['minus'][0]
        
        expected_plus = [
            1410.819956368491,
            1384.04710234217,
            1285.186595032015,
            1199.3875525297283,
            1113.7170130633192,
        ]
        expected_minus = [
            3602.8631384045057,
            3345.5157713756125,
            3258.5503591344973,
            3025.796762053462,
            3493.668421906786,
        ]
        
        # Check last 5 values match
        # Match Rust tolerance: absolute diff <= 1e-6 (no looser than Rust)
        assert_close(
            default_plus[-5:],
            expected_plus,
            rtol=0.0,
            atol=1e-6,
            msg="DM batch default plus row mismatch"
        )
        assert_close(
            default_minus[-5:],
            expected_minus,
            rtol=0.0,
            atol=1e-6,
            msg="DM batch default minus row mismatch"
        )
    
    def test_dm_batch_multiple_periods(self, test_data):
        """Test DM batch with multiple period values"""
        high = test_data['high'][:100]  # Use smaller dataset for speed
        low = test_data['low'][:100]
        
        # Multiple periods: 10, 14, 20
        result = ta_indicators.dm_batch(
            high, low,
            period_range=(10, 20, 5),  # 10, 15, 20
        )
        
        assert 'plus' in result
        assert 'minus' in result
        assert 'periods' in result
        
        # Should have 3 combinations
        assert result['plus'].shape[0] == 3
        assert result['plus'].shape[1] == 100
        assert result['minus'].shape[0] == 3
        assert result['minus'].shape[1] == 100
        
        # Verify periods
        assert np.array_equal(result['periods'], [10, 15, 20])
        
        # Verify each row matches individual calculation
        periods = [10, 15, 20]
        for i, period in enumerate(periods):
            single_plus, single_minus = ta_indicators.dm(high, low, period)
            
            assert_close(
                result['plus'][i], 
                single_plus, 
                rtol=1e-10, 
                msg=f"Batch plus period {period} mismatch"
            )
            assert_close(
                result['minus'][i], 
                single_minus, 
                rtol=1e-10, 
                msg=f"Batch minus period {period} mismatch"
            )
    
    def test_dm_tuple_return(self, test_data):
        """Test DM returns (plus, minus) tuple correctly"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        
        result = ta_indicators.dm(high, low, period=14)
        
        # Should return a tuple
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        plus, minus = result
        
        # Both should be numpy arrays
        assert isinstance(plus, np.ndarray)
        assert isinstance(minus, np.ndarray)
        
        # Both should have same length as input
        assert len(plus) == len(high)
        assert len(minus) == len(low)
        
        # Check warmup period
        assert np.all(np.isnan(plus[:13]))
        assert np.all(np.isnan(minus[:13]))
        
        # After warmup should have values
        assert not np.any(np.isnan(plus[13:]))
        assert not np.any(np.isnan(minus[13:]))
    
    def test_dm_mismatched_lengths(self):
        """Test DM fails when high/low lengths differ"""
        high = np.array([100.0, 110.0, 120.0, 130.0])
        low = np.array([90.0, 100.0, 110.0])  # Different length
        
        with pytest.raises(ValueError, match="high/low length mismatch"):
            ta_indicators.dm(high, low, period=2)
    
    def test_dm_with_slice_data(self):
        """Test DM with slice data - mirrors check_dm_with_slice_data"""
        high_values = np.array([8000.0, 8050.0, 8100.0, 8075.0, 8110.0, 8050.0])
        low_values = np.array([7800.0, 7900.0, 7950.0, 7950.0, 8000.0, 7950.0])
        
        plus, minus = ta_indicators.dm(high_values, low_values, period=3)
        
        assert len(plus) == 6
        assert len(minus) == 6
        
        # First period-1 values should be NaN
        for i in range(2):
            assert np.isnan(plus[i])
            assert np.isnan(minus[i])
    
    def test_dm_batch_edge_cases(self, test_data):
        """Test edge cases for batch processing"""
        high = test_data['high'][:20]
        low = test_data['low'][:20]
        
        # Single value sweep
        single_batch = ta_indicators.dm_batch(
            high, low,
            period_range=(14, 14, 1)
        )
        
        assert single_batch['plus'].shape[0] == 1
        assert single_batch['minus'].shape[0] == 1
        assert len(single_batch['periods']) == 1
        
        # Step larger than range
        large_batch = ta_indicators.dm_batch(
            high, low,
            period_range=(5, 7, 10)  # Step larger than range
        )
        
        # Should only have period=5
        assert large_batch['plus'].shape[0] == 1
        assert large_batch['minus'].shape[0] == 1
        assert large_batch['periods'][0] == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
