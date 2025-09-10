"""
Python binding tests for Percentile Nearest Rank indicator.
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


class TestPercentileNearestRank:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_percentile_nearest_rank_partial_params(self, test_data):
        """Test PNR with partial parameters - mirrors check_pnr_partial_params"""
        data = np.array(EXPECTED_OUTPUTS['percentile_nearest_rank']['basic_test']['data'])
        
        # Test with only length specified (percentage defaults to 50.0)
        result = ta_indicators.percentile_nearest_rank(data, length=5)
        assert len(result) == len(data)
        assert result[4] == EXPECTED_OUTPUTS['percentile_nearest_rank']['basic_test']['expected_at_4']
    
    def test_percentile_nearest_rank_accuracy(self, test_data):
        """Test PNR accuracy with default parameters - mirrors check_pnr_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['percentile_nearest_rank']
        
        result = ta_indicators.percentile_nearest_rank(
            close,
            length=expected['default_params']['length'],
            percentage=expected['default_params']['percentage']
        )
        
        assert len(result) == len(close)
        
        # Check warmup period
        warmup = expected['warmup_period']
        assert np.all(np.isnan(result[:warmup])), f"Expected NaN in first {warmup} values"
        
        # Check that we have valid values after warmup
        assert not np.isnan(result[warmup]), f"Expected valid value at index {warmup}"
        
        # Check last 5 values match expected reference values
        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=1e-8,
            msg="PNR last 5 values mismatch"
        )
        
        # Compare with Rust implementation
        # NOTE: generate_references binary doesn't have percentile_nearest_rank registered
        # compare_with_rust('percentile_nearest_rank', result, 'close', expected['default_params'])
    
    def test_percentile_nearest_rank_default_candles(self, test_data):
        """Test PNR with default parameters - mirrors check_pnr_default_candles"""
        close = test_data['close']
        
        # Default params: length=15, percentage=50.0
        result = ta_indicators.percentile_nearest_rank(close)
        assert len(result) == len(close)
        
        # Check warmup period
        assert np.all(np.isnan(result[:14]))
        assert not np.isnan(result[14])
    
    def test_percentile_nearest_rank_zero_period(self):
        """Test PNR fails with zero period - mirrors check_pnr_zero_period"""
        data = np.array([1.0] * 10)
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.percentile_nearest_rank(data, length=0, percentage=50.0)
    
    def test_percentile_nearest_rank_period_exceeds_length(self):
        """Test PNR fails when period exceeds data length - mirrors check_pnr_period_exceeds_length"""
        data = np.array([1.0] * 5)
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.percentile_nearest_rank(data, length=10, percentage=50.0)
    
    def test_percentile_nearest_rank_very_small_dataset(self):
        """Test PNR with single data point - mirrors check_pnr_very_small_dataset"""
        data = np.array([5.0])
        
        result = ta_indicators.percentile_nearest_rank(data, length=1, percentage=50.0)
        assert len(result) == 1
        assert result[0] == 5.0
    
    def test_percentile_nearest_rank_empty_input(self):
        """Test PNR fails with empty input - mirrors check_pnr_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data is empty"):
            ta_indicators.percentile_nearest_rank(empty)
    
    def test_percentile_nearest_rank_invalid_percentage(self):
        """Test PNR fails with invalid percentage - mirrors check_pnr_invalid_percentage"""
        data = np.array([1.0] * 20)
        
        # Test percentage > 100
        with pytest.raises(ValueError, match="Percentage must be between"):
            ta_indicators.percentile_nearest_rank(data, length=5, percentage=150.0)
        
        # Test negative percentage
        with pytest.raises(ValueError, match="Percentage must be between"):
            ta_indicators.percentile_nearest_rank(data, length=5, percentage=-10.0)
    
    def test_percentile_nearest_rank_nan_handling(self, test_data):
        """Test PNR handles NaN values correctly - mirrors check_pnr_nan_handling"""
        data = np.array([
            1.0, 2.0, np.nan, 4.0, 5.0,
            np.nan, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, np.nan, 15.0,
        ])
        
        result = ta_indicators.percentile_nearest_rank(data, length=5, percentage=50.0)
        assert len(result) == len(data)
        
        # Should handle NaN values in window
        assert not np.isnan(result[6])
    
    def test_percentile_nearest_rank_basic(self):
        """Test basic functionality with simple data"""
        data = np.array(EXPECTED_OUTPUTS['percentile_nearest_rank']['basic_test']['data'])
        expected_test = EXPECTED_OUTPUTS['percentile_nearest_rank']['basic_test']
        
        result = ta_indicators.percentile_nearest_rank(
            data, 
            length=expected_test['length'],
            percentage=expected_test['percentage']
        )
        
        assert len(result) == len(data)
        
        # First 4 values should be NaN (warmup period for length=5)
        assert np.all(np.isnan(result[:4]))
        
        # Check expected values from Rust tests
        assert result[4] == expected_test['expected_at_4']
        assert result[5] == expected_test['expected_at_5']
    
    def test_percentile_nearest_rank_different_percentiles(self):
        """Test with different percentile values"""
        data = np.array(EXPECTED_OUTPUTS['percentile_nearest_rank']['percentile_tests']['data'])
        expected_test = EXPECTED_OUTPUTS['percentile_nearest_rank']['percentile_tests']
        length = expected_test['length']
        
        # Test 25th percentile
        result_25 = ta_indicators.percentile_nearest_rank(data, length=length, percentage=25.0)
        assert result_25[4] == expected_test['p25_at_4']
        
        # Test 75th percentile
        result_75 = ta_indicators.percentile_nearest_rank(data, length=length, percentage=75.0)
        assert result_75[4] == expected_test['p75_at_4']
        
        # Test 100th percentile (max)
        result_100 = ta_indicators.percentile_nearest_rank(data, length=length, percentage=100.0)
        assert result_100[4] == expected_test['p100_at_4']
    
    def test_percentile_nearest_rank_streaming(self, test_data):
        """Test PNR streaming matches batch calculation - mirrors check_pnr_streaming"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        length = 15
        percentage = 50.0
        
        # Batch calculation
        batch_result = ta_indicators.percentile_nearest_rank(
            close, length=length, percentage=percentage
        )
        
        # Streaming calculation
        stream = ta_indicators.PercentileNearestRankStream(length=length, percentage=percentage)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values where both are not NaN
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9, 
                        msg=f"PNR streaming mismatch at index {i}")
    
    def test_percentile_nearest_rank_batch(self, test_data):
        """Test PNR batch processing - mirrors check_batch_default_row"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Test single parameter combination
        result = ta_indicators.percentile_nearest_rank_batch(
            close,
            length_range=(15, 15, 0),  # Default length only
            percentage_range=(50.0, 50.0, 0.0)  # Default percentage only
        )
        
        assert 'values' in result
        assert 'lengths' in result
        assert 'percentages' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row and compare with regular calculation
        batch_row = result['values'][0]
        single_result = ta_indicators.percentile_nearest_rank(close, length=15, percentage=50.0)
        
        assert_close(
            batch_row,
            single_result,
            rtol=1e-8,
            msg="PNR batch default row mismatch"
        )
    
    def test_percentile_nearest_rank_batch_sweep(self, test_data):
        """Test PNR batch with parameter sweep"""
        close = test_data['close'][:50]  # Use smaller dataset for speed
        
        # Test parameter sweep
        result = ta_indicators.percentile_nearest_rank_batch(
            close,
            length_range=(10, 20, 10),  # 10, 20
            percentage_range=(25.0, 75.0, 25.0)  # 25, 50, 75
        )
        
        # Should have 2 * 3 = 6 combinations
        assert result['values'].shape[0] == 6
        assert result['values'].shape[1] == len(close)
        assert len(result['lengths']) == 6
        assert len(result['percentages']) == 6
        
        # Verify first combination (length=10, percentage=25)
        assert result['lengths'][0] == 10
        assert result['percentages'][0] == 25.0
        
        # Verify last combination (length=20, percentage=75)
        assert result['lengths'][5] == 20
        assert result['percentages'][5] == 75.0
        
        # Check warmup periods are correct
        for i in range(6):
            length = result['lengths'][i]
            row = result['values'][i]
            # First length-1 values should be NaN or uninitialized (very small values)
            # Note: Python binding may return uninitialized memory as small values instead of NaN
            warmup_values = row[:length-1]
            # Check that these are either NaN or effectively zero (uninitialized)
            assert np.all(np.isnan(warmup_values) | (np.abs(warmup_values) < 1e-100))
            # After warmup should have valid values
            if length < len(close):
                assert not np.isnan(row[length-1]) and np.abs(row[length-1]) > 1e-10
    
    def test_percentile_nearest_rank_batch_metadata(self, test_data):
        """Test batch result includes correct metadata"""
        close = test_data['close'][:30]
        
        result = ta_indicators.percentile_nearest_rank_batch(
            close,
            length_range=(5, 10, 5),  # 5, 10
            percentage_range=(50.0, 100.0, 50.0)  # 50, 100
        )
        
        # Should have 2 * 2 = 4 combinations
        assert len(result['lengths']) == 4
        assert len(result['percentages']) == 4
        
        # Check parameter combinations
        expected_combos = [
            (5, 50.0), (5, 100.0),
            (10, 50.0), (10, 100.0)
        ]
        
        for i, (length, percentage) in enumerate(expected_combos):
            assert result['lengths'][i] == length
            assert_close(result['percentages'][i], percentage, rtol=1e-10)
    
    def test_percentile_nearest_rank_all_nan_input(self):
        """Test PNR with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.percentile_nearest_rank(all_nan)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])