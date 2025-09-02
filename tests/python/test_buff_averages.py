"""
Python binding tests for Buff Averages indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import the built module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'target/wheels'))

try:
    import my_project as ta_indicators
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


class TestBuffAverages:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_buff_averages_accuracy(self, test_data):
        """Test Buff Averages matches expected values from Rust tests - mirrors check_buff_averages_accuracy"""
        close = test_data['close']
        volume = test_data['volume']
        
        # Call the buff_averages function with default parameters
        fast_buff, slow_buff = ta_indicators.buff_averages(
            close,
            volume,
            fast_period=5,
            slow_period=20
        )
        
        assert len(fast_buff) == len(close)
        assert len(slow_buff) == len(close)
        
        # Expected values from PineScript (NEVER MODIFY!)
        expected_fast = [
            58740.30855637,
            59132.28418702,
            59309.76658172,
            59266.10492431,
            59194.11908892,
        ]
        
        expected_slow = [
            59209.26229392,
            59201.87047432,
            59217.15739355,
            59195.74527194,
            59196.26139533,
        ]
        
        # Check last 6 values, use first 5 for comparison
        assert_close(
            fast_buff[-6:-1],
            expected_fast,
            rtol=1e-3,
            msg="Buff Averages fast buffer last 5 values mismatch"
        )
        
        assert_close(
            slow_buff[-6:-1],
            expected_slow,
            rtol=1e-3,
            msg="Buff Averages slow buffer last 5 values mismatch"
        )
    
    def test_buff_averages_partial_params(self, test_data):
        """Test Buff Averages with partial parameters - mirrors check_buff_averages_partial_params"""
        close = test_data['close']
        volume = test_data['volume']
        
        # Test with default parameters (None values should use defaults)
        fast_buff, slow_buff = ta_indicators.buff_averages(
            close,
            volume
            # Using default fast_period=5, slow_period=20
        )
        
        assert len(fast_buff) == len(close)
        assert len(slow_buff) == len(close)
    
    def test_buff_averages_nan_handling(self, test_data):
        """Test Buff Averages handles NaN values correctly - mirrors check_buff_nan_prefix"""
        close = test_data['close']
        volume = test_data['volume']
        
        fast_buff, slow_buff = ta_indicators.buff_averages(
            close,
            volume,
            fast_period=5,
            slow_period=20
        )
        
        # Find first non-NaN value in input
        first_valid = next((i for i, x in enumerate(close) if not np.isnan(x)), 0)
        warmup_period = first_valid + 20 - 1  # slow_period - 1
        
        # Check warmup period has NaN values
        assert np.all(np.isnan(fast_buff[:warmup_period])), "Expected NaN in fast buffer warmup period"
        assert np.all(np.isnan(slow_buff[:warmup_period])), "Expected NaN in slow buffer warmup period"
        
        # Check post-warmup has no NaN values
        assert np.all(np.isfinite(fast_buff[warmup_period:])), "Found unexpected NaN in fast buffer after warmup"
        assert np.all(np.isfinite(slow_buff[warmup_period:])), "Found unexpected NaN in slow buffer after warmup"
    
    def test_buff_averages_empty_input(self):
        """Test Buff Averages fails with empty input - mirrors check_buff_averages_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.buff_averages(empty, empty)
    
    def test_buff_averages_all_nan(self):
        """Test Buff Averages fails with all NaN values - mirrors check_buff_averages_all_nan"""
        nan_data = np.full(100, np.nan)
        nan_volume = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.buff_averages(nan_data, nan_volume)
    
    def test_buff_averages_zero_period(self):
        """Test Buff Averages fails with zero period - mirrors check_buff_averages_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        volume_data = np.array([100.0, 200.0, 300.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.buff_averages(input_data, volume_data, fast_period=0)
    
    def test_buff_averages_period_exceeds_length(self):
        """Test Buff Averages fails when period exceeds data length - mirrors check_buff_averages_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        volume_small = np.array([100.0, 200.0, 300.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.buff_averages(data_small, volume_small, slow_period=10)
    
    def test_buff_averages_very_small_dataset(self):
        """Test Buff Averages fails with insufficient data - mirrors check_buff_averages_very_small_dataset"""
        single_point = np.array([42.0])
        single_volume = np.array([100.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.buff_averages(single_point, single_volume)
    
    def test_buff_averages_mismatched_lengths(self):
        """Test Buff Averages fails with mismatched data lengths - mirrors check_buff_averages_mismatched_lengths"""
        price_data = np.array([10.0, 20.0, 30.0])
        volume_data = np.array([100.0, 200.0])  # Different length
        
        with pytest.raises(ValueError, match="mismatched|different length"):
            ta_indicators.buff_averages(price_data, volume_data)
    
    def test_buff_averages_streaming(self, test_data):
        """Test BuffAveragesStream for real-time updates - improved version"""
        stream = ta_indicators.BuffAveragesStream(fast_period=5, slow_period=20)
        
        # Feed data until we get output
        prices = test_data['close'][:50]
        volumes = test_data['volume'][:50]
        
        stream_results = []
        for i in range(len(prices)):
            result = stream.update(prices[i], volumes[i])
            if i < 19:  # Before slow_period - 1
                assert result is None, f"Should not have output before period at index {i}"
            else:
                assert result is not None, f"Should have output after period at index {i}"
                assert len(result) == 2, "Should return (fast_buff, slow_buff) tuple"
                stream_results.append(result)
        
        # Verify streaming matches batch calculation for the values we can compare
        fast_buff_batch, slow_buff_batch = ta_indicators.buff_averages(
            prices,
            volumes,
            fast_period=5,
            slow_period=20
        )
        
        # Compare last values from streaming with batch
        for i, (stream_fast, stream_slow) in enumerate(stream_results[-5:]):
            batch_idx = 45 + i  # Last 5 indices
            assert_close(
                stream_fast,
                fast_buff_batch[batch_idx],
                rtol=1e-9,
                msg=f"Stream vs batch fast mismatch at index {batch_idx}"
            )
            assert_close(
                stream_slow,
                slow_buff_batch[batch_idx],
                rtol=1e-9,
                msg=f"Stream vs batch slow mismatch at index {batch_idx}"
            )
    
    def test_buff_averages_batch_single(self, test_data):
        """Test Buff Averages batch with single parameter set - mirrors check_buff_averages_batch_single"""
        close = test_data['close']
        volume = test_data['volume']
        
        # Batch with single combination (default parameters)
        result = ta_indicators.buff_averages_batch(
            close,
            volume,
            fast_range=(5, 5, 1),  # Single value: 5
            slow_range=(20, 20, 1)  # Single value: 20
        )
        
        assert 'fast' in result
        assert 'slow' in result
        assert 'fast_periods' in result
        assert 'slow_periods' in result
        
        # Should have 1 combination
        assert result['fast'].shape[0] == 1
        assert result['slow'].shape[0] == 1
        assert result['fast'].shape[1] == len(close)
        assert result['slow'].shape[1] == len(close)
        
        # Extract the single rows
        fast_row = result['fast'][0]
        slow_row = result['slow'][0]
        
        # Should match regular calculation
        fast_regular, slow_regular = ta_indicators.buff_averages(close, volume, 5, 20)
        
        assert_close(
            fast_row,
            fast_regular,
            rtol=1e-9,
            msg="Batch fast vs regular mismatch"
        )
        
        assert_close(
            slow_row,
            slow_regular,
            rtol=1e-9,
            msg="Batch slow vs regular mismatch"
        )
    
    def test_buff_averages_batch_multiple(self, test_data):
        """Test Buff Averages batch with multiple parameter combinations"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        volume = test_data['volume'][:100]
        
        # Multiple combinations: fast=[3,5], slow=[15,20]
        result = ta_indicators.buff_averages_batch(
            close,
            volume,
            fast_range=(3, 5, 2),   # 3, 5
            slow_range=(15, 20, 5)  # 15, 20
        )
        
        # Should have 2*2 = 4 combinations
        expected_combos = [
            (3, 15), (3, 20),
            (5, 15), (5, 20)
        ]
        
        assert result['fast'].shape[0] == 4
        assert result['slow'].shape[0] == 4
        assert len(result['fast_periods']) == 4
        assert len(result['slow_periods']) == 4
        
        # Verify parameter combinations
        for i, (fast_p, slow_p) in enumerate(expected_combos):
            assert result['fast_periods'][i] == fast_p
            assert result['slow_periods'][i] == slow_p
            
            # Verify each row matches individual calculation
            fast_single, slow_single = ta_indicators.buff_averages(
                close, volume, fast_p, slow_p
            )
            
            assert_close(
                result['fast'][i],
                fast_single,
                rtol=1e-9,
                msg=f"Batch fast mismatch for combo ({fast_p}, {slow_p})"
            )
            
            assert_close(
                result['slow'][i],
                slow_single,
                rtol=1e-9,
                msg=f"Batch slow mismatch for combo ({fast_p}, {slow_p})"
            )
    
    def test_buff_averages_batch_edge_cases(self):
        """Test Buff Averages batch edge cases"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
        volume = np.array([1.0] * 20)
        
        # Step larger than range (should give single value)
        result = ta_indicators.buff_averages_batch(
            data,
            volume,
            fast_range=(5, 7, 10),  # Step > range, only 5
            slow_range=(10, 10, 1)  # Single value
        )
        
        assert result['fast'].shape[0] == 1
        assert result['fast_periods'][0] == 5
        assert result['slow_periods'][0] == 10
        
        # Zero step (should use step=1)
        result = ta_indicators.buff_averages_batch(
            data,
            volume,
            fast_range=(5, 5, 0),   # Zero step
            slow_range=(10, 10, 0)  # Zero step
        )
        
        assert result['fast'].shape[0] == 1
        
        # Empty data should fail
        empty = np.array([])
        with pytest.raises(ValueError):
            ta_indicators.buff_averages_batch(
                empty,
                empty,
                fast_range=(5, 5, 1),
                slow_range=(20, 20, 1)
            )
    
    def test_buff_averages_batch_warmup_periods(self, test_data):
        """Test that batch processing correctly handles different warmup periods"""
        close = test_data['close'][:50]
        volume = test_data['volume'][:50]
        
        result = ta_indicators.buff_averages_batch(
            close,
            volume,
            fast_range=(3, 5, 2),   # 3, 5
            slow_range=(10, 20, 10) # 10, 20
        )
        
        # Each combination should have different warmup period
        combos = [(3, 10), (3, 20), (5, 10), (5, 20)]
        
        for i, (fast_p, slow_p) in enumerate(combos):
            fast_row = result['fast'][i]
            slow_row = result['slow'][i]
            
            # Warmup period is slow_period - 1
            warmup = slow_p - 1
            
            # Check NaN in warmup period
            assert np.all(np.isnan(fast_row[:warmup])), \
                f"Expected NaN in fast warmup for combo ({fast_p}, {slow_p})"
            assert np.all(np.isnan(slow_row[:warmup])), \
                f"Expected NaN in slow warmup for combo ({fast_p}, {slow_p})"
            
            # Check values after warmup
            assert np.all(np.isfinite(fast_row[warmup:])), \
                f"Unexpected NaN after fast warmup for combo ({fast_p}, {slow_p})"
            assert np.all(np.isfinite(slow_row[warmup:])), \
                f"Unexpected NaN after slow warmup for combo ({fast_p}, {slow_p})"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])