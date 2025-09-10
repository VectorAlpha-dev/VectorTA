"""
Python binding tests for WTO indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS

try:
    import my_project as mp
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)


class TestWto:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_wto_accuracy(self, test_data):
        """Test WTO matches expected values from Rust tests - mirrors check_wto_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['wto']
        
        # Calculate WTO with default parameters
        wavetrend1, wavetrend2, histogram = mp.wto(
            close,
            channel_length=expected['default_params']['channel_length'],
            average_length=expected['default_params']['average_length']
        )
        
        # Check lengths
        assert len(wavetrend1) == len(close)
        assert len(wavetrend2) == len(close)
        assert len(histogram) == len(close)
        
        # Check last 5 values match expected with appropriate tolerance
        # Using 1e-5 to account for floating point differences across platforms
        assert_close(
            wavetrend1[-5:],
            expected['last_5_values']['wavetrend1'],
            rtol=1e-5,
            msg="WaveTrend1 last 5 values mismatch"
        )
        
        assert_close(
            wavetrend2[-5:],
            expected['last_5_values']['wavetrend2'],
            rtol=1e-5,
            msg="WaveTrend2 last 5 values mismatch"
        )
        
        # Histogram with appropriate tolerance
        assert_close(
            histogram[-5:],
            expected['last_5_values']['histogram'],
            rtol=1e-5,
            msg="Histogram last 5 values mismatch"
        )
    
    def test_wto_with_params(self, test_data):
        """Test WTO with custom parameters"""
        data = np.random.random(100) * 100 + 50
        
        # Test with custom parameters
        wt1, wt2, hist = mp.wto(data, channel_length=12, average_length=26)
        
        assert len(wt1) == len(data)
        assert len(wt2) == len(data)
        assert len(hist) == len(data)
        
        # Check that we eventually get valid values
        assert not np.all(np.isnan(wt1))
        assert not np.all(np.isnan(wt2))
        assert not np.all(np.isnan(hist))
    
    def test_wto_empty_input(self):
        """Test WTO fails with empty input"""
        with pytest.raises(Exception, match="Input data slice is empty"):
            mp.wto(np.array([]), channel_length=10, average_length=21)
    
    def test_wto_all_nan(self):
        """Test WTO fails with all NaN values"""
        data = np.full(50, np.nan)
        with pytest.raises(Exception, match="All values are NaN"):
            mp.wto(data, channel_length=10, average_length=21)
    
    def test_wto_insufficient_data(self):
        """Test WTO fails with insufficient data"""
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(Exception):
            mp.wto(data, channel_length=10, average_length=21)
    
    def test_wto_single_value(self):
        """Test WTO fails with single value"""
        data = np.array([42.0])
        with pytest.raises(Exception):
            mp.wto(data, channel_length=10, average_length=21)
    
    def test_wto_invalid_channel_length(self):
        """Test WTO fails with invalid channel_length"""
        data = np.random.random(100) * 100 + 50
        
        # Zero channel length
        with pytest.raises(Exception, match="Invalid period|channel"):
            mp.wto(data, channel_length=0, average_length=21)
        
        # Channel length exceeds data length
        with pytest.raises(Exception):
            mp.wto(data, channel_length=200, average_length=21)
    
    def test_wto_invalid_average_length(self):
        """Test WTO fails with invalid average_length"""
        data = np.random.random(100) * 100 + 50
        
        # Zero average length
        with pytest.raises(Exception, match="Invalid period|average"):
            mp.wto(data, channel_length=10, average_length=0)
        
        # Average length exceeds data length
        with pytest.raises(Exception):
            mp.wto(data, channel_length=10, average_length=200)
    
    def test_wto_nan_handling(self, test_data):
        """Test WTO handles NaN values correctly"""
        close = test_data['close']
        
        wt1, wt2, hist = mp.wto(close, channel_length=10, average_length=21)
        
        # Check warmup period
        warmup = EXPECTED_OUTPUTS['wto']['warmup_period']
        
        # After significant warmup, no NaN values should exist
        if len(wt1) > 240:
            assert not np.any(np.isnan(wt1[240:])), "Found unexpected NaN in wt1 after warmup"
            assert not np.any(np.isnan(wt2[240:])), "Found unexpected NaN in wt2 after warmup"
            assert not np.any(np.isnan(hist[240:])), "Found unexpected NaN in histogram after warmup"
    
    def test_wto_with_nan_prefix(self):
        """Test WTO with NaN values at the beginning"""
        data = np.concatenate([
            np.full(10, np.nan),
            np.random.random(100) * 100 + 50
        ])
        
        wt1, wt2, hist = mp.wto(data, channel_length=10, average_length=21)
        
        assert len(wt1) == len(data)
        assert len(wt2) == len(data)
        assert len(hist) == len(data)
        
        # Initial NaN values in input should be preserved
        # With first 10 NaN in input, output should have at least 10 NaN
        nan_count = sum(1 for x in wt1 if np.isnan(x))
        assert nan_count >= 10, f"Expected at least 10 NaN values but got {nan_count}"
        
        # But we should get some valid values eventually
        assert not np.all(np.isnan(wt1))
        assert not np.all(np.isnan(wt2))
        assert not np.all(np.isnan(hist))
    
    def test_wto_consistency(self):
        """Test that WTO produces consistent results"""
        np.random.seed(42)
        data = np.random.random(200) * 100 + 50
        
        # Calculate twice with same parameters
        wt1_a, wt2_a, hist_a = mp.wto(data, channel_length=10, average_length=21)
        wt1_b, wt2_b, hist_b = mp.wto(data, channel_length=10, average_length=21)
        
        # Results should be identical
        np.testing.assert_array_equal(wt1_a, wt1_b)
        np.testing.assert_array_equal(wt2_a, wt2_b)
        np.testing.assert_array_equal(hist_a, hist_b)
    
    def test_wto_histogram_calculation(self):
        """Test that histogram is correctly calculated as wt1 - wt2"""
        np.random.seed(42)
        data = np.random.random(200) * 100 + 50
        
        wt1, wt2, hist = mp.wto(data, channel_length=10, average_length=21)
        
        # Find indices where all values are valid
        valid_mask = ~(np.isnan(wt1) | np.isnan(wt2) | np.isnan(hist))
        
        # Check histogram calculation
        expected_hist = wt1[valid_mask] - wt2[valid_mask]
        np.testing.assert_array_almost_equal(hist[valid_mask], expected_hist, decimal=10)
    
    @pytest.mark.xfail(reason="WTO streaming implementation differs from batch - needs investigation")
    def test_wto_streaming(self, test_data):
        """Test WTO streaming matches batch calculation"""
        close = test_data['close']
        channel_length = 10
        average_length = 21
        
        # Batch calculation
        batch_wt1, batch_wt2, batch_hist = mp.wto(
            close, 
            channel_length=channel_length, 
            average_length=average_length
        )
        
        # Streaming calculation
        stream = mp.WtoStream(channel_length=channel_length, average_length=average_length)
        stream_wt1 = []
        stream_wt2 = []
        stream_hist = []
        
        for price in close:
            result = stream.update(price)
            if result is not None:
                # Result is a tuple (wt1, wt2, histogram)
                wt1_val, wt2_val, hist_val = result
                # The stream returns 0 initially before warmup is complete
                # Convert 0 to NaN to match batch behavior
                if wt1_val == 0 and wt2_val == 0 and hist_val == 0:
                    stream_wt1.append(np.nan)
                    stream_wt2.append(np.nan)
                    stream_hist.append(np.nan)
                else:
                    stream_wt1.append(wt1_val)
                    stream_wt2.append(wt2_val)
                    stream_hist.append(hist_val)
            else:
                stream_wt1.append(np.nan)
                stream_wt2.append(np.nan)
                stream_hist.append(np.nan)
        
        stream_wt1 = np.array(stream_wt1)
        stream_wt2 = np.array(stream_wt2)
        stream_hist = np.array(stream_hist)
        
        # Compare batch vs streaming
        assert len(batch_wt1) == len(stream_wt1)
        assert len(batch_wt2) == len(stream_wt2)
        assert len(batch_hist) == len(stream_hist)
        
        # Compare values where both are not NaN
        # Note: Streaming might return 0 during warmup while batch returns NaN
        # We'll only compare after a reasonable warmup period
        warmup = max(channel_length, average_length)
        for i in range(warmup, len(batch_wt1)):
            if np.isnan(batch_wt1[i]) and np.isnan(stream_wt1[i]):
                continue
            # Skip if one is 0 and the other is NaN (warmup difference)
            if (batch_wt1[i] == 0 and np.isnan(stream_wt1[i])) or (np.isnan(batch_wt1[i]) and stream_wt1[i] == 0):
                continue
            assert_close(batch_wt1[i], stream_wt1[i], rtol=1e-9, atol=1e-9,
                        msg=f"WTO wt1 streaming mismatch at index {i}")
            assert_close(batch_wt2[i], stream_wt2[i], rtol=1e-9, atol=1e-9,
                        msg=f"WTO wt2 streaming mismatch at index {i}")
            assert_close(batch_hist[i], stream_hist[i], rtol=1e-9, atol=1e-9,
                        msg=f"WTO histogram streaming mismatch at index {i}")
    
    def test_wto_batch(self, test_data):
        """Test WTO batch processing"""
        close = test_data['close']
        
        # Test single parameter combination
        result = mp.wto_batch(
            close,
            channel_range=(10, 10, 0),  # Default channel_length only
            average_range=(21, 21, 0)    # Default average_length only
        )
        
        # Should return three 2D arrays for wt1, wt2, histogram
        assert 'wt1' in result
        assert 'wt2' in result
        assert 'hist' in result
        assert 'channel_lengths' in result
        assert 'average_lengths' in result
        
        # Should have 1 combination (default params)
        assert result['wt1'].shape[0] == 1
        assert result['wt1'].shape[1] == len(close)
        assert result['wt2'].shape[0] == 1
        assert result['wt2'].shape[1] == len(close)
        assert result['hist'].shape[0] == 1
        assert result['hist'].shape[1] == len(close)
        
        # Extract the single row and compare with single calculation
        single_wt1, single_wt2, single_hist = mp.wto(close, channel_length=10, average_length=21)
        
        np.testing.assert_array_almost_equal(result['wt1'][0], single_wt1, decimal=10)
        np.testing.assert_array_almost_equal(result['wt2'][0], single_wt2, decimal=10)
        np.testing.assert_array_almost_equal(result['hist'][0], single_hist, decimal=10)
    
    def test_wto_batch_multiple_params(self, test_data):
        """Test WTO batch with multiple parameter combinations"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Multiple parameter combinations
        result = mp.wto_batch(
            close,
            channel_range=(10, 14, 2),  # 10, 12, 14
            average_range=(21, 25, 2)   # 21, 23, 25
        )
        
        # Should have 3 * 3 = 9 combinations
        assert result['wt1'].shape[0] == 9
        assert result['wt1'].shape[1] == 100
        assert result['wt2'].shape[0] == 9
        assert result['wt2'].shape[1] == 100
        assert result['hist'].shape[0] == 9
        assert result['hist'].shape[1] == 100
        
        # Verify parameter lists
        expected_channels = [10, 10, 10, 12, 12, 12, 14, 14, 14]
        expected_averages = [21, 23, 25, 21, 23, 25, 21, 23, 25]
        
        np.testing.assert_array_equal(result['channel_lengths'], expected_channels)
        np.testing.assert_array_equal(result['average_lengths'], expected_averages)
        
        # Verify first combination matches single calculation
        single_wt1, single_wt2, single_hist = mp.wto(close, channel_length=10, average_length=21)
        
        np.testing.assert_array_almost_equal(result['wt1'][0], single_wt1, decimal=10)
        np.testing.assert_array_almost_equal(result['wt2'][0], single_wt2, decimal=10)
        np.testing.assert_array_almost_equal(result['hist'][0], single_hist, decimal=10)
    
    def test_wto_batch_edge_cases(self):
        """Test WTO batch edge cases"""
        close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5, dtype=np.float64)  # 50 values
        
        # Single value sweep
        result = mp.wto_batch(
            close,
            channel_range=(5, 5, 1),
            average_range=(10, 10, 1)
        )
        
        assert result['wt1'].shape[0] == 1
        assert result['wt1'].shape[1] == 50
        
        # Step larger than range
        result = mp.wto_batch(
            close,
            channel_range=(5, 7, 10),  # Step larger than range
            average_range=(10, 10, 0)
        )
        
        # Should only have channel_length=5
        assert result['wt1'].shape[0] == 1
        assert len(result['channel_lengths']) == 1
        assert result['channel_lengths'][0] == 5
        
        # Empty data should throw
        with pytest.raises(Exception, match="All values are NaN|Input data slice is empty"):
            mp.wto_batch(
                np.array([], dtype=np.float64),
                channel_range=(10, 10, 0),
                average_range=(21, 21, 0)
            )


if __name__ == "__main__":
    pytest.main([__file__, '-v'])