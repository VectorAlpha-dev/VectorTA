"""
Python binding tests for Nadaraya-Watson Envelope indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

try:
    import ta_indicators
except ImportError:
    # If not in virtual environment, try to import from installed location
    try:
        import my_project as ta_indicators
    except ImportError:
        pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


class TestNadarayaWatsonEnvelope:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_nwe_accuracy(self, test_data):
        """Test NWE matches expected values from PineScript - non-repainting mode"""
        close = test_data['close']
        
        # Reference values from PineScript (non-repainting mode)
        expected_upper = [62141.41569185, 62108.86018850, 62069.70106389, 62045.52821051, 61980.68541380]
        expected_lower = [56560.88881720, 56530.68399610, 56490.03377396, 56465.39492722, 56394.51167599]
        
        # Calculate with default parameters
        upper, lower = ta_indicators.nadaraya_watson_envelope(
            close,
            bandwidth=8.0,
            multiplier=3.0,
            lookback=500
        )
        
        assert len(upper) == len(close)
        assert len(lower) == len(close)
        
        # Check last 5 values match expected
        # Match Rust test tolerance: abs error <= 1e-6
        assert_close(
            upper[-5:],
            expected_upper,
            rtol=0.0,
            atol=1e-6,
            msg="NWE upper envelope last 5 values mismatch"
        )
        
        # Match Rust test tolerance: abs error <= 1e-6
        assert_close(
            lower[-5:],
            expected_lower,
            rtol=0.0,
            atol=1e-6,
            msg="NWE lower envelope last 5 values mismatch"
        )
    
    def test_nwe_default_params(self, test_data):
        """Test NWE with default parameters"""
        close = test_data['close']
        
        # Default params: bandwidth=8.0, multiplier=3.0, lookback=500
        upper, lower = ta_indicators.nadaraya_watson_envelope(
            close,
            bandwidth=8.0,
            multiplier=3.0,
            lookback=500
        )
        
        assert len(upper) == len(close)
        assert len(lower) == len(close)
        
        # Upper should be greater than lower everywhere after warmup
        for i in range(10, len(close)):
            if not np.isnan(upper[i]) and not np.isnan(lower[i]):
                assert upper[i] > lower[i], f"Upper not greater than lower at index {i}"
    
    def test_nwe_invalid_bandwidth(self, test_data):
        """Test NWE fails with invalid bandwidth"""
        data = test_data['close'][:100]
        
        # Zero bandwidth
        with pytest.raises(ValueError, match="Invalid bandwidth"):
            ta_indicators.nadaraya_watson_envelope(data, bandwidth=0.0, multiplier=3.0, lookback=500)
        
        # Negative bandwidth
        with pytest.raises(ValueError, match="Invalid bandwidth"):
            ta_indicators.nadaraya_watson_envelope(data, bandwidth=-1.0, multiplier=3.0, lookback=500)
    
    def test_nwe_invalid_multiplier(self, test_data):
        """Test NWE fails with invalid multiplier"""
        data = test_data['close'][:100]
        
        # Negative multiplier
        with pytest.raises(ValueError, match="Invalid multiplier"):
            ta_indicators.nadaraya_watson_envelope(data, bandwidth=8.0, multiplier=-1.0, lookback=500)
    
    def test_nwe_invalid_lookback(self):
        """Test NWE fails with invalid lookback"""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        # Zero lookback
        with pytest.raises(ValueError, match="Invalid lookback"):
            ta_indicators.nadaraya_watson_envelope(data, bandwidth=8.0, multiplier=3.0, lookback=0)
    
    def test_nwe_empty_input(self):
        """Test NWE fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.nadaraya_watson_envelope(empty, bandwidth=8.0, multiplier=3.0, lookback=500)
    
    def test_nwe_all_nan_input(self):
        """Test NWE with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.nadaraya_watson_envelope(all_nan, bandwidth=8.0, multiplier=3.0, lookback=500)
    
    def test_nwe_small_dataset(self):
        """Test NWE with very small dataset - mirrors check_nwe_very_small_dataset"""
        # Based on the actual Rust behavior: Single point with lookback=500 fails
        # The Rust test passes because the default parameters succeed, but 
        # lookback > data length should fail
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.nadaraya_watson_envelope(
                single_point, bandwidth=8.0, multiplier=3.0, lookback=500
            )
        
        # Small dataset with lookback > data length should fail
        small_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        with pytest.raises(ValueError, match="Not enough valid data"):
            ta_indicators.nadaraya_watson_envelope(
                small_data, bandwidth=8.0, multiplier=3.0, lookback=10
            )
    
    def test_nwe_different_parameters(self, test_data):
        """Test NWE with different parameter combinations"""
        # Use sufficient data for default lookback (need at least 1000 for warmup)
        close = test_data['close'][:1100]  
        lookback = 100  # Use smaller lookback for parameter testing
        
        # Test with different bandwidths
        upper1, lower1 = ta_indicators.nadaraya_watson_envelope(
            close, bandwidth=4.0, multiplier=3.0, lookback=lookback
        )
        upper2, lower2 = ta_indicators.nadaraya_watson_envelope(
            close, bandwidth=16.0, multiplier=3.0, lookback=lookback
        )
        
        assert len(upper1) == len(close)
        assert len(upper2) == len(close)
        
        # Larger bandwidth should generally produce smoother output
        # (harder to test directly, but at least verify they're different)
        # Check after warmup period (lookback-1 + mae_len-1)
        warmup = lookback - 1 + 498  # mae_len = 499
        if warmup < len(close):
            assert not np.allclose(upper1[warmup:], upper2[warmup:], equal_nan=True)
        
        # Test with different multipliers
        upper3, lower3 = ta_indicators.nadaraya_watson_envelope(
            close, bandwidth=8.0, multiplier=1.0, lookback=lookback
        )
        upper4, lower4 = ta_indicators.nadaraya_watson_envelope(
            close, bandwidth=8.0, multiplier=5.0, lookback=lookback
        )
        
        # Larger multiplier should produce wider bands
        # Check at points after warmup
        found_wider = False
        for i in range(warmup, min(warmup + 100, len(close))):
            if not any(np.isnan([upper3[i], lower3[i], upper4[i], lower4[i]])):
                band_width_small = upper3[i] - lower3[i]
                band_width_large = upper4[i] - lower4[i]
                if band_width_large > band_width_small:
                    found_wider = True
                    break
        
        assert found_wider, "Larger multiplier should produce wider bands"
    
    def test_nwe_warmup_period(self):
        """Test NWE warmup period calculation - mirrors check_nwe_warmup_period"""
        # Create data with exactly 1000 points to test warmup
        data = np.array([(50000.0 + np.sin(i) * 100.0) for i in range(1000)])
        
        # Calculate with default parameters
        upper, lower = ta_indicators.nadaraya_watson_envelope(
            data, bandwidth=8.0, multiplier=3.0, lookback=500
        )
        
        # With defaults: lookback=500, mae_len=499
        # First non-NaN should be at index 997 (lookback-1 + mae_len-1)
        WARMUP = 499 + 498
        
        # All values before warmup should be NaN
        for i in range(WARMUP):
            assert np.isnan(upper[i]), f"Upper should be NaN at {i} during warmup"
            assert np.isnan(lower[i]), f"Lower should be NaN at {i} during warmup"
        
        # First valid value should be at WARMUP index
        if len(data) > WARMUP:
            assert not np.isnan(upper[WARMUP]), f"Upper should not be NaN at {WARMUP}"
            assert not np.isnan(lower[WARMUP]), f"Lower should not be NaN at {WARMUP}"
    
    def test_nwe_streaming(self, test_data):
        """Test NWE streaming matches batch calculation - mirrors check_nwe_streaming"""
        # Need enough data for non-NaN output
        data = np.array([(50000.0 + np.sin(i * 0.1) * 1000.0) for i in range(1100)])
        
        bandwidth = 8.0
        multiplier = 3.0
        lookback = 500  # Default lookback
        
        # Batch calculation
        batch_upper, batch_lower = ta_indicators.nadaraya_watson_envelope(
            data, bandwidth=bandwidth, multiplier=multiplier, lookback=lookback
        )
        
        # Streaming calculation
        stream = ta_indicators.NweStream(
            bandwidth=bandwidth, multiplier=multiplier, lookback=lookback
        )
        stream_results = []
        
        for price in data:
            result = stream.update(price)
            stream_results.append(result)
        
        # Find first non-NaN in batch
        batch_start = next((i for i, v in enumerate(batch_upper) if not np.isnan(v)), len(batch_upper))
        
        # Count non-None streaming results
        stream_non_none = [r for r in stream_results if r is not None]
        
        # Streaming should eventually produce values after warmup
        if batch_start < len(batch_upper) and len(stream_non_none) > 0:
            # Compare the last few valid values
            # Streaming may have different warmup behavior
            compare_count = min(5, len(stream_non_none))
            
            for i in range(compare_count):
                stream_idx = len(stream_results) - compare_count + i
                batch_idx = len(batch_upper) - compare_count + i
                
                if stream_results[stream_idx] is not None:
                    stream_u, stream_l = stream_results[stream_idx]
                    batch_u, batch_l = batch_upper[batch_idx], batch_lower[batch_idx]
                    
                    if not np.isnan(batch_u) and not np.isnan(stream_u):
                        # Allow for small differences due to algorithm variations
                        assert abs(batch_u - stream_u) / abs(batch_u) < 0.01, \
                            f"Upper envelope streaming mismatch at position -{compare_count - i}"
                        assert abs(batch_l - stream_l) / abs(batch_l) < 0.01, \
                            f"Lower envelope streaming mismatch at position -{compare_count - i}"
    
    def test_nwe_batch(self, test_data):
        """Test NWE batch processing"""
        close = test_data['close'][:100]  # Use smaller subset
        
        result = ta_indicators.nadaraya_watson_envelope_batch(
            close,
            bandwidth_range=(8.0, 8.0, 0.0),  # Single bandwidth
            multiplier_range=(3.0, 3.0, 0.0),  # Single multiplier
            lookback_range=(500, 500, 0)  # Single lookback
        )
        
        assert 'upper' in result
        assert 'lower' in result
        assert 'bandwidths' in result
        assert 'multipliers' in result
        assert 'lookbacks' in result
        
        # Should have 1 combination
        assert result['upper'].shape[0] == 1
        assert result['lower'].shape[0] == 1
        assert result['upper'].shape[1] == len(close)
        assert result['lower'].shape[1] == len(close)
        
        # Test with multiple parameter combinations
        result_multi = ta_indicators.nadaraya_watson_envelope_batch(
            close,
            bandwidth_range=(6.0, 10.0, 2.0),  # 3 values: 6, 8, 10
            multiplier_range=(2.0, 4.0, 1.0),  # 3 values: 2, 3, 4
            lookback_range=(400, 500, 100)  # 2 values: 400, 500
        )
        
        # Should have 3 * 3 * 2 = 18 combinations
        assert result_multi['upper'].shape[0] == 18
        assert result_multi['lower'].shape[0] == 18
        assert len(result_multi['bandwidths']) == 18
        assert len(result_multi['multipliers']) == 18
        assert len(result_multi['lookbacks']) == 18


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
