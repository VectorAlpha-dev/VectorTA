"""
Python binding tests for Aroon indicator.
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


class TestAroon:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_aroon_partial_params(self, test_data):
        """Test Aroon with partial parameters (None values) - mirrors check_aroon_partial_params"""
        high = test_data['high']
        low = test_data['low']
        
        # Test with default params (length=14)
        result = ta_indicators.aroon(high, low, 14)  # Using default
        assert 'up' in result
        assert 'down' in result
        assert len(result['up']) == len(high)
        assert len(result['down']) == len(low)
    
    def test_aroon_accuracy(self, test_data):
        """Test Aroon matches expected values from Rust tests - mirrors check_aroon_accuracy"""
        high = test_data['high']
        low = test_data['low']
        expected = EXPECTED_OUTPUTS['aroon']
        
        result = ta_indicators.aroon(
            high, low,
            length=expected['default_params']['length']
        )
        
        assert len(result['up']) == len(high)
        assert len(result['down']) == len(low)
        
        # Check last 5 values match expected
        assert_close(
            result['up'][-5:], 
            expected['last_5_up'],
            rtol=1e-2,  # Aroon uses 1e-2 tolerance in Rust tests
            msg="Aroon up last 5 values mismatch"
        )
        assert_close(
            result['down'][-5:], 
            expected['last_5_down'],
            rtol=1e-2,  # Aroon uses 1e-2 tolerance in Rust tests
            msg="Aroon down last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('aroon', result, 'hl', expected['default_params'])
    
    def test_aroon_default_candles(self, test_data):
        """Test Aroon with default parameters - mirrors check_aroon_default_candles"""
        high = test_data['high']
        low = test_data['low']
        
        # Default params: length=14
        result = ta_indicators.aroon(high, low, 14)
        assert len(result['up']) == len(high)
        assert len(result['down']) == len(low)
    
    def test_aroon_zero_length(self):
        """Test Aroon fails with zero length - mirrors check_aroon_zero_length"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 11.0])
        
        with pytest.raises(ValueError, match="Invalid length"):
            ta_indicators.aroon(high, low, length=0)
    
    def test_aroon_length_exceeds_data(self):
        """Test Aroon fails when length exceeds data length - mirrors check_aroon_length_exceeds_data"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 11.0])
        
        with pytest.raises(ValueError, match="Invalid length"):
            ta_indicators.aroon(high, low, length=14)
    
    def test_aroon_very_small_dataset(self):
        """Test Aroon fails with insufficient data - mirrors check_aroon_very_small_data_set"""
        high = np.array([100.0])
        low = np.array([99.5])
        
        with pytest.raises(ValueError, match="Invalid length|Not enough valid data"):
            ta_indicators.aroon(high, low, length=14)
    
    def test_aroon_empty_input(self):
        """Test Aroon fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Invalid length"):
            ta_indicators.aroon(empty, empty, length=14)
    
    def test_aroon_mismatched_lengths(self):
        """Test Aroon fails with mismatched input lengths"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0])  # Different length
        
        with pytest.raises(ValueError, match="High/low data length mismatch"):
            ta_indicators.aroon(high, low, length=2)
    
    def test_aroon_reinput(self, test_data):
        """Test Aroon applied with different parameters - mirrors check_aroon_reinput"""
        high = test_data['high']
        low = test_data['low']
        
        # First pass with length=14
        first_result = ta_indicators.aroon(high, low, length=14)
        assert len(first_result['up']) == len(high)
        assert len(first_result['down']) == len(low)
        
        # Second pass with length=5
        second_result = ta_indicators.aroon(high, low, length=5)
        assert len(second_result['up']) == len(high)
        assert len(second_result['down']) == len(low)
    
    def test_aroon_nan_handling(self, test_data):
        """Test Aroon handles NaN values correctly - mirrors check_aroon_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        
        result = ta_indicators.aroon(high, low, length=14)
        assert len(result['up']) == len(high)
        assert len(result['down']) == len(low)
        
        # After warmup period (240), no NaN values should exist
        if len(result['up']) > 240:
            assert not np.any(np.isnan(result['up'][240:])), "Found unexpected NaN in up after warmup period"
            assert not np.any(np.isnan(result['down'][240:])), "Found unexpected NaN in down after warmup period"
        
        # First `length` values should be NaN
        expected_warmup = 14  # for length=14
        assert np.all(np.isnan(result['up'][:expected_warmup])), "Expected NaN in up warmup period"
        assert np.all(np.isnan(result['down'][:expected_warmup])), "Expected NaN in down warmup period"
    
    def test_aroon_streaming(self, test_data):
        """Test Aroon streaming API - mirrors check_aroon_streaming"""
        high = test_data['high']
        low = test_data['low']
        length = 14
        
        # Batch calculation
        batch_result = ta_indicators.aroon(high, low, length=length)
        
        # Streaming calculation
        stream = ta_indicators.AroonStream(length=length)
        stream_up = []
        stream_down = []
        
        for h, l in zip(high, low):
            result = stream.update(h, l)
            if result is not None:
                stream_up.append(result[0])
                stream_down.append(result[1])
            else:
                stream_up.append(np.nan)
                stream_down.append(np.nan)
        
        stream_up = np.array(stream_up)
        stream_down = np.array(stream_down)
        
        # Compare batch vs streaming
        assert len(batch_result['up']) == len(stream_up)
        assert len(batch_result['down']) == len(stream_down)
        
        # Check they match (allowing for floating point differences)
        mask_up = ~(np.isnan(batch_result['up']) | np.isnan(stream_up))
        mask_down = ~(np.isnan(batch_result['down']) | np.isnan(stream_down))
        
        if np.any(mask_up):
            assert_close(
                batch_result['up'][mask_up],
                stream_up[mask_up],
                rtol=1e-8,
                msg="Aroon up streaming mismatch"
            )
        
        if np.any(mask_down):
            assert_close(
                batch_result['down'][mask_down],
                stream_down[mask_down],
                rtol=1e-8,
                msg="Aroon down streaming mismatch"
            )
    
    def test_aroon_batch(self, test_data):
        """Test Aroon batch processing - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        
        result = ta_indicators.aroon_batch(
            high, low,
            length_range=(14, 14, 0),  # Default length only
        )
        
        assert 'up' in result
        assert 'down' in result
        assert 'lengths' in result
        
        # Should have 1 combination (default params)
        assert result['up'].shape[0] == 1
        assert result['up'].shape[1] == len(high)
        assert result['down'].shape[0] == 1
        assert result['down'].shape[1] == len(low)
        
        # Extract the single row
        default_up = result['up'][0]
        default_down = result['down'][0]
        expected = EXPECTED_OUTPUTS['aroon']
        
        # Check last 5 values match (with Aroon tolerance)
        assert_close(
            default_up[-5:],
            expected['last_5_up'],
            rtol=1e-2,
            msg="Aroon batch default up row mismatch"
        )
        assert_close(
            default_down[-5:],
            expected['last_5_down'],
            rtol=1e-2,
            msg="Aroon batch default down row mismatch"
        )
    
    def test_aroon_all_nan_input(self):
        """Test Aroon with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        # Aroon should handle all NaN inputs by returning all NaN outputs
        result = ta_indicators.aroon(all_nan, all_nan, length=14)
        assert np.all(np.isnan(result['up']))
        assert np.all(np.isnan(result['down']))
    
    def test_aroon_batch_multiple_lengths(self, test_data):
        """Test Aroon batch with multiple lengths"""
        high = test_data['high'][:100]  # Use smaller dataset for speed
        low = test_data['low'][:100]
        
        result = ta_indicators.aroon_batch(
            high, low,
            length_range=(10, 20, 5),  # lengths: 10, 15, 20
        )
        
        assert 'up' in result
        assert 'down' in result
        assert 'lengths' in result
        
        # Should have 3 combinations
        assert result['up'].shape[0] == 3
        assert result['up'].shape[1] == 100
        assert result['down'].shape[0] == 3
        assert result['down'].shape[1] == 100
        
        # Check lengths array
        assert len(result['lengths']) == 3
        assert list(result['lengths']) == [10, 15, 20]
        
        # Verify each row has proper warmup
        for i, length in enumerate([10, 15, 20]):
            row_up = result['up'][i]
            row_down = result['down'][i]
            expected_warmup = length
            # Check warmup NaNs
            assert np.all(np.isnan(row_up[:expected_warmup])), f"Expected NaN in up warmup for length {length}"
            assert np.all(np.isnan(row_down[:expected_warmup])), f"Expected NaN in down warmup for length {length}"
            # Check we have values after warmup
            if expected_warmup < 100:
                assert not np.all(np.isnan(row_up[expected_warmup:])), f"Expected values in up after warmup for length {length}"
                assert not np.all(np.isnan(row_down[expected_warmup:])), f"Expected values in down after warmup for length {length}"
    
    def test_aroon_kernel_selection(self, test_data):
        """Test Aroon with different kernel selections"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        
        # Test with scalar kernel
        result_scalar = ta_indicators.aroon(high, low, length=14, kernel="scalar")
        assert len(result_scalar['up']) == 100
        assert len(result_scalar['down']) == 100
        
        # Test with auto kernel (default)
        result_auto = ta_indicators.aroon(high, low, length=14)
        assert len(result_auto['up']) == 100
        assert len(result_auto['down']) == 100
        
        # Results should be very close regardless of kernel
        # Skip NaN values in comparison
        mask_up = ~(np.isnan(result_scalar['up']) | np.isnan(result_auto['up']))
        mask_down = ~(np.isnan(result_scalar['down']) | np.isnan(result_auto['down']))
        
        if np.any(mask_up):
            assert_close(
                result_scalar['up'][mask_up],
                result_auto['up'][mask_up],
                rtol=1e-10,
                msg="Kernel up results should match"
            )
        
        if np.any(mask_down):
            assert_close(
                result_scalar['down'][mask_down],
                result_auto['down'][mask_down],
                rtol=1e-10,
                msg="Kernel down results should match"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])