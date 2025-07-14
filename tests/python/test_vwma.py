"""
Python binding tests for VWMA indicator.
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


class TestVwma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_vwma_partial_params(self, test_data):
        """Test VWMA with partial parameters - mirrors check_vwma_partial_params"""
        close = test_data['close']
        volume = test_data['volume']
        
        # Test with default period (20)
        result = ta_indicators.vwma(close, volume, 20)
        assert len(result) == len(close)
        
        # Test with custom period
        result_custom = ta_indicators.vwma(close, volume, 10)
        assert len(result_custom) == len(close)
    
    def test_vwma_accuracy(self, test_data):
        """Test VWMA matches expected values from Rust tests - mirrors check_vwma_accuracy"""
        close = test_data['close']
        volume = test_data['volume']
        expected = EXPECTED_OUTPUTS['vwma']
        
        result = ta_indicators.vwma(
            close,
            volume,
            period=expected['default_params']['period']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected['last_5_values'],
            rtol=1e-3,  # VWMA uses 1e-3 tolerance in Rust tests
            msg="VWMA last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('vwma', result, 'close', expected['default_params'])
    
    def test_vwma_price_volume_mismatch(self):
        """Test VWMA fails when price and volume lengths don't match"""
        prices = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        volumes = np.array([100.0, 200.0, 300.0])  # Shorter array
        
        with pytest.raises(ValueError, match="Price and volume mismatch"):
            ta_indicators.vwma(prices, volumes, 3)
    
    def test_vwma_invalid_period(self):
        """Test VWMA fails with invalid period"""
        prices = np.array([10.0, 20.0, 30.0])
        volumes = np.array([100.0, 200.0, 300.0])
        
        # Period = 0
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.vwma(prices, volumes, 0)
        
        # Period exceeds length
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.vwma(prices, volumes, 10)
    
    def test_vwma_all_nan(self):
        """Test VWMA fails when all values are NaN"""
        prices = np.full(10, np.nan)
        volumes = np.full(10, np.nan)
        
        with pytest.raises(ValueError, match="All"):
            ta_indicators.vwma(prices, volumes, 5)
    
    def test_vwma_not_enough_valid_data(self):
        """Test VWMA fails with insufficient valid data"""
        # First 8 values are NaN, only 2 valid values for period=5
        prices = np.array([np.nan] * 8 + [10.0, 20.0])
        volumes = np.array([np.nan] * 8 + [100.0, 200.0])
        
        with pytest.raises(ValueError, match="Not enough valid"):
            ta_indicators.vwma(prices, volumes, 5)
    
    def test_vwma_with_default_candles(self, test_data):
        """Test VWMA with default parameters - mirrors check_vwma_input_with_default_candles"""
        close = test_data['close']
        volume = test_data['volume']
        
        # Default period is 20
        result = ta_indicators.vwma(close, volume, 20)
        assert len(result) == len(close)
    
    def test_vwma_candles_plus_prices(self, test_data):
        """Test VWMA with custom prices - mirrors check_vwma_candles_plus_prices"""
        close = test_data['close']
        volume = test_data['volume']
        
        # Use slightly modified prices
        custom_prices = close * 1.001
        
        result = ta_indicators.vwma(custom_prices, volume, 20)
        assert len(result) == len(custom_prices)
    
    def test_vwma_slice_reinput(self, test_data):
        """Test VWMA applied twice (re-input) - mirrors check_vwma_slice_data_reinput"""
        close = test_data['close']
        volume = test_data['volume']
        
        # First pass
        first_result = ta_indicators.vwma(close, volume, 20)
        assert len(first_result) == len(close)
        
        # Second pass - use VWMA output as prices, keep same volumes
        second_result = ta_indicators.vwma(first_result, volume, 10)
        assert len(second_result) == len(first_result)
        
        # After warmup, should have valid values
        start = 20 + 10 - 2  # first period + second period - 2
        for i in range(start, len(second_result)):
            assert not np.isnan(second_result[i]), f"Unexpected NaN at index {i}"
    
    def test_vwma_nan_handling(self, test_data):
        """Test VWMA handles NaN values correctly"""
        close = test_data['close']
        volume = test_data['volume']
        
        result = ta_indicators.vwma(close, volume, 20)
        assert len(result) == len(close)
        
        # First period-1 values should be NaN (warmup)
        assert np.all(np.isnan(result[:19])), "Expected NaN in warmup period"
        
        # After warmup period, no NaN values should exist
        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"
    
    def test_vwma_streaming(self, test_data):
        """Test VWMA streaming matches batch calculation"""
        close = test_data['close']
        volume = test_data['volume']
        period = 20
        
        # Batch calculation
        batch_result = ta_indicators.vwma(close, volume, period)
        
        # Streaming calculation
        stream = ta_indicators.VwmaStream(period)
        stream_values = []
        
        for price, vol in zip(close, volume):
            result = stream.update(price, vol)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values where both are not NaN
        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9, 
                        msg=f"VWMA streaming mismatch at index {i}")
    
    def test_vwma_batch(self, test_data):
        """Test VWMA batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        volume = test_data['volume']
        
        result = ta_indicators.vwma_batch(
            close,
            volume,
            period_range=(20, 20, 0),  # Default period only
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        expected = EXPECTED_OUTPUTS['vwma']['last_5_values']
        
        # Check last 5 values match
        assert_close(
            default_row[-5:],
            expected,
            rtol=1e-3,  # VWMA uses 1e-3 tolerance
            msg="VWMA batch default row mismatch"
        )
    
    def test_vwma_kernel_parameter(self, test_data):
        """Test VWMA with different kernel parameters"""
        close = test_data['close']
        volume = test_data['volume']
        
        # Test different kernels
        kernels = ['auto', 'scalar', 'avx2', 'avx512']
        for kernel in kernels:
            try:
                result = ta_indicators.vwma(close, volume, 20, kernel=kernel)
                assert len(result) == len(close)
            except ValueError as e:
                # AVX kernels might not be available on all systems
                if "Unknown kernel" not in str(e) and "not available on this CPU" not in str(e) and "not compiled in this build" not in str(e):
                    raise


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
