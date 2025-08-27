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
        # First pass warmup: first + period - 1 = 0 + 20 - 1 = 19
        # Second pass warmup: first_warmup + period2 - 1 = 19 + 10 - 1 = 28
        expected_warmup = 28
        for i in range(expected_warmup, len(second_result)):
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
    
    def test_vwma_zero_volume(self):
        """Test VWMA handles zero volume correctly"""
        prices = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        volumes = np.array([100.0, 0.0, 300.0, 0.0, 500.0, 0.0, 700.0, 0.0, 900.0, 0.0])
        
        result = ta_indicators.vwma(prices, volumes, 3)
        assert len(result) == len(prices)
        
        # Check that we get NaN where all volumes in window are zero
        # but valid values where at least one volume is non-zero
        assert not np.isnan(result[2])  # Window has non-zero volumes
    
    def test_vwma_partial_nan_data(self, test_data):
        """Test VWMA with NaN values in middle of dataset"""
        close = test_data['close'].copy()
        volume = test_data['volume'].copy()
        
        # Inject NaN values in middle of data
        close[100:110] = np.nan
        volume[100:110] = np.nan
        
        # VWMA should handle NaN gracefully but might propagate them
        # This test verifies the function doesn't crash
        result = ta_indicators.vwma(close, volume, 20)
        assert len(result) == len(close)
        
        # Should have NaN during warmup
        assert np.all(np.isnan(result[:19])), "Expected NaN in warmup period"
        
        # Note: NaN handling depends on implementation - some may continue
        # producing NaN until all data is valid again
    
    def test_vwma_warmup_period(self, test_data):
        """Test VWMA warmup period calculation matches Rust exactly"""
        close = test_data['close']
        volume = test_data['volume']
        period = 20
        
        result = ta_indicators.vwma(close, volume, period)
        
        # Warmup should be first + period - 1 = 0 + 20 - 1 = 19
        # So indices 0-18 should be NaN, index 19 should be first valid
        assert np.all(np.isnan(result[:19])), "First 19 values should be NaN"
        assert not np.isnan(result[19]), "Index 19 should be first valid value"
    
    def test_vwma_batch_multiple_periods(self, test_data):
        """Test VWMA batch with multiple period values"""
        close = test_data['close'][:100]  # Use smaller dataset
        volume = test_data['volume'][:100]
        
        result = ta_indicators.vwma_batch(
            close,
            volume,
            period_range=(10, 30, 10),  # periods: 10, 20, 30
        )
        
        assert 'values' in result
        assert 'periods' in result
        
        # Should have 3 combinations
        assert result['values'].shape[0] == 3
        assert result['values'].shape[1] == 100
        assert len(result['periods']) == 3
        assert np.array_equal(result['periods'], [10, 20, 30])
        
        # Verify each row matches single calculation
        for i, period in enumerate([10, 20, 30]):
            single_result = ta_indicators.vwma(close, volume, period)
            assert_close(
                result['values'][i],
                single_result,
                rtol=1e-10,
                msg=f"Batch period {period} mismatch"
            )
    
    def test_vwma_batch_edge_cases(self):
        """Test VWMA batch with edge case parameters"""
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        volumes = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        
        # Test with step=0 (single value)
        result = ta_indicators.vwma_batch(
            prices,
            volumes,
            period_range=(5, 5, 0),
        )
        assert result['values'].shape[0] == 1
        assert result['periods'] == [5]
        
        # Test with step > range (single value)
        result = ta_indicators.vwma_batch(
            prices,
            volumes,
            period_range=(5, 7, 10),
        )
        assert result['values'].shape[0] == 1
        assert result['periods'] == [5]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
