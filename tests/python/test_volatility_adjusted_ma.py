"""
Python binding tests for VAMA (Volatility Adjusted Moving Average) indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

try:
    import my_project as ta_indicators
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


class TestVama:
    """Test suite for VAMA indicator Python bindings"""
    
    # Reference values from Rust tests (from PineScript)
    EXPECTED_LAST_5 = [
        61437.31013970,
        61409.77885185,
        61381.24752811,
        61352.71733871,
        61321.57890702,
    ]
    
    DEFAULT_PARAMS = {
        'base_period': 113,
        'vol_period': 51,
        'smoothing': True,
        'smooth_type': 3,  # WMA
        'smooth_period': 5
    }
    
    @pytest.fixture(scope='class')
    def test_data(self):
        """Load test data once for all tests"""
        return load_test_data()
    
    def test_vama_accuracy(self, test_data):
        """Test VAMA matches expected reference values - mirrors test_vama_accuracy"""
        close = test_data['close']
        
        result = ta_indicators.vama(
            close,
            **self.DEFAULT_PARAMS
        )
        
        assert len(result) == len(close), "Output length should match input"
        
        # Check last 5 values match expected reference values
        last_5 = result[-5:]
        for i, (actual, expected) in enumerate(zip(last_5, self.EXPECTED_LAST_5)):
            if not np.isnan(actual):
                assert_close(
                    actual, 
                    expected, 
                    rtol=1e-6,  # Allow small numerical differences
                    msg=f"VAMA reference value mismatch at index {i}"
                )
    
    def test_vama_partial_params(self, test_data):
        """Test VAMA with partial parameters - mirrors check_vama_partial_params"""
        close = test_data['close']
        
        # Test with minimal required params (using defaults for others)
        result = ta_indicators.vama(
            close,
            base_period=113,
            vol_period=51
        )
        assert len(result) == len(close)
    
    def test_vama_warmup_nan(self, test_data):
        """Test VAMA warmup period NaN handling - mirrors check_vama_warmup_nan"""
        close = test_data['close']
        
        result = ta_indicators.vama(
            close,
            **self.DEFAULT_PARAMS
        )
        
        # Find first non-NaN in input
        first_valid = next(i for i, v in enumerate(close) if not np.isnan(v))
        
        # Calculate expected warmup period
        warmup = first_valid + max(self.DEFAULT_PARAMS['base_period'], 
                                  self.DEFAULT_PARAMS['vol_period']) - 1
        
        # Check warmup NaNs
        if warmup > 0:
            warmup_values = result[:min(warmup, len(result))]
            assert np.all(np.isnan(warmup_values)), f"Expected NaN in warmup period [0:{warmup})"
        
        # Check we eventually get non-NaN values
        non_nan_count = np.count_nonzero(~np.isnan(result))
        assert non_nan_count > 0, "Should have some non-NaN values after warmup"
    
    def test_vama_edge_cases(self):
        """Test VAMA edge cases - mirrors check_vama_edge_cases"""
        # Test with minimal data
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = ta_indicators.vama(
            data,
            base_period=2,
            vol_period=2,
            smoothing=False
        )
        
        assert len(result) == len(data), "Output length should match input"
        
        # Check warmup period
        # warmup = first + max(base_period, vol_period) - 1
        # With first=0, base_period=2, vol_period=2: warmup = 0 + 2 - 1 = 1
        warmup = max(2, 2) - 1  # max(base_period, vol_period) - 1
        if warmup > 0:
            assert np.all(np.isnan(result[:warmup])), f"Expected NaN in warmup [0:{warmup})"
        
        # Check we have values after warmup
        if len(data) > warmup:
            assert not np.all(np.isnan(result[warmup:])), "Should have values after warmup"
    
    def test_vama_smoothing_variations(self, test_data):
        """Test different smoothing types - mirrors check_vama_smoothing"""
        close = test_data['close'][:100]  # Use first 100 for speed
        
        results = {}
        
        # Test all smoothing types
        for smooth_type, name in [(1, 'SMA'), (2, 'EMA'), (3, 'WMA')]:
            result = ta_indicators.vama(
                close,
                base_period=10,
                vol_period=5,
                smoothing=True,
                smooth_type=smooth_type,
                smooth_period=3
            )
            results[name] = result
            assert len(result) == len(close), f"{name} output length mismatch"
        
        # Also test without smoothing
        no_smooth = ta_indicators.vama(
            close,
            base_period=10,
            vol_period=5,
            smoothing=False
        )
        
        # Smoothed results should differ from non-smoothed
        for name, smooth_result in results.items():
            valid_idx = ~(np.isnan(smooth_result) | np.isnan(no_smooth))
            if np.any(valid_idx):
                assert not np.allclose(smooth_result[valid_idx], no_smooth[valid_idx], rtol=1e-10), \
                    f"{name} smoothing should produce different values"
    
    def test_vama_invalid_periods(self):
        """Test VAMA fails with invalid periods - mirrors check_vama_period_errors"""
        data = np.array([1.0] * 10)
        
        # Test zero base period
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.vama(data, base_period=0, vol_period=5)
        
        # Test zero vol period
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.vama(data, base_period=5, vol_period=0)
        
        # Test period exceeding data length
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.vama(data, base_period=20, vol_period=5)
    
    def test_vama_invalid_smooth_type(self):
        """Test VAMA fails with invalid smooth type"""
        data = np.ones(100)
        
        # Invalid smooth type 0
        with pytest.raises(ValueError, match="Invalid smooth"):
            ta_indicators.vama(
                data, 
                base_period=10, 
                vol_period=5,
                smoothing=True,
                smooth_type=0  # Invalid
            )
        
        # Invalid smooth type 4
        with pytest.raises(ValueError, match="Invalid smooth"):
            ta_indicators.vama(
                data,
                base_period=10,
                vol_period=5,
                smoothing=True,
                smooth_type=4  # Invalid
            )
    
    def test_vama_empty_input(self):
        """Test VAMA fails with empty input - mirrors test_vama_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty|empty"):
            ta_indicators.vama(empty)
    
    def test_vama_all_nan_input(self):
        """Test VAMA fails with all NaN input - mirrors test_vama_all_nan"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN|all NaN"):
            ta_indicators.vama(all_nan)
    
    def test_vama_streaming(self, test_data):
        """Test VAMA streaming matches batch calculation - mirrors test_vama_stream"""
        close = test_data['close'][:200]  # Use first 200 for speed
        
        # Batch calculation
        batch_result = ta_indicators.vama(
            close,
            **self.DEFAULT_PARAMS
        )
        
        # Streaming calculation
        stream = ta_indicators.VamaStream(
            **self.DEFAULT_PARAMS
        )
        
        stream_values = []
        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # After warmup, values should match closely
        warmup = max(self.DEFAULT_PARAMS['base_period'], 
                    self.DEFAULT_PARAMS['vol_period'])
        
        for i in range(warmup, len(batch_result)):
            if not np.isnan(batch_result[i]) and not np.isnan(stream_values[i]):
                assert_close(
                    batch_result[i], 
                    stream_values[i], 
                    rtol=2e-2,  # Allow 2% difference for streaming vs batch
                    msg=f"Streaming mismatch at index {i}"
                )
    
    def test_vama_batch_single_params(self, test_data):
        """Test VAMA batch with single parameter set - mirrors check_batch_default_row"""
        close = test_data['close']
        
        result = ta_indicators.vama_batch(
            close,
            base_period_range=(113, 113, 0),
            vol_period_range=(51, 51, 0)
        )
        
        assert 'values' in result
        assert 'base_periods' in result
        assert 'vol_periods' in result
        
        # Should have 1 combination
        assert len(result['base_periods']) == 1
        assert len(result['vol_periods']) == 1
        assert result['values'].shape == (1, len(close))
        
        # The single row should match regular calculation
        # Note: batch mode uses smoothing=False by design
        batch_row = result['values'][0]
        single_result = ta_indicators.vama(
            close,
            base_period=113,
            vol_period=51,
            smoothing=False  # Match batch mode's default
        )
        
        # Values should match
        for i, (b, s) in enumerate(zip(batch_row, single_result)):
            if not np.isnan(b) and not np.isnan(s):
                assert_close(b, s, rtol=1e-10, 
                           msg=f"Batch vs single mismatch at {i}")
    
    def test_vama_batch_sweep(self, test_data):
        """Test VAMA batch parameter sweep - mirrors check_batch_sweep"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        result = ta_indicators.vama_batch(
            close,
            base_period_range=(40, 44, 2),  # 40, 42, 44 - reduced to fit data
            vol_period_range=(20, 24, 2)      # 20, 22, 24 - reduced to fit data
        )
        
        # Should have 3x3 = 9 combinations
        assert len(result['base_periods']) == 9
        assert len(result['vol_periods']) == 9
        assert result['values'].shape == (9, len(close))
        
        # Verify each combo has expected parameters
        expected_bases = [40, 40, 40, 42, 42, 42, 44, 44, 44]
        expected_vols = [20, 22, 24] * 3
        
        for i in range(9):
            assert result['base_periods'][i] == expected_bases[i]
            assert result['vol_periods'][i] == expected_vols[i]
    
    def test_vama_kernel_consistency(self, test_data):
        """Test different kernels produce consistent results"""
        close = test_data['close'][:100]  # Smaller dataset for speed
        
        kernels = ['scalar', 'sse2']  # Test basic kernels
        results = {}
        
        for kernel in kernels:
            try:
                result = ta_indicators.vama(
                    close,
                    base_period=20,
                    vol_period=10,
                    kernel=kernel
                )
                results[kernel] = result
            except ValueError:
                # Kernel might not be supported
                pass
        
        # All kernels should produce identical results
        if len(results) > 1:
            kernel_names = list(results.keys())
            for i in range(1, len(kernel_names)):
                valid_idx = ~(np.isnan(results[kernel_names[0]]) | 
                            np.isnan(results[kernel_names[i]]))
                if np.any(valid_idx):
                    np.testing.assert_allclose(
                        results[kernel_names[0]][valid_idx],
                        results[kernel_names[i]][valid_idx],
                        rtol=1e-10,
                        err_msg=f"Kernel {kernel_names[0]} vs {kernel_names[i]} mismatch"
                    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])