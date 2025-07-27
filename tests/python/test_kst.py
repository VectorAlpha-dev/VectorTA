"""
Python binding tests for KST indicator.
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


class TestKst:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_kst_default_params(self, test_data):
        """Test KST with default parameters - mirrors check_kst_default_params"""
        close = test_data['close']
        
        result = ta_indicators.kst(close)  # Using defaults
        assert 'line' in result
        assert 'signal' in result
        assert len(result['line']) == len(close)
        assert len(result['signal']) == len(close)
    
    def test_kst_accuracy(self, test_data):
        """Test KST matches expected values from Rust tests - mirrors check_kst_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['kst']
        
        result = ta_indicators.kst(close)  # Use defaults which should match expected params
        
        assert len(result['line']) == len(close)
        assert len(result['signal']) == len(close)
        
        # Check last 5 values match expected with 1e-1 tolerance (as in Rust tests)
        assert_close(
            result['line'][-5:],
            expected['last_5_line'],
            rtol=0,
            atol=1e-1,
            msg="KST line last 5 values mismatch"
        )
        
        assert_close(
            result['signal'][-5:],
            expected['last_5_signal'],
            rtol=0,
            atol=1e-1,
            msg="KST signal last 5 values mismatch"
        )
        
        # Also run full comparison with Rust for line values
        compare_with_rust('kst_line', result['line'])
        compare_with_rust('kst_signal', result['signal'])
    
    def test_kst_partial_params(self, test_data):
        """Test KST with partial parameters"""
        close = test_data['close']
        
        # Test with some custom params
        result = ta_indicators.kst(
            close,
            sma_period1=12,
            roc_period1=12,
            signal_period=10
        )  # Others use defaults
        assert len(result['line']) == len(close)
        assert len(result['signal']) == len(close)
    
    def test_kst_zero_period(self):
        """Test KST fails with zero period"""
        data = np.array([10.0, 20.0, 30.0] * 20)  # Need enough data
        
        with pytest.raises(ValueError):
            ta_indicators.kst(data, sma_period1=0)
            
        with pytest.raises(ValueError):
            ta_indicators.kst(data, roc_period1=0)
            
        with pytest.raises(ValueError):
            ta_indicators.kst(data, signal_period=0)
    
    def test_kst_period_exceeds_data(self):
        """Test KST fails when period exceeds data length"""
        small_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError):
            ta_indicators.kst(small_data, roc_period4=50)  # Default is 30, but 50 exceeds data
    
    def test_kst_nan_handling(self, test_data):
        """Test KST handles NaN values correctly - mirrors check_kst_nan_handling"""
        all_nan = np.full(10, np.nan)
        
        with pytest.raises(ValueError):
            ta_indicators.kst(all_nan)
    
    def test_kst_all_params(self, test_data):
        """Test KST with all parameters specified"""
        close = test_data['close']
        
        result = ta_indicators.kst(
            close,
            sma_period1=10,
            sma_period2=10,
            sma_period3=10,
            sma_period4=15,
            roc_period1=10,
            roc_period2=15,
            roc_period3=20,
            roc_period4=30,
            signal_period=9
        )
        
        assert len(result['line']) == len(close)
        assert len(result['signal']) == len(close)
    
    def test_kst_kernel_selection(self, test_data):
        """Test KST with different kernel selections"""
        close = test_data['close']
        
        # Test with explicit kernel selection
        result_auto = ta_indicators.kst(close, kernel=None)  # Auto-detect
        result_scalar = ta_indicators.kst(close, kernel='scalar')
        
        assert len(result_auto['line']) == len(close)
        assert len(result_scalar['line']) == len(close)
        
        # Results should be very close regardless of kernel
        assert_close(result_auto['line'], result_scalar['line'], rtol=1e-10)
        assert_close(result_auto['signal'], result_scalar['signal'], rtol=1e-10)
    
    def test_kst_streaming(self, test_data):
        """Test KST streaming functionality"""
        close = test_data['close']
        
        # Create stream with default params
        stream = ta_indicators.KstStream()
        
        # Process values one by one
        stream_results = []
        for price in close:
            result = stream.update(price)
            stream_results.append(result)
        
        # Convert None to (NaN, NaN) for comparison
        line_results = []
        signal_results = []
        for r in stream_results:
            if r is None:
                line_results.append(np.nan)
                signal_results.append(np.nan)
            else:
                line_results.append(r[0])  # line value
                signal_results.append(r[1])  # signal value
        
        # Compare with batch calculation
        batch_results = ta_indicators.kst(close)
        
        # Results should match closely
        assert_close(line_results, batch_results['line'], rtol=1e-10)
        assert_close(signal_results, batch_results['signal'], rtol=1e-10)
    
    def test_kst_batch_single_parameter_set(self, test_data):
        """Test batch processing with single parameter combination"""
        close = test_data['close']
        
        # Single parameter set with defaults
        batch_result = ta_indicators.kst_batch(close)
        
        # Should return a dict with lines, signals and parameters
        assert 'lines' in batch_result
        assert 'signals' in batch_result
        assert 'sma_period1' in batch_result
        assert 'sma_period2' in batch_result
        assert 'sma_period3' in batch_result
        assert 'sma_period4' in batch_result
        assert 'roc_period1' in batch_result
        assert 'roc_period2' in batch_result
        assert 'roc_period3' in batch_result
        assert 'roc_period4' in batch_result
        assert 'signal_period' in batch_result
        
        # Should have shape (1, len(close))
        assert batch_result['lines'].shape == (1, len(close))
        assert batch_result['signals'].shape == (1, len(close))
        
        # Should match single calculation
        single_result = ta_indicators.kst(close)
        assert_close(batch_result['lines'][0], single_result['line'], rtol=1e-10)
        assert_close(batch_result['signals'][0], single_result['signal'], rtol=1e-10)
    
    def test_kst_batch_multiple_parameters(self, test_data):
        """Test batch processing with multiple parameter combinations"""
        close = test_data['close'][:200]  # Use smaller dataset for speed
        
        # Multiple parameter combinations (vary only a few to keep test manageable)
        batch_result = ta_indicators.kst_batch(
            close,
            roc_period1_range=(10, 12, 2),  # 10, 12
            signal_period_range=(8, 10, 2)  # 8, 10
        )
        
        # Should have 2 * 2 = 4 combinations
        assert batch_result['lines'].shape[0] == 4
        assert batch_result['lines'].shape[1] == len(close)
        assert batch_result['signals'].shape[0] == 4
        assert batch_result['signals'].shape[1] == len(close)
        
        # Verify parameter arrays
        assert len(batch_result['roc_period1']) == 4
        assert len(batch_result['signal_period']) == 4
    
    def test_kst_batch_custom_ranges(self, test_data):
        """Test batch processing with custom parameter ranges"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        batch_result = ta_indicators.kst_batch(
            close,
            sma_period1_range=(8, 12, 2),    # 8, 10, 12
            roc_period4_range=(25, 30, 5),   # 25, 30
            signal_period_range=(9, 9, 0)     # Just 9
        )
        
        # Should have 3 * 2 * 1 = 6 combinations
        assert batch_result['lines'].shape[0] == 6
        assert batch_result['signals'].shape[0] == 6
    
    def test_kst_edge_cases(self):
        """Test KST with edge case inputs"""
        # Minimum valid data (enough for largest default period)
        min_data = np.random.rand(50)  # Need at least roc_period4 + sma_period4
        result = ta_indicators.kst(min_data)
        assert len(result['line']) == len(min_data)
        assert len(result['signal']) == len(min_data)
        
        # Check warmup period - should have NaN values at start
        # With default params, warmup is approximately roc_period4 + sma_period4 - 1 = 44
        assert np.isnan(result['line'][0])
        assert np.isnan(result['signal'][0])
        
        # Should have valid values after warmup
        assert not np.isnan(result['line'][-1])
        assert not np.isnan(result['signal'][-1])
    
    def test_kst_batch_kernel_selection(self, test_data):
        """Test KST batch with different kernels"""
        close = test_data['close'][:100]
        
        # Test with explicit kernel selection
        result_auto = ta_indicators.kst_batch(close, kernel=None)
        result_scalar = ta_indicators.kst_batch(close, kernel='scalar_batch')
        
        # Results should be very close regardless of kernel
        assert_close(result_auto['lines'], result_scalar['lines'], rtol=1e-10)
        assert_close(result_auto['signals'], result_scalar['signals'], rtol=1e-10)