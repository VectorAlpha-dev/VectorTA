"""
Python binding tests for Pivot indicator.
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


class TestPivot:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_pivot_default_mode_camarilla(self, test_data):
        """Test Pivot with default mode (Camarilla) - mirrors check_pivot_default_mode_camarilla"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        open_ = test_data['open']
        
        # Default mode is 3 (Camarilla)
        r4, r3, r2, r1, pp, s1, s2, s3, s4 = ta_indicators.pivot(high, low, close, open_)
        
        assert len(r4) == len(close)
        assert len(r3) == len(close)
        assert len(r2) == len(close)
        assert len(r1) == len(close)
        assert len(pp) == len(close)
        assert len(s1) == len(close)
        assert len(s2) == len(close)
        assert len(s3) == len(close)
        assert len(s4) == len(close)
        
        # Spot-check Camarilla outputs for last few points
        expected_r4 = [59466.5, 59357.55, 59243.6, 59334.85, 59170.35]
        assert_close(
            r4[-5:],
            expected_r4,
            rtol=1e-1,
            msg="Camarilla r4 mismatch"
        )
    
    def test_pivot_standard_mode(self, test_data):
        """Test Pivot with Standard mode - mirrors check_pivot_standard_mode"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        open_ = test_data['open']
        
        # Mode 0 is Standard
        r4, r3, r2, r1, pp, s1, s2, s3, s4 = ta_indicators.pivot(high, low, close, open_, mode=0)
        
        assert len(r2) == len(close)
        # In Standard mode, only r2, r1, s1, s2 are calculated
        # r4, r3, s3, s4 should be NaN
        assert np.all(np.isnan(r4)), "r4 should be NaN in Standard mode"
        assert np.all(np.isnan(r3)), "r3 should be NaN in Standard mode"
        assert np.all(np.isnan(s3)), "s3 should be NaN in Standard mode"
        assert np.all(np.isnan(s4)), "s4 should be NaN in Standard mode"
    
    def test_pivot_fibonacci_mode(self, test_data):
        """Test Pivot with Fibonacci mode - mirrors check_pivot_fibonacci_mode"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        open_ = test_data['open']
        
        # Mode 1 is Fibonacci
        r4, r3, r2, r1, pp, s1, s2, s3, s4 = ta_indicators.pivot(high, low, close, open_, mode=1)
        
        assert len(r3) == len(close)
        # In Fibonacci mode, r4 and s4 should be NaN
        assert np.all(np.isnan(r4)), "r4 should be NaN in Fibonacci mode"
        assert np.all(np.isnan(s4)), "s4 should be NaN in Fibonacci mode"
    
    def test_pivot_demark_mode(self, test_data):
        """Test Pivot with Demark mode - mirrors check_pivot_demark_mode"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        open_ = test_data['open']
        
        # Mode 2 is Demark
        r4, r3, r2, r1, pp, s1, s2, s3, s4 = ta_indicators.pivot(high, low, close, open_, mode=2)
        
        assert len(r1) == len(close)
        # In Demark mode, only r1, s1, and pp are calculated
        assert np.all(np.isnan(r4)), "r4 should be NaN in Demark mode"
        assert np.all(np.isnan(r3)), "r3 should be NaN in Demark mode"
        assert np.all(np.isnan(r2)), "r2 should be NaN in Demark mode"
        assert np.all(np.isnan(s2)), "s2 should be NaN in Demark mode"
        assert np.all(np.isnan(s3)), "s3 should be NaN in Demark mode"
        assert np.all(np.isnan(s4)), "s4 should be NaN in Demark mode"
    
    def test_pivot_woodie_mode(self, test_data):
        """Test Pivot with Woodie mode - mirrors check_pivot_woodie_mode"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        open_ = test_data['open']
        
        # Mode 4 is Woodie
        r4, r3, r2, r1, pp, s1, s2, s3, s4 = ta_indicators.pivot(high, low, close, open_, mode=4)
        
        assert len(r4) == len(close)
        # In Woodie mode, all levels are calculated
    
    def test_pivot_nan_values(self):
        """Test Pivot with NaN values - mirrors check_pivot_nan_values"""
        high = np.array([10.0, np.nan, 30.0])
        low = np.array([9.0, 8.5, np.nan])
        close = np.array([9.5, 9.0, 29.0])
        open_ = np.array([9.1, 8.8, 28.5])
        
        r4, r3, r2, r1, pp, s1, s2, s3, s4 = ta_indicators.pivot(high, low, close, open_, mode=3)
        assert len(pp) == len(high)
        
        # Values should be NaN where any input is NaN
        assert np.isnan(pp[1]), "PP should be NaN when high is NaN"
        assert np.isnan(pp[2]), "PP should be NaN when low is NaN"
    
    def test_pivot_no_data(self):
        """Test Pivot with no data - mirrors check_pivot_no_data"""
        high = np.array([])
        low = np.array([])
        close = np.array([])
        open_ = np.array([])
        
        with pytest.raises(ValueError, match="One or more required fields"):
            ta_indicators.pivot(high, low, close, open_, mode=3)
    
    def test_pivot_all_nan(self):
        """Test Pivot with all NaN values - mirrors check_pivot_all_nan"""
        high = np.array([np.nan, np.nan])
        low = np.array([np.nan, np.nan])
        close = np.array([np.nan, np.nan])
        open_ = np.array([np.nan, np.nan])
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.pivot(high, low, close, open_, mode=3)
    
    def test_pivot_mismatched_lengths(self):
        """Test Pivot with mismatched input lengths"""
        high = np.array([10.0, 20.0])
        low = np.array([9.0, 18.0, 25.0])  # Different length
        close = np.array([9.5, 19.0])
        open_ = np.array([9.2, 18.5])
        
        with pytest.raises(ValueError, match="One or more required fields"):
            ta_indicators.pivot(high, low, close, open_, mode=3)
    
    def test_pivot_streaming(self, test_data):
        """Test Pivot streaming class"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        open_ = test_data['open']
        
        # Batch calculation for comparison
        r4_batch, r3_batch, r2_batch, r1_batch, pp_batch, s1_batch, s2_batch, s3_batch, s4_batch = \
            ta_indicators.pivot(high, low, close, open_, mode=3)
        
        # Streaming calculation
        stream = ta_indicators.PivotStream(mode=3)
        
        # Get the last non-NaN index for comparison
        last_idx = -1
        
        # Get last values from streaming
        r4_s, r3_s, r2_s, r1_s, pp_s, s1_s, s2_s, s3_s, s4_s = stream.update(
            high[last_idx], low[last_idx], close[last_idx], open_[last_idx]
        )
        
        # Compare last values (allowing for single-point calculation differences)
        assert_close(r4_s, r4_batch[last_idx], rtol=1e-5, msg="Streaming r4 mismatch")
        assert_close(r3_s, r3_batch[last_idx], rtol=1e-5, msg="Streaming r3 mismatch")
        assert_close(r2_s, r2_batch[last_idx], rtol=1e-5, msg="Streaming r2 mismatch")
        assert_close(r1_s, r1_batch[last_idx], rtol=1e-5, msg="Streaming r1 mismatch")
        assert_close(pp_s, pp_batch[last_idx], rtol=1e-5, msg="Streaming pp mismatch")
        assert_close(s1_s, s1_batch[last_idx], rtol=1e-5, msg="Streaming s1 mismatch")
        assert_close(s2_s, s2_batch[last_idx], rtol=1e-5, msg="Streaming s2 mismatch")
        assert_close(s3_s, s3_batch[last_idx], rtol=1e-5, msg="Streaming s3 mismatch")
        assert_close(s4_s, s4_batch[last_idx], rtol=1e-5, msg="Streaming s4 mismatch")
    
    def test_pivot_batch(self, test_data):
        """Test Pivot batch processing - mirrors check_pivot_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        open_ = test_data['open']
        
        # Test batch with all 5 modes
        result = ta_indicators.pivot_batch(
            high, low, close, open_,
            mode_range=(0, 4, 1)  # All modes: 0, 1, 2, 3, 4
        )
        
        assert 'values' in result
        assert 'modes' in result
        assert 'rows' in result
        assert 'cols' in result
        assert 'n_levels' in result
        
        assert result['rows'] == 5  # 5 modes
        assert result['cols'] == len(close)
        assert result['n_levels'] == 9  # 9 output levels
        
        # Values should be shaped (rows, cols * 9)
        assert result['values'].shape == (5, len(close) * 9)
        
        # Test single mode batch
        single_result = ta_indicators.pivot_batch(
            high, low, close, open_,
            mode_range=(3, 3, 1)  # Only Camarilla mode
        )
        
        assert single_result['rows'] == 1
        
        # Extract and verify Camarilla values match single calculation
        r4_single, r3_single, r2_single, r1_single, pp_single, s1_single, s2_single, s3_single, s4_single = \
            ta_indicators.pivot(high, low, close, open_, mode=3)
        
        # Extract first row from batch (interleaved values)
        batch_values = single_result['values'][0]
        # Values are interleaved: r4[0], r3[0], r2[0], r1[0], pp[0], s1[0], s2[0], s3[0], s4[0], r4[1], ...
        
        # Check a few values
        for i in range(min(10, len(close))):
            idx = i * 9
            assert_close(batch_values[idx], r4_single[i], rtol=1e-10, 
                        msg=f"Batch r4[{i}] mismatch")
            assert_close(batch_values[idx + 4], pp_single[i], rtol=1e-10, 
                        msg=f"Batch pp[{i}] mismatch")
    
    def test_pivot_kernel_param(self, test_data):
        """Test Pivot with kernel parameter"""
        high = test_data['high'][:100]  # Use smaller dataset
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        open_ = test_data['open'][:100]
        
        # Test with different kernels
        for kernel in ['scalar', 'auto']:
            r4, r3, r2, r1, pp, s1, s2, s3, s4 = ta_indicators.pivot(
                high, low, close, open_, mode=3, kernel=kernel
            )
            assert len(pp) == len(close)


if __name__ == "__main__":
    pytest.main([__file__])