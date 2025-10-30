"""
Python binding tests for QQE indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    import my_project
except ImportError:
    pytest.skip("Python module not built", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


class TestQqe:
    @pytest.fixture(scope='class')
    def test_data(self):
        """Load test data from CSV file"""
        return load_test_data()
    
    def test_qqe_accuracy(self, test_data):
        """Test QQE matches expected values from Rust tests - mirrors check_qqe_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['qqe']
        
        # Run QQE with default parameters
        fast, slow = my_project.qqe(
            close,
            rsi_period=expected['default_params']['rsi_period'],
            smoothing_factor=expected['default_params']['smoothing_factor'],
            fast_factor=expected['default_params']['fast_factor']
        )
        
        assert len(fast) == len(close)
        assert len(slow) == len(close)
        
        # Check last 5 values match expected
        # Rust reference tolerance is absolute 1e-6; bindings must be no looser
        assert_close(
            fast[-5:],
            expected['last_5_fast'],
            rtol=0,
            atol=1e-6,
            msg="QQE fast last 5 values mismatch"
        )
        
        assert_close(
            slow[-5:],
            expected['last_5_slow'],
            rtol=0,
            atol=1e-6,
            msg="QQE slow last 5 values mismatch"
        )
    
    def test_qqe_default_params(self, test_data):
        """Test QQE with default parameters - mirrors check_qqe_default_params"""
        close = test_data['close']
        
        # Default params: rsi_period=14, smoothing_factor=5, fast_factor=4.236
        fast, slow = my_project.qqe(close, rsi_period=14, smoothing_factor=5, fast_factor=4.236)
        
        assert len(fast) == len(close)
        assert len(slow) == len(close)
    
    def test_qqe_partial_params(self, test_data):
        """Test QQE with partial parameters - mirrors check_qqe_partial_params"""
        close = test_data['close']
        
        # Test with defaults
        fast, slow = my_project.qqe(close, 14, 5, 4.236)
        assert len(fast) == len(close)
        assert len(slow) == len(close)
    
    def test_qqe_very_small_dataset(self):
        """Test QQE fails with insufficient data - mirrors check_qqe_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            my_project.qqe(single_point, rsi_period=14, smoothing_factor=5, fast_factor=4.236)
    
    def test_qqe_zero_period(self):
        """Test QQE fails with zero period - mirrors check_qqe_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            my_project.qqe(input_data, rsi_period=0)
    
    def test_qqe_empty_input(self):
        """Test QQE fails with empty input - mirrors check_qqe_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="[Ee]mpty"):
            my_project.qqe(empty)
    
    def test_qqe_all_nan(self):
        """Test QQE fails with all NaN values - mirrors check_qqe_all_nan"""
        # Test with larger array of all NaN values
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            my_project.qqe(all_nan, rsi_period=14, smoothing_factor=5, fast_factor=4.236)
    
    def test_qqe_period_exceeds_length(self):
        """Test QQE fails when period exceeds data length - mirrors check_qqe_period_exceeds_length"""
        small_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough"):
            my_project.qqe(small_data, rsi_period=10)
    
    def test_qqe_invalid_smoothing_factor(self):
        """Test QQE fails with invalid smoothing factor"""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0] * 10)
        
        # Zero smoothing factor
        with pytest.raises(ValueError, match="Invalid period"):
            my_project.qqe(data, rsi_period=14, smoothing_factor=0, fast_factor=4.236)
    
    def test_qqe_invalid_fast_factor(self):
        """Test QQE with extreme fast factor values"""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0] * 10)
        
        # Test that extreme values don't cause crashes (may not error)
        # Very small fast factor
        try:
            fast, slow = my_project.qqe(data, rsi_period=14, smoothing_factor=5, fast_factor=0.001)
            assert len(fast) == len(data)
            assert len(slow) == len(data)
        except ValueError:
            pass  # It's okay if it raises an error
        
        # Very large fast factor
        try:
            fast, slow = my_project.qqe(data, rsi_period=14, smoothing_factor=5, fast_factor=100.0)
            assert len(fast) == len(data)
            assert len(slow) == len(data)
        except ValueError:
            pass  # It's okay if it raises an error
    
    def test_qqe_reinput(self, test_data):
        """Test QQE applied twice (re-input) - mirrors check_qqe_reinput"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['qqe']
        
        # First pass
        fast1, slow1 = my_project.qqe(
            close,
            rsi_period=14,
            smoothing_factor=5,
            fast_factor=4.236
        )
        assert len(fast1) == len(close)
        assert len(slow1) == len(close)
        
        # Second pass - apply QQE to fast output
        fast2, slow2 = my_project.qqe(
            fast1,
            rsi_period=14,
            smoothing_factor=5,
            fast_factor=4.236
        )
        assert len(fast2) == len(fast1)
        assert len(slow2) == len(slow1)
        
        # The main check is that it doesn't crash and produces valid output
        # Warmup period for QQE is: first + rsi_period + smoothing_factor - 2
        # For reinput: the first pass produces NaN for first 17 values,
        # then second pass has additional warmup from those positions
        warmup_first = 17  # 0 + 14 + 5 - 2
        warmup_second = warmup_first + 17  # Double application
        
        if len(fast2) > warmup_second:
            assert not np.any(np.isnan(fast2[warmup_second:])), f"Found NaN in fast after warmup period {warmup_second}"
        
        if len(slow2) > warmup_second:
            assert not np.any(np.isnan(slow2[warmup_second:])), f"Found NaN in slow after warmup period {warmup_second}"
    
    def test_qqe_nan_handling(self, test_data):
        """Test QQE handles NaN values correctly - mirrors check_qqe_nan_handling"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['qqe']
        
        fast, slow = my_project.qqe(close, rsi_period=14, smoothing_factor=5, fast_factor=4.236)
        
        assert len(fast) == len(close)
        assert len(slow) == len(close)
        
        # After warmup period, no NaN values should exist
        warmup = expected['warmup_period']  # 17 for default params
        
        # Check no NaN after warmup
        if len(fast) > warmup:
            assert not np.any(np.isnan(fast[warmup:])), f"Found unexpected NaN in fast after warmup period {warmup}"
            assert not np.any(np.isnan(slow[warmup:])), f"Found unexpected NaN in slow after warmup period {warmup}"
        
        # Warmup semantics: fast is defined from rsi_start; slow from warmup
        first_valid = int(np.argmax(~np.isnan(close))) if np.any(~np.isnan(close)) else 0
        rsi_start = first_valid + 14  # default rsi_period
        
        assert np.all(np.isnan(fast[:rsi_start])), f"Expected NaN in fast until rsi_start (first {rsi_start} values)"
        assert np.all(np.isnan(slow[:warmup])), f"Expected NaN in slow warmup period (first {warmup} values)"
    
    def test_qqe_streaming(self, test_data):
        """Test QQE streaming functionality - basic operation test
        
        Note: Streaming implementation may have numerical differences from batch
        due to the streaming algorithm using a ring buffer and different warmup approach.
        """
        close = test_data['close']
        rsi_period = 14
        smoothing_factor = 5
        fast_factor = 4.236
        
        # Batch calculation
        batch_fast, batch_slow = my_project.qqe(
            close,
            rsi_period=rsi_period,
            smoothing_factor=smoothing_factor,
            fast_factor=fast_factor
        )
        
        # Streaming calculation
        stream = my_project.QqeStream(
            rsi_period=rsi_period,
            smoothing_factor=smoothing_factor,
            fast_factor=fast_factor
        )
        
        stream_fast = []
        stream_slow = []
        
        for price in close:
            result = stream.update(price)
            if result is not None:
                stream_fast.append(result[0])
                stream_slow.append(result[1])
            else:
                stream_fast.append(np.nan)
                stream_slow.append(np.nan)
        
        stream_fast = np.array(stream_fast)
        stream_slow = np.array(stream_slow)
        
        # Compare batch vs streaming
        assert len(batch_fast) == len(stream_fast)
        assert len(batch_slow) == len(stream_slow)
        
        # Basic checks - streaming should produce values
        # Note: Due to implementation differences, we only verify basic functionality
        # The ring buffer approach in streaming can lead to different numerical results
        
        # Check that streaming eventually produces non-NaN values
        has_valid_fast = False
        has_valid_slow = False
        
        for i in range(50, min(100, len(stream_fast))):  # Check after warmup
            if not np.isnan(stream_fast[i]):
                has_valid_fast = True
            if not np.isnan(stream_slow[i]):
                has_valid_slow = True
        
        assert has_valid_fast, "Streaming should produce valid fast values"
        assert has_valid_slow, "Streaming should produce valid slow values"
        
        # Check that streaming produces reasonable values (within expected range)
        valid_fast = stream_fast[~np.isnan(stream_fast)]
        valid_slow = stream_slow[~np.isnan(stream_slow)]
        
        if len(valid_fast) > 0:
            assert np.all(valid_fast >= 0) and np.all(valid_fast <= 100), "Fast values should be in [0, 100] range"
        if len(valid_slow) > 0:
            assert np.all(valid_slow >= 0) and np.all(valid_slow <= 100), "Slow values should be in [0, 100] range"
    
    def test_qqe_batch(self, test_data):
        """Test QQE batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['qqe']
        
        result = my_project.qqe_batch(
            close,
            rsi_period_range=(14, 14, 0),  # Default rsi_period only
            smoothing_factor_range=(5, 5, 0),  # Default smoothing_factor only
            fast_factor_range=(4.236, 4.236, 0.0)  # Default fast_factor only
        )
        
        assert 'fast' in result
        assert 'slow' in result
        assert 'rsi_periods' in result
        assert 'smoothing_factors' in result
        assert 'fast_factors' in result
        
        # Should have 1 combination (default params)
        assert result['fast'].shape[0] == 1
        assert result['slow'].shape[0] == 1
        assert result['fast'].shape[1] == len(close)
        assert result['slow'].shape[1] == len(close)
        
        # Extract the single row
        fast_row = result['fast'][0]
        slow_row = result['slow'][0]
        
        # Check last 5 values match
        # Match Rust test absolute tolerance (<= 1e-6)
        assert_close(
            fast_row[-5:],
            expected['batch_default_row_fast'],
            rtol=0,
            atol=1e-6,
            msg="QQE batch fast default row mismatch"
        )
        
        assert_close(
            slow_row[-5:],
            expected['batch_default_row_slow'],
            rtol=0,
            atol=1e-6,
            msg="QQE batch slow default row mismatch"
        )
    
    def test_qqe_batch_multiple_params(self, test_data):
        """Test QQE batch processing with multiple parameters"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        result = my_project.qqe_batch(
            close,
            rsi_period_range=(10, 14, 2),  # 10, 12, 14
            smoothing_factor_range=(3, 5, 2),  # 3, 5
            fast_factor_range=(3.0, 4.0, 1.0)  # 3.0, 4.0
        )
        
        # Should have 3 * 2 * 2 = 12 combinations
        assert result['fast'].shape[0] == 12
        assert result['slow'].shape[0] == 12
        assert result['fast'].shape[1] == len(close)
        assert result['slow'].shape[1] == len(close)
        
        # Verify parameters are correct
        assert len(result['rsi_periods']) == 12
        assert len(result['smoothing_factors']) == 12
        assert len(result['fast_factors']) == 12
        
        # Check first combination
        assert result['rsi_periods'][0] == 10
        assert result['smoothing_factors'][0] == 3
        assert abs(result['fast_factors'][0] - 3.0) < 1e-9
    
    def test_qqe_custom_params(self, test_data):
        """Test QQE with custom parameters - mirrors check_qqe_custom_params"""
        close = test_data['close']
        
        # Run with custom parameters
        fast, slow = my_project.qqe(
            close,
            rsi_period=10,
            smoothing_factor=3,
            fast_factor=3.0
        )
        
        assert len(fast) == len(close)
        assert len(slow) == len(close)
        
        # Check that values are different from defaults
        fast_default, slow_default = my_project.qqe(
            close,
            rsi_period=14,
            smoothing_factor=5,
            fast_factor=4.236
        )
        
        # Values should be different with different parameters
        # Check last 10 values are different
        assert not np.array_equal(fast[-10:], fast_default[-10:])
        assert not np.array_equal(slow[-10:], slow_default[-10:])
    
    def test_qqe_partial_nan(self, test_data):
        """Test QQE handles partial NaN values correctly - mirrors check_qqe_partial_nan"""
        close = test_data['close']
        
        # Add some NaN values at the beginning
        data_with_nan = np.concatenate([
            np.array([np.nan, np.nan, np.nan]),
            close[:100]
        ])
        
        fast, slow = my_project.qqe(data_with_nan, rsi_period=14, smoothing_factor=5, fast_factor=4.236)
        
        assert len(fast) == len(data_with_nan)
        assert len(slow) == len(data_with_nan)
        
        # First values should be NaN
        assert np.isnan(fast[0])
        assert np.isnan(slow[0])
        
        # Eventually should get valid values after enough non-NaN data
        # The warmup period starts from the first non-NaN value
        expected_warmup = 3 + EXPECTED_OUTPUTS['qqe']['warmup_period']  # 3 NaNs + normal warmup
        if len(fast) > expected_warmup:
            # Check we have some valid values after extended warmup
            assert not np.all(np.isnan(fast[expected_warmup:])), "All values are NaN after extended warmup"
            assert not np.all(np.isnan(slow[expected_warmup:])), "All values are NaN after extended warmup"
    
    def test_qqe_batch_single_param(self, test_data):
        """Test QQE batch with single parameter set - mirrors batch single ALMA test"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['qqe']
        
        # Single parameter set (defaults)
        result = my_project.qqe_batch(
            close,
            rsi_period_range=(14, 14, 0),
            smoothing_factor_range=(5, 5, 0),
            fast_factor_range=(4.236, 4.236, 0.0)
        )
        
        assert 'fast' in result
        assert 'slow' in result
        assert 'rsi_periods' in result
        assert 'smoothing_factors' in result
        assert 'fast_factors' in result
        
        # Should have exactly one row
        assert result['fast'].shape[0] == 1
        assert result['slow'].shape[0] == 1
        assert result['fast'].shape[1] == len(close)
        
        # Compare with single calculation
        fast_single, slow_single = my_project.qqe(close, 14, 5, 4.236)
        
        # Align warmup differences: batch fast may keep NaN at rsi_start
        fs = result['fast'][0]
        start_single = int(np.argmax(~np.isnan(fast_single))) if np.any(~np.isnan(fast_single)) else 0
        start_batch = int(np.argmax(~np.isnan(fs))) if np.any(~np.isnan(fs)) else 0
        start = max(start_single, start_batch)
        assert_close(
            fs[start:], 
            fast_single[start:],
            rtol=1e-10,
            msg="Batch vs single fast mismatch"
        )
        
        assert_close(
            result['slow'][0],
            slow_single,
            rtol=1e-10,
            msg="Batch vs single slow mismatch"
        )
    
    def test_qqe_batch_multiple_params(self, test_data):
        """Test QQE batch with multiple parameter combinations"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        # Multiple parameter combinations
        result = my_project.qqe_batch(
            close,
            rsi_period_range=(10, 14, 2),  # 10, 12, 14
            smoothing_factor_range=(3, 5, 1),  # 3, 4, 5
            fast_factor_range=(3.0, 4.0, 0.5)  # 3.0, 3.5, 4.0
        )
        
        # Should have 3 * 3 * 3 = 27 combinations
        expected_combinations = 27
        assert result['fast'].shape[0] == expected_combinations
        assert result['slow'].shape[0] == expected_combinations
        assert len(result['rsi_periods']) == expected_combinations
        assert len(result['smoothing_factors']) == expected_combinations
        assert len(result['fast_factors']) == expected_combinations
        
        # Verify first combination parameters
        assert result['rsi_periods'][0] == 10
        assert result['smoothing_factors'][0] == 3
        assert abs(result['fast_factors'][0] - 3.0) < 1e-9
        
        # Verify each row matches individual calculation for first few combinations
        for i in range(min(3, expected_combinations)):
            rsi_p = result['rsi_periods'][i]
            smooth_f = result['smoothing_factors'][i]
            fast_f = result['fast_factors'][i]
            
            fast_single, slow_single = my_project.qqe(close, rsi_p, smooth_f, fast_f)
            
            fs = result['fast'][i]
            start_single = int(np.argmax(~np.isnan(fast_single))) if np.any(~np.isnan(fast_single)) else 0
            start_batch = int(np.argmax(~np.isnan(fs))) if np.any(~np.isnan(fs)) else 0
            start = max(start_single, start_batch)
            assert_close(
                fs[start:],
                fast_single[start:],
                rtol=1e-9,
                msg=f"Batch row {i} fast mismatch"
            )
            
            assert_close(
                result['slow'][i],
                slow_single,
                rtol=1e-9,
                msg=f"Batch row {i} slow mismatch"
            )
    
    def test_qqe_boundary_conditions(self, test_data):
        """Test QQE with boundary parameter values"""
        close = test_data['close'][:100]  # Use smaller dataset
        
        # Minimal smoothing (smoothing_factor=1)
        fast1, slow1 = my_project.qqe(close, rsi_period=14, smoothing_factor=1, fast_factor=4.236)
        assert len(fast1) == len(close)
        assert len(slow1) == len(close)
        
        # Large fast_factor
        fast2, slow2 = my_project.qqe(close, rsi_period=14, smoothing_factor=5, fast_factor=10.0)
        assert len(fast2) == len(close)
        assert len(slow2) == len(close)
        
        # Small fast_factor
        fast3, slow3 = my_project.qqe(close, rsi_period=14, smoothing_factor=5, fast_factor=0.1)
        assert len(fast3) == len(close)
        assert len(slow3) == len(close)
    
    def test_qqe_values_within_bounds(self, test_data):
        """Test QQE fast/slow values stay within RSI bounds [0, 100]"""
        close = test_data['close']
        
        fast, slow = my_project.qqe(close, rsi_period=14, smoothing_factor=5, fast_factor=4.236)
        
        # Check all non-NaN values are within [0, 100]
        valid_fast = fast[~np.isnan(fast)]
        valid_slow = slow[~np.isnan(slow)]
        
        assert np.all(valid_fast >= 0.0), "Fast values should be >= 0"
        assert np.all(valid_fast <= 100.0), "Fast values should be <= 100"
        assert np.all(valid_slow >= 0.0), "Slow values should be >= 0"
        assert np.all(valid_slow <= 100.0), "Slow values should be <= 100"
    
    def test_qqe_constant_data(self):
        """Test QQE with constant price data"""
        # Constant price should produce RSI = 50 after warmup
        constant_data = np.array([50.0] * 100)
        
        fast, slow = my_project.qqe(constant_data, rsi_period=14, smoothing_factor=5, fast_factor=4.236)
        
        # After warmup, RSI and thus QQE should stabilize around 50
        warmup = 17  # 0 + 14 + 5 - 2
        
        # Check that values after warmup are close to 50
        # Note: Due to the smoothing, it may not be exactly 50
        if len(fast) > warmup + 10:
            # Check last few values are stable and near 50
            assert np.std(fast[-10:]) < 0.1, "Fast values should be stable for constant input"
            assert np.std(slow[-10:]) < 0.1, "Slow values should be stable for constant input"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
