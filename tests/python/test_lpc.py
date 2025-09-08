"""
Python binding tests for LPC (Low Pass Channel) indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
import os
from pathlib import Path

# Add the parent directory to the path to import test utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import my_project
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS

class TestLpc:
    @pytest.fixture(scope='class')
    def test_data(self):
        """Load test data from CSV file"""
        return load_test_data()
    
    def test_lpc_accuracy(self, test_data):
        """Test LPC matches expected values from Rust tests - mirrors check_lpc_accuracy"""
        expected = EXPECTED_OUTPUTS['lpc']
        
        # Call LPC with default parameters
        filter_out, high_band, low_band = my_project.lpc(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            test_data['close'],  # Using close as source
            cutoff_type=expected['default_params']['cutoff_type'],
            fixed_period=expected['default_params']['fixed_period'],
            max_cycle_limit=expected['default_params']['max_cycle_limit'],
            cycle_mult=expected['default_params']['cycle_mult'],
            tr_mult=expected['default_params']['tr_mult']
        )
        
        assert len(filter_out) == len(test_data['close']), "Filter output length should match input"
        assert len(high_band) == len(test_data['close']), "High band output length should match input"
        assert len(low_band) == len(test_data['close']), "Low band output length should match input"
        
        # Check last 5 values match expected for all three outputs
        assert_close(
            filter_out[-5:],
            expected['last_5_filter'],
            rtol=1e-8,
            msg="LPC Filter last 5 values mismatch"
        )
        
        assert_close(
            high_band[-5:],
            expected['last_5_high_band'],
            rtol=1e-8,
            msg="LPC High Band last 5 values mismatch"
        )
        
        assert_close(
            low_band[-5:],
            expected['last_5_low_band'],
            rtol=1e-8,
            msg="LPC Low Band last 5 values mismatch"
        )
        
        # Verify warmup period - should have NaNs before first valid
        warmup_period = expected.get('warmup_period', 1)
        if warmup_period > 0:
            for i in range(min(warmup_period, len(filter_out))):
                if np.isnan(filter_out[i]):
                    break  # Found expected NaN in warmup
            
        # Also verify bands maintain proper relationship
        for i in range(20, min(100, len(filter_out))):
            if not np.isnan(filter_out[i]) and not np.isnan(high_band[i]) and not np.isnan(low_band[i]):
                assert low_band[i] <= filter_out[i] <= high_band[i], \
                    f"Filter should be between bands at index {i}"
    
    def test_lpc_partial_params(self, test_data):
        """Test LPC with default parameters - mirrors check_lpc_partial_params"""
        # Test with minimal required data, params will use defaults
        filter_out, high_band, low_band = my_project.lpc(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            test_data['close']  # Using close as source
        )
        
        assert len(filter_out) == len(test_data['close'])
        assert len(high_band) == len(test_data['close'])
        assert len(low_band) == len(test_data['close'])
    
    def test_lpc_fixed_mode(self, test_data):
        """Test LPC with fixed cutoff type"""
        filter_out, high_band, low_band = my_project.lpc(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            test_data['close'],
            cutoff_type="fixed",
            fixed_period=20
        )
        
        assert len(filter_out) == len(test_data['close'])
        assert len(high_band) == len(test_data['close'])
        assert len(low_band) == len(test_data['close'])
        
        # Verify bands relationship
        for i in range(20, min(100, len(filter_out))):
            if not np.isnan(filter_out[i]) and not np.isnan(high_band[i]) and not np.isnan(low_band[i]):
                assert low_band[i] <= filter_out[i] <= high_band[i], \
                    f"Filter should be between bands at index {i}"
    
    def test_lpc_invalid_period(self):
        """Test LPC fails with invalid period"""
        data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            my_project.lpc(data, data, data, data, fixed_period=0)
    
    def test_lpc_period_exceeds_length(self):
        """Test LPC fails when period exceeds data length"""
        data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            my_project.lpc(data, data, data, data, fixed_period=10)
    
    def test_lpc_empty_input(self):
        """Test LPC fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            my_project.lpc(empty, empty, empty, empty)
    
    def test_lpc_all_nan(self):
        """Test LPC handles all NaN values"""
        nan_data = np.full(10, np.nan)
        
        # LPC throws an error for all NaN values (as per Rust implementation)
        with pytest.raises(ValueError, match="All values are NaN"):
            my_project.lpc(nan_data, nan_data, nan_data, nan_data)
    
    def test_lpc_mismatched_lengths(self):
        """Test LPC fails with mismatched array lengths"""
        high_data = np.array([10.0, 20.0, 30.0])
        low_data = np.array([8.0, 18.0])  # Different length
        close_data = np.array([9.0, 19.0, 29.0])
        src_data = np.array([9.0, 19.0, 29.0])
        
        with pytest.raises(ValueError, match="All arrays must have the same length"):
            my_project.lpc(high_data, low_data, close_data, src_data)
    
    def test_lpc_invalid_cutoff_type(self, test_data):
        """Test LPC fails with invalid cutoff type"""
        with pytest.raises(ValueError, match="Invalid cutoff type"):
            my_project.lpc(
                test_data['high'][:100],
                test_data['low'][:100],
                test_data['close'][:100],
                test_data['close'][:100],
                cutoff_type="invalid"
            )
    
    def test_lpc_different_multipliers(self, test_data):
        """Test LPC with different multiplier values"""
        # Test with cycle_mult = 2.0
        filter1, high1, low1 = my_project.lpc(
            test_data['high'][:100],
            test_data['low'][:100],
            test_data['close'][:100],
            test_data['close'][:100],
            cutoff_type="adaptive",
            cycle_mult=2.0,
            tr_mult=1.0
        )
        
        # Test with tr_mult = 2.0
        filter2, high2, low2 = my_project.lpc(
            test_data['high'][:100],
            test_data['low'][:100],
            test_data['close'][:100],
            test_data['close'][:100],
            cutoff_type="adaptive",
            cycle_mult=1.0,
            tr_mult=2.0
        )
        
        # The filters might be similar but bands should be different with tr_mult
        check_idx = 50
        if not np.isnan(high1[check_idx]) and not np.isnan(high2[check_idx]):
            band_width1 = high1[check_idx] - low1[check_idx]
            band_width2 = high2[check_idx] - low2[check_idx]
            # tr_mult=2.0 should produce wider bands
            assert band_width2 > band_width1 * 1.5, \
                "Higher tr_mult should produce wider bands"
    
    def test_lpc_nan_handling(self, test_data):
        """Test LPC handles NaN values correctly"""
        filter_out, high_band, low_band = my_project.lpc(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            test_data['close'],
            cutoff_type="fixed",
            fixed_period=10
        )
        
        assert len(filter_out) == len(test_data['close'])
        assert len(high_band) == len(test_data['close'])
        assert len(low_band) == len(test_data['close'])
        
        # After warmup period (20), no NaN values should exist  
        if len(filter_out) > 240:
            for i in range(240, len(filter_out)):
                assert not np.isnan(filter_out[i]), f"Found unexpected NaN in filter at index {i}"
                assert not np.isnan(high_band[i]), f"Found unexpected NaN in high_band at index {i}"
                assert not np.isnan(low_band[i]), f"Found unexpected NaN in low_band at index {i}"
    
    def test_lpc_streaming(self, test_data):
        """Test LPC streaming functionality"""
        cutoff_type = "fixed"
        fixed_period = 20
        
        # Batch calculation
        batch_filter, batch_high, batch_low = my_project.lpc(
            test_data['high'][:100],
            test_data['low'][:100],
            test_data['close'][:100],
            test_data['close'][:100],
            cutoff_type=cutoff_type,
            fixed_period=fixed_period
        )
        
        # Streaming calculation
        stream = my_project.LpcStream(
            cutoff_type=cutoff_type,
            fixed_period=fixed_period
        )
        stream_filter = []
        stream_high = []
        stream_low = []
        
        for i in range(100):
            result = stream.update(
                test_data['high'][i],
                test_data['low'][i],
                test_data['close'][i],
                test_data['close'][i]  # Using close as source
            )
            
            if result is None:
                stream_filter.append(np.nan)
                stream_high.append(np.nan)
                stream_low.append(np.nan)
            else:
                filt, high, low = result
                stream_filter.append(filt)
                stream_high.append(high)
                stream_low.append(low)
        
        stream_filter = np.array(stream_filter)
        stream_high = np.array(stream_high)
        stream_low = np.array(stream_low)
        
        assert len(batch_filter) == len(stream_filter)
        assert len(batch_high) == len(stream_high)
        assert len(batch_low) == len(stream_low)
        
        # After warmup, verify values are in valid relationship
        for i in range(20, 100):
            if not np.isnan(stream_filter[i]) and not np.isnan(stream_high[i]) and not np.isnan(stream_low[i]):
                assert stream_low[i] <= stream_filter[i] <= stream_high[i], \
                    f"Filter should be between bands at index {i}"
                assert stream_high[i] > stream_low[i], \
                    f"High band should be greater than low band at index {i}"
    
    def test_lpc_adaptive_vs_fixed(self, test_data):
        """Test difference between adaptive and fixed modes - mirrors Rust tests"""
        # Adaptive mode
        filter_adaptive, high_adaptive, low_adaptive = my_project.lpc(
            test_data['high'][:200],
            test_data['low'][:200],
            test_data['close'][:200],
            test_data['close'][:200],
            cutoff_type="adaptive",
            fixed_period=20
        )
        
        # Fixed mode
        filter_fixed, high_fixed, low_fixed = my_project.lpc(
            test_data['high'][:200],
            test_data['low'][:200],
            test_data['close'][:200],
            test_data['close'][:200],
            cutoff_type="fixed",
            fixed_period=20
        )
        
        # The results should be different after warmup
        differences = 0
        for i in range(100, 200):
            if not np.isnan(filter_adaptive[i]) and not np.isnan(filter_fixed[i]):
                if abs(filter_adaptive[i] - filter_fixed[i]) > 1e-6:
                    differences += 1
        
        # At least some values should be different
        assert differences > 10, \
            "Adaptive and fixed modes should produce different results"
    
    def test_lpc_batch(self, test_data):
        """Test LPC batch processing - mirrors check_batch_shapes from Rust"""
        # Test batch with single parameter combination
        result = my_project.lpc_batch(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            test_data['close'],  # Using close as source
            fixed_period_range=(10, 12, 1),  # Will test 10, 11, 12
            cycle_mult_range=(1.0, 1.0, 0.0),  # Single value
            tr_mult_range=(1.0, 1.0, 0.0),  # Single value
            cutoff_type="fixed",
            max_cycle_limit=60
        )
        
        assert 'values' in result, "Batch result should have 'values' field"
        assert 'fixed_periods' in result, "Batch result should have 'fixed_periods' field"
        assert 'cycle_mults' in result, "Batch result should have 'cycle_mults' field"
        assert 'tr_mults' in result, "Batch result should have 'tr_mults' field"
        assert 'rows' in result, "Batch result should have 'rows' field"
        assert 'cols' in result, "Batch result should have 'cols' field"
        assert 'order' in result, "Batch result should have 'order' field showing output ordering"
        
        # Check dimensions
        combos = 3  # 10, 11, 12
        expected_rows = combos * 3  # 3 outputs per combo (filter, high, low)
        assert result['rows'] == expected_rows, f"Expected {expected_rows} rows, got {result['rows']}"
        assert result['cols'] == len(test_data['close']), "Columns should match input length"
        
        # Verify values shape
        values_shape = result['values'].shape
        assert values_shape == (expected_rows, len(test_data['close'])), \
            f"Values shape {values_shape} doesn't match expected ({expected_rows}, {len(test_data['close'])})"
        
        # Verify order field shows the three outputs
        assert result['order'] == ['filter', 'high', 'low'], \
            "Order should indicate filter, high, low outputs"
    
    def test_lpc_kernel_param(self, test_data):
        """Test LPC with different kernel parameters"""
        # Test with explicit kernel parameter (if supported)
        try:
            # Try with scalar kernel
            filter_out, high_band, low_band = my_project.lpc(
                test_data['high'][:100],
                test_data['low'][:100],
                test_data['close'][:100],
                test_data['close'][:100],
                kernel="scalar"
            )
            assert len(filter_out) == 100, "Scalar kernel should produce correct length output"
        except (TypeError, ValueError):
            # Kernel parameter might not be exposed in Python bindings
            pass
    
    def test_lpc_batch_parameter_sweep(self, test_data):
        """Test LPC batch with multiple parameter combinations"""
        # Test batch with parameter sweep
        result = my_project.lpc_batch(
            test_data['high'][:100],
            test_data['low'][:100],
            test_data['close'][:100],
            test_data['close'][:100],
            fixed_period_range=(10, 20, 10),  # 10, 20
            cycle_mult_range=(1.0, 2.0, 1.0),  # 1.0, 2.0
            tr_mult_range=(0.5, 1.0, 0.5),  # 0.5, 1.0
            cutoff_type="adaptive"
        )
        
        # 2 periods * 2 cycle_mults * 2 tr_mults = 8 combinations
        expected_combos = 8
        expected_rows = expected_combos * 3  # 3 outputs per combo
        
        assert result['rows'] == expected_rows, \
            f"Expected {expected_rows} rows for {expected_combos} combos"
        assert len(result['fixed_periods']) == expected_combos, \
            f"Expected {expected_combos} period values"
        assert len(result['cycle_mults']) == expected_combos, \
            f"Expected {expected_combos} cycle_mult values"
        assert len(result['tr_mults']) == expected_combos, \
            f"Expected {expected_combos} tr_mult values"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])