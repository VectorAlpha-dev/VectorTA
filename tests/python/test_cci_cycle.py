"""Python binding tests for CCI_CYCLE indicator.
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

from test_utils import load_test_data, assert_close
from rust_comparison import compare_with_rust


class TestCci_Cycle:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_cci_cycle_partial_params(self, test_data):
        """Test CCI_CYCLE with partial parameters (None values) - mirrors check_cci_cycle_partial_params"""
        close = test_data['close']
        
        # Test with all default params (None)
        result = ta_indicators.cci_cycle(close, 10, 0.5)  # Using defaults
        assert len(result) == len(close)
    
    def test_cci_cycle_accuracy(self, test_data):
        """Test CCI_CYCLE matches expected values from Rust tests - mirrors check_cci_cycle_accuracy"""
        close = test_data['close']
        
        # Default parameters from Rust
        length = 10
        factor = 0.5
        
        result = ta_indicators.cci_cycle(close, length, factor)
        
        assert len(result) == len(close)
        
        # Reference values from PineScript
        expected_last_five = [
            9.25177192,
            20.49219826,
            35.42917181,
            55.57843075,
            77.78921538,
        ]
        
        # Check last 5 values match expected
        assert_close(
            result[-5:], 
            expected_last_five,
            rtol=1e-6,
            msg="CCI_CYCLE last 5 values mismatch"
        )
        
        # Compare full output with Rust (skip for now since reference generator doesn't have cci_cycle)
        # compare_with_rust('cci_cycle', result, 'close', {'length': length, 'factor': factor})
    
    def test_cci_cycle_default_candles(self, test_data):
        """Test CCI_CYCLE with default parameters - mirrors check_cci_cycle_default_candles"""
        close = test_data['close']
        
        # Default params: length=10, factor=0.5
        result = ta_indicators.cci_cycle(close, 10, 0.5)
        assert len(result) == len(close)
    
    def test_cci_cycle_zero_period(self):
        """Test CCI_CYCLE fails with zero period - mirrors check_cci_cycle_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cci_cycle(input_data, 0, 0.5)
    
    def test_cci_cycle_period_exceeds_length(self):
        """Test CCI_CYCLE fails when period exceeds data length - mirrors check_cci_cycle_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.cci_cycle(data_small, 10, 0.5)
    
    def test_cci_cycle_very_small_dataset(self):
        """Test CCI_CYCLE fails with insufficient data - mirrors check_cci_cycle_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.cci_cycle(single_point, 10, 0.5)
    
    def test_cci_cycle_empty_input(self):
        """Test CCI_CYCLE fails with empty input - mirrors check_cci_cycle_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty|Empty input data"):
            ta_indicators.cci_cycle(empty, 10, 0.5)
    
    def test_cci_cycle_invalid_factor(self):
        """Test CCI_CYCLE with invalid factor"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                         11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
        
        # Test with NaN factor - CCI Cycle accepts NaN and returns mostly NaN results
        result = ta_indicators.cci_cycle(data, 5, float('nan'))
        assert len(result) == len(data)
        # Most values should be NaN when factor is NaN (some may be 0 due to implementation)
        nan_count = np.sum(np.isnan(result))
        assert nan_count >= len(data) - 5, f"Expected mostly NaN when factor is NaN, got {nan_count}/{len(data)} NaN values"
        
        # Test with negative factor (valid, just testing bounds)
        # CCI Cycle doesn't restrict factor range, so negative is valid
        result = ta_indicators.cci_cycle(data, 5, -0.5)
        assert len(result) == len(data)
        
        # Test with very large factor (valid)
        result = ta_indicators.cci_cycle(data, 5, 10.0)
        assert len(result) == len(data)
    
    def test_cci_cycle_all_nan(self):
        """Test CCI_CYCLE with all NaN values - mirrors check_cci_cycle_all_nan"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN|all NaN"):
            ta_indicators.cci_cycle(all_nan, 10, 0.5)
    
    def test_cci_cycle_nan_handling(self, test_data):
        """Test CCI_CYCLE handles NaN values correctly - mirrors check_cci_cycle_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.cci_cycle(close, 10, 0.5)
        assert len(result) == len(close)
        
        # CCI Cycle has a complex warmup period involving multiple indicators
        # Check that we have NaN values at the beginning
        initial_nans = 0
        for i in range(min(50, len(result))):
            if np.isnan(result[i]):
                initial_nans += 1
            else:
                break
        
        # Should have at least some warmup period
        assert initial_nans > 0, "Expected some NaN values during warmup period"
        
        # After sufficient data, should have valid values
        if len(result) > 100:
            non_nan_count = 0
            for i in range(100, min(200, len(result))):
                if not np.isnan(result[i]):
                    non_nan_count += 1
            assert non_nan_count > 0, "Should have some valid values after sufficient data"
    
    def test_cci_cycle_streaming(self, test_data):
        """Test CCI_CYCLE streaming matches batch calculation - mirrors check_cci_cycle_streaming"""
        close = test_data['close']
        length = 10
        factor = 0.5
        
        # Batch calculation
        batch_result = ta_indicators.cci_cycle(close, length, factor)
        
        # Streaming calculation if available
        try:
            stream = ta_indicators.CciCycleStream(length, factor)
            stream_values = []
            
            for price in close:
                result = stream.update(price)
                stream_values.append(result if result is not None else np.nan)
            
            stream_values = np.array(stream_values)
            
            # Compare batch vs streaming
            assert len(batch_result) == len(stream_values)

            # Note: CCI Cycle streaming warms up more conservatively than batch (length*4).
            # Only compare indices where BOTH batch and streaming are finite.
            if not np.all(np.isnan(stream_values)):
                compared = 0
                for i, (b, s) in enumerate(zip(batch_result, stream_values)):
                    if np.isnan(s) or np.isnan(b):
                        continue
                    # Skip the very first emitted streaming index to avoid seeding edge
                    # effects; subsequent points must match batch.
                    if i < (length * 4):
                        continue
                    assert_close(b, s, rtol=1e-9, atol=1e-9,
                                 msg=f"CCI_CYCLE streaming mismatch at index {i}")
                    compared += 1
                # Ensure we validated a reasonable number of points
                assert compared > 0, "Streaming produced no comparable finite values"
        except AttributeError:
            # Streaming not implemented, skip
            pytest.skip("CciCycleStream not available")
    
    def test_cci_cycle_batch(self, test_data):
        """Test CCI_CYCLE batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        
        try:
            result = ta_indicators.cci_cycle_batch(
                close,
                length_range=(10, 10, 0),  # Default length only
                factor_range=(0.5, 0.5, 0.0)  # Default factor only
            )
            
            assert 'values' in result
            assert 'lengths' in result
            assert 'factors' in result
            
            # Should have 1 combination (default params)
            assert result['values'].shape[0] == 1
            assert result['values'].shape[1] == len(close)
            
            # Extract the single row
            default_row = result['values'][0]
            expected = [
                9.25177192,
                20.49219826,
                35.42917181,
                55.57843075,
                77.78921538,
            ]
            
            # Check last 5 values match
            assert_close(
                default_row[-5:],
                expected,
                rtol=1e-6,
                msg="CCI_CYCLE batch default row mismatch"
            )
        except AttributeError:
            # Batch not implemented, skip
            pytest.skip("cci_cycle_batch not available")
    
    def test_cci_cycle_batch_sweep(self, test_data):
        """Test CCI_CYCLE batch with parameter sweep - mirrors check_batch_sweep"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        try:
            result = ta_indicators.cci_cycle_batch(
                close,
                length_range=(10, 20, 5),   # 10, 15, 20
                factor_range=(0.3, 0.7, 0.2) # 0.3, 0.5, 0.7
            )
            
            # Should have 3 * 3 = 9 combinations
            assert result['values'].shape[0] == 9
            assert result['values'].shape[1] == 100
            
            # Check we have the right parameter values
            assert len(result['lengths']) == 9
            assert len(result['factors']) == 9
            
            # Verify parameter combinations
            expected_lengths = [10, 10, 10, 15, 15, 15, 20, 20, 20]
            expected_factors = [0.3, 0.5, 0.7] * 3
            
            for i in range(9):
                assert result['lengths'][i] == expected_lengths[i]
                assert_close(result['factors'][i], expected_factors[i], rtol=1e-10)
        except AttributeError:
            # Batch not implemented, skip  
            pytest.skip("cci_cycle_batch not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
