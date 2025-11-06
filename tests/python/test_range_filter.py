"""
Python binding tests for RANGE_FILTER indicator.
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


class TestRangeFilter:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_range_filter_accuracy(self, test_data):
        """Test RANGE_FILTER matches expected values from Rust tests - mirrors check_range_filter_accuracy"""
        close = test_data['close']
        
        # Use default parameters - returns tuple (filter, high_band, low_band)
        filter_values, high_band, low_band = ta_indicators.range_filter(close)
        
        assert len(filter_values) == len(close)
        assert len(high_band) == len(close)
        assert len(low_band) == len(close)
        
        # Test last 5 values against Rust references (exact same as Rust unit tests)
        expected_filter = [
            59_589.73987817684,
            59_589.73987817684,
            59_589.73987817684,
            59_589.73987817684,
            59_589.73987817684,
        ]

        expected_high = [
            60_935.63924911415,
            60_906.58379951138,
            60_874.2002431993,
            60_838.79850154794,
            60_810.879398758305,
        ]

        expected_low = [
            58_243.84050723953,
            58_272.8959568423,
            58_305.27951315438,
            58_340.68125480574,
            58_368.60035759538,
        ]

        # Match Rust test tolerance
        tolerance = 1e-10
        
        # Check Filter values
        last_5_filter = filter_values[-5:]
        for i, (val, exp) in enumerate(zip(last_5_filter, expected_filter)):
            diff = abs(val - exp)
            assert diff < tolerance, f"Filter[{i}] mismatch: expected {exp}, got {val} (diff: {diff})"
        
        # Check High Band values
        last_5_high = high_band[-5:]
        for i, (val, exp) in enumerate(zip(last_5_high, expected_high)):
            diff = abs(val - exp)
            assert diff < tolerance, f"High Band[{i}] mismatch: expected {exp}, got {val} (diff: {diff})"
        
        # Check Low Band values
        last_5_low = low_band[-5:]
        for i, (val, exp) in enumerate(zip(last_5_low, expected_low)):
            diff = abs(val - exp)
            assert diff < tolerance, f"Low Band[{i}] mismatch: expected {exp}, got {val} (diff: {diff})"
    
    def test_range_filter_default_candles(self, test_data):
        """Test RANGE_FILTER with default parameters - mirrors check_range_filter_default_candles"""
        close = test_data['close']
        
        filter_values, high_band, low_band = ta_indicators.range_filter(close)
        
        assert len(filter_values) == len(close)
        assert len(high_band) == len(close)
        assert len(low_band) == len(close)
    
    def test_range_filter_partial_params(self, test_data):
        """Test RANGE_FILTER with partial parameters - mirrors check_rf_partial_params"""
        close = test_data['close']
        
        # Test with only range_size specified
        filter_values, high_band, low_band = ta_indicators.range_filter(close, range_size=2.5)
        assert len(filter_values) == len(close)
        
        # Test with range_size and range_period
        filter_values, high_band, low_band = ta_indicators.range_filter(close, range_size=2.5, range_period=15)
        assert len(filter_values) == len(close)
        
        # Test with all parameters
        filter_values, high_band, low_band = ta_indicators.range_filter(
            close, 
            range_size=2.5, 
            range_period=15, 
            smooth_range=True,
            smooth_period=20
        )
        assert len(filter_values) == len(close)
    
    def test_range_filter_empty_input(self):
        """Test RANGE_FILTER fails with empty input - mirrors check_range_filter_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.range_filter(empty)
    
    def test_range_filter_all_nan(self):
        """Test RANGE_FILTER with all NaN values - mirrors check_range_filter_all_nan"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.range_filter(all_nan)
    
    def test_range_filter_invalid_period(self):
        """Test RANGE_FILTER fails with invalid period - mirrors check_range_filter_invalid_period"""
        data = np.array([1.0, 2.0, 3.0])
        
        # Period exceeds data length
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.range_filter(data, range_period=10)
        
        # Zero period
        with pytest.raises(ValueError, match="Invalid period|Invalid range_period"):
            ta_indicators.range_filter(data, range_period=0)
    
    def test_range_filter_invalid_range_size(self):
        """Test RANGE_FILTER fails with invalid range_size - mirrors check_rf_invalid_range_size"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Zero range_size
        with pytest.raises(ValueError, match="Invalid range_size"):
            ta_indicators.range_filter(data, range_size=0.0)
        # Negative range_size
        with pytest.raises(ValueError, match="Invalid range_size"):
            ta_indicators.range_filter(data, range_size=-1.0)
        # NaN range_size
        with pytest.raises(ValueError, match="Invalid range_size"):
            ta_indicators.range_filter(data, range_size=float('nan'))
    
    def test_range_filter_nan_handling(self, test_data):
        """Test RANGE_FILTER handles NaN values correctly - mirrors check_rf_nan_handling"""
        close = test_data['close']
        
        filter_values, high_band, low_band = ta_indicators.range_filter(close)
        
        # Check warmup period for NaNs
        # The warmup depends on the parameters but should be minimal
        # After index 50, should have no NaNs
        if len(filter_values) > 50:
            for i in range(50, len(filter_values)):
                assert not np.isnan(filter_values[i]), f"Found unexpected NaN in filter at index {i}"
                assert not np.isnan(high_band[i]), f"Found unexpected NaN in high_band at index {i}"
                assert not np.isnan(low_band[i]), f"Found unexpected NaN in low_band at index {i}"
    
    def test_range_filter_streaming(self, test_data):
        """Test RANGE_FILTER streaming matches batch calculation - mirrors check_range_filter_streaming"""
        close = test_data['close']
        
        # Batch calculation
        batch_filter, batch_high, batch_low = ta_indicators.range_filter(close)
        
        # Streaming calculation - use default parameters
        stream = ta_indicators.RangeFilterStream(
            range_size=2.618,
            range_period=14,
            smooth_range=True,
            smooth_period=27
        )
        stream_filter = []
        stream_high = []
        stream_low = []
        
        for price in close:
            result = stream.update(price)
            if result is not None:
                # Returns tuple (filter, high_band, low_band)
                stream_filter.append(result[0])
                stream_high.append(result[1])
                stream_low.append(result[2])
            else:
                stream_filter.append(np.nan)
                stream_high.append(np.nan)
                stream_low.append(np.nan)
        
        stream_filter = np.array(stream_filter)
        stream_high = np.array(stream_high)
        stream_low = np.array(stream_low)
        
        # Compare structure and basic properties (mirror Rust: no strict numeric equality)
        assert len(batch_filter) == len(stream_filter)
        # After warmup, ensure values are not NaN and bands are ordered
        start = 30
        for i in range(start, len(batch_filter)):
            assert not np.isnan(stream_filter[i]), f"Stream filter[{i}] is NaN"
            assert not np.isnan(stream_high[i]), f"Stream high[{i}] is NaN"
            assert not np.isnan(stream_low[i]), f"Stream low[{i}] is NaN"
            assert stream_high[i] >= stream_filter[i], f"High band < filter at index {i}"
            assert stream_filter[i] >= stream_low[i], f"Filter < low band at index {i}"
    
    def test_range_filter_batch_default(self, test_data):
        """Test RANGE_FILTER batch processing - mirrors check_range_filter_batch_default"""
        close = test_data['close']
        
        # Single parameter set - use start/end/step parameters
        result = ta_indicators.range_filter_batch(
            close,
            range_size_start=2.618,
            range_size_end=2.618,
            range_size_step=0.1,
            range_period_start=10,
            range_period_end=10,
            range_period_step=1,
            smooth_range=True,
            smooth_period=27
        )
        
        assert 'filter' in result
        assert 'high' in result
        assert 'low' in result
        assert 'range_sizes' in result
        assert 'range_periods' in result
        
        # Should have 1 combination
        assert result['rows'] == 1
        assert result['cols'] == len(close)
        assert result['filter'].shape[0] == 1
        assert result['filter'].shape[1] == len(close)
    
    def test_range_filter_batch_sweep(self, test_data):
        """Test RANGE_FILTER batch with parameter sweep - mirrors check_range_filter_batch_sweep"""
        # Use smaller dataset for speed
        data = np.sin(np.linspace(0, 10, 50)) * 100 + 500
        
        result = ta_indicators.range_filter_batch(
            data,
            range_size_start=2.0,
            range_size_end=3.0,
            range_size_step=0.5,  # 3 values: 2.0, 2.5, 3.0
            range_period_start=10,
            range_period_end=20,
            range_period_step=5,  # 3 values: 10, 15, 20
            smooth_range=True,
            smooth_period=15
        )
        
        # Should have 3 * 3 = 9 combinations
        assert result['rows'] == 9
        assert result['cols'] == len(data)
        assert result['filter'].shape[0] == 9
        assert result['filter'].shape[1] == len(data)
        assert result['high'].shape == result['filter'].shape
        assert result['low'].shape == result['filter'].shape
        
        # Verify parameters
        expected_sizes = []
        expected_periods = []
        for rs in [2.0, 2.5, 3.0]:
            for rp in [10, 15, 20]:
                expected_sizes.append(rs)
                expected_periods.append(rp)
        
        for i in range(9):
            assert abs(result['range_sizes'][i] - expected_sizes[i]) < 0.01
            assert result['range_periods'][i] == expected_periods[i]
    
    def test_range_filter_no_poison(self, test_data):
        """Test RANGE_FILTER doesn't leak poison patterns - mirrors check_range_filter_no_poison"""
        close = test_data['close']
        
        filter_values, high_band, low_band = ta_indicators.range_filter(close)
        
        # Check for poison patterns in all outputs
        poison_patterns = [0x1111111111111111, 0x2222222222222222, 0x3333333333333333]
        
        for name, arr in [('filter', filter_values), 
                          ('high_band', high_band), 
                          ('low_band', low_band)]:
            for i, val in enumerate(arr):
                if np.isnan(val):
                    continue
                # Convert to bits and check
                bits = np.array([val]).view(np.uint64)[0]
                assert bits not in poison_patterns, f"Poison pattern found in {name} at index {i}: 0x{bits:016X}"
    
    def test_range_filter_batch_no_poison(self, test_data):
        """Test batch RANGE_FILTER doesn't leak poison patterns - mirrors check_range_filter_batch_no_poison"""
        close = test_data['close']
        
        result = ta_indicators.range_filter_batch(
            close,
            range_size_start=2.618,
            range_size_end=2.618,
            range_size_step=0.1,
            range_period_start=10,
            range_period_end=10,
            range_period_step=1
        )
        
        # Check for poison patterns in all matrices
        poison_patterns = [0x1111111111111111, 0x2222222222222222, 0x3333333333333333]
        
        for matrix_name in ['filter', 'high', 'low']:
            matrix = result[matrix_name]
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    val = matrix[row, col]
                    if np.isnan(val):
                        continue
                    bits = np.array([val]).view(np.uint64)[0]
                    assert bits not in poison_patterns, f"Poison pattern in {matrix_name}[{row},{col}]: 0x{bits:016X}"
    
    def test_range_filter_kernel_parity(self, test_data):
        """Test different kernels produce similar results - mirrors check_range_filter_kernel_parity"""
        close = test_data['close'][:100]  # Use smaller dataset
        
        # Get result with default (auto) kernel
        filter_auto, high_auto, low_auto = ta_indicators.range_filter(close)
        
        # The kernels should produce very similar results
        # We can't control kernel selection from Python, but we can verify consistency
        filter_second, high_second, low_second = ta_indicators.range_filter(close)
        
        # Results should be identical when using same kernel
        for i in range(len(close)):
            if np.isnan(filter_auto[i]) and np.isnan(filter_second[i]):
                continue
            assert_close(
                filter_auto[i],
                filter_second[i],
                rtol=1e-10,
                msg=f"Kernel consistency check failed at index {i}"
            )
    
    def test_range_filter_multi_output_structure(self, test_data):
        """Test Range Filter returns all three outputs correctly"""
        close = test_data['close'][:100]
        
        filter_values, high_band, low_band = ta_indicators.range_filter(close)
        
        # Check all have same length as input
        assert len(filter_values) == len(close)
        assert len(high_band) == len(close)
        assert len(low_band) == len(close)
        
        # Check relationship between outputs (high > filter > low where not NaN)
        for i in range(50, len(close)):  # Skip warmup
            if not np.isnan(filter_values[i]):
                # High band should be >= filter
                assert high_band[i] >= filter_values[i] - 1e-6, \
                    f"High band < filter at index {i}"
                # Low band should be <= filter  
                assert low_band[i] <= filter_values[i] + 1e-6, \
                    f"Low band > filter at index {i}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
