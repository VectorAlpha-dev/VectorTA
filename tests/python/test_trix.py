"""
Python binding tests for TRIX indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'target/wheels'))

try:
    import my_project as ta_indicators
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


class TestTrix:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_trix_accuracy(self, test_data):
        """Test TRIX matches expected values from Rust tests"""
        close_prices = test_data['close']
        period = 18


        result = ta_indicators.trix(close_prices, period)

        assert len(result) == len(close_prices), "TRIX length mismatch"


        expected_last_five = [-16.03736447, -15.92084231, -15.76171478, -15.53571033, -15.34967155]


        assert len(result) >= 5, "TRIX length too short"
        result_last_five = result[-5:]

        for i, (actual, expected) in enumerate(zip(result_last_five, expected_last_five)):
            assert abs(actual - expected) < 1e-6, f"TRIX mismatch at index {i}: expected {expected}, got {actual}"

    def test_trix_partial_params(self, test_data):
        """Test TRIX with partial parameters"""
        close_prices = test_data['close']


        result_default = ta_indicators.trix(close_prices, 18)
        assert len(result_default) == len(close_prices)


        result_period_14 = ta_indicators.trix(close_prices, 14)
        assert len(result_period_14) == len(close_prices)


        result_custom = ta_indicators.trix(close_prices, 20)
        assert len(result_custom) == len(close_prices)

    def test_trix_errors(self):
        """Test error handling"""

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.trix(np.array([10.0, 20.0, 30.0]), 0)


        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.trix(np.array([10.0, 20.0, 30.0]), 10)


        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.trix(np.array([42.0]), 18)


        with pytest.raises(ValueError, match="Empty"):
            ta_indicators.trix(np.array([]), 18)


        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.trix(np.full(100, np.nan), 18)

    def test_trix_stream(self):
        """Test TrixStream class"""
        stream = ta_indicators.TrixStream(18)


        results = []
        test_values = [100.0, 101.0, 99.5, 102.0, 100.5, 101.5, 99.0, 102.5, 101.0, 100.0]

        for val in test_values:
            result = stream.update(val)
            results.append(result)



        assert results[0] is None

    def test_trix_batch(self, test_data):
        """Test batch computation"""
        close_prices = test_data['close']


        result = ta_indicators.trix_batch(close_prices, (18, 18, 0))
        assert 'values' in result
        assert 'periods' in result
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close_prices)


        result_range = ta_indicators.trix_batch(close_prices, (10, 20, 5))
        assert result_range['values'].shape[0] == 3
        assert result_range['values'].shape[1] == len(close_prices)
        assert np.array_equal(result_range['periods'], [10, 15, 20])

    def test_trix_kernel_selection(self, test_data):
        """Test kernel parameter"""
        close_prices = test_data['close']


        result_auto = ta_indicators.trix(close_prices, 18)
        result_scalar = ta_indicators.trix(close_prices, 18, 'scalar')


        assert_close(result_auto, result_scalar, rtol=1e-10)

    def test_trix_reinput(self, test_data):
        """Test TRIX on its own output"""
        close_prices = test_data['close']
        period = 10


        first_result = ta_indicators.trix(close_prices, period)


        second_result = ta_indicators.trix(first_result, period)

        assert len(first_result) == len(second_result)


        first_valid = np.where(~np.isnan(first_result))[0]
        second_valid = np.where(~np.isnan(second_result))[0]

        if len(first_valid) > 0 and len(second_valid) > 0:
            assert second_valid[0] > first_valid[0]

    def test_trix_nan_handling(self, test_data):
        """Test TRIX handles NaN values correctly"""
        close_prices = test_data['close'].copy()


        close_prices[100:110] = np.nan
        close_prices[200] = np.nan
        close_prices[300:305] = np.nan


        result = ta_indicators.trix(close_prices, 18)
        assert len(result) == len(close_prices)





        valid_before_first_nan = ~np.isnan(result[90:100])
        assert np.any(valid_before_first_nan), "Should have valid values before first NaN region"



        all_nan_after = np.isnan(result[110:])
        assert np.all(all_nan_after), "TRIX should propagate NaN through subsequent calculations"

    def test_trix_empty_input(self):
        """Test TRIX with empty input"""
        with pytest.raises(ValueError, match="Empty"):
            ta_indicators.trix(np.array([]), 18)

    def test_trix_all_nan_input(self):
        """Test TRIX with all NaN input"""
        all_nan = np.full(100, np.nan)
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.trix(all_nan, 18)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
