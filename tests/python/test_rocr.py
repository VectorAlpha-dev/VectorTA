"""
Python binding tests for ROCR indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

try:
    import my_project as ta_indicators
except ImportError:

    try:
        import my_project as ta_indicators
    except ImportError:
        pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust


class TestRocr:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_rocr_partial_params(self, test_data):
        """Test ROCR with partial parameters (None values) - mirrors check_rocr_partial_params"""
        close = test_data['close']


        result = ta_indicators.rocr(close, 10)
        assert len(result) == len(close)

    def test_rocr_accuracy(self, test_data):
        """Test ROCR matches expected values from Rust tests - mirrors check_rocr_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['rocr']

        result = ta_indicators.rocr(
            close,
            period=expected['default_params']['period']
        )

        assert len(result) == len(close)


        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=1e-8,
            msg="ROCR last 5 values mismatch"
        )




    def test_rocr_default_candles(self, test_data):
        """Test ROCR with default parameters - mirrors check_rocr_default_candles"""
        close = test_data['close']


        result = ta_indicators.rocr(close, 10)
        assert len(result) == len(close)

    def test_rocr_zero_period(self):
        """Test ROCR fails with zero period - mirrors check_rocr_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.rocr(input_data, period=0)

    def test_rocr_period_exceeds_length(self):
        """Test ROCR fails when period exceeds data length - mirrors check_rocr_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.rocr(data_small, period=10)

    def test_rocr_very_small_dataset(self):
        """Test ROCR fails with insufficient data - mirrors check_rocr_very_small_dataset"""
        single_point = np.array([42.0])

        with pytest.raises(ValueError, match="Invalid period|Not enough"):
            ta_indicators.rocr(single_point, period=9)

    def test_rocr_empty_input(self):
        """Test ROCR fails with empty input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.rocr(empty, period=10)

    def test_rocr_all_nan_input(self):
        """Test ROCR with all NaN values"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.rocr(all_nan, period=10)

    def test_rocr_reinput(self, test_data):
        """Test ROCR applied twice (re-input) - mirrors check_rocr_reinput"""
        close = test_data['close']


        first_result = ta_indicators.rocr(close, period=14)
        assert len(first_result) == len(close)


        second_result = ta_indicators.rocr(first_result, period=14)
        assert len(second_result) == len(first_result)


        if len(second_result) > 28:
            assert not np.any(np.isnan(second_result[28:])), "Found unexpected NaN after double warmup period"

    def test_rocr_nan_handling(self, test_data):
        """Test ROCR handles NaN values correctly - mirrors check_rocr_nan_handling"""
        close = test_data['close']

        result = ta_indicators.rocr(close, period=9)
        assert len(result) == len(close)


        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"


        assert np.all(np.isnan(result[:9])), "Expected NaN in warmup period"

    def test_rocr_streaming(self, test_data):
        """Test ROCR streaming matches batch calculation - mirrors check_rocr_streaming"""
        close = test_data['close']
        period = 9


        batch_result = ta_indicators.rocr(close, period=period)


        stream = ta_indicators.RocrStream(period=period)
        stream_values = []

        for price in close:
            result = stream.update(price)

            stream_values.append(result if result is not None else np.nan)

        stream_values = np.array(stream_values)


        assert len(batch_result) == len(stream_values)


        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue
            assert_close(b, s, rtol=1e-9, atol=1e-9,
                        msg=f"ROCR streaming mismatch at index {i}")


    def test_rocr_edge_cases(self, test_data):
        """Test ROCR edge cases"""

        data_with_zeros = np.array([1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0, 10.0])
        result = ta_indicators.rocr(data_with_zeros, period=2)
        assert len(result) == len(data_with_zeros)


        assert result[2] == 0.0
        assert result[6] == 0.0


        result = ta_indicators.rocr(test_data['close'][:20], period=1)
        assert len(result) == 20


        large_data = test_data['close'][:200]
        result = ta_indicators.rocr(large_data, period=100)
        assert len(result) == 200
        assert np.all(np.isnan(result[:100])), "Expected NaN for first 100 values"
        assert not np.any(np.isnan(result[100:])), "Expected valid values after warmup"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])