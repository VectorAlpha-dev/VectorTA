"""
Python binding tests for MOM indicator.
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

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


class TestMom:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_mom_partial_params(self, test_data):
        """Test MOM with default parameters - mirrors check_mom_partial_params"""
        close = test_data['close']


        result = ta_indicators.mom(close, 10)
        assert len(result) == len(close)

    def test_mom_accuracy(self, test_data):
        """Test MOM matches expected values from Rust tests - mirrors check_mom_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['mom']

        result = ta_indicators.mom(
            close,
            period=expected['default_params']['period']
        )

        assert len(result) == len(close)



        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=0,
            atol=1e-1,
            msg="MOM last 5 values mismatch"
        )

    def test_mom_default_params(self, test_data):
        """Test MOM with default parameters - mirrors check_mom_default_candles"""
        close = test_data['close']


        result = ta_indicators.mom(close, 10)
        assert len(result) == len(close)

    def test_mom_zero_period(self):
        """Test MOM fails with zero period - mirrors check_mom_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.mom(input_data, period=0)

    def test_mom_period_exceeds_length(self):
        """Test MOM fails when period exceeds data length - mirrors check_mom_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.mom(data_small, period=10)

    def test_mom_very_small_dataset(self):
        """Test MOM fails with insufficient data - mirrors check_mom_very_small_dataset"""
        single_point = np.array([42.0])

        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.mom(single_point, period=9)

    def test_mom_empty_input(self):
        """Test MOM fails with empty input - mirrors check_mom_empty_input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.mom(empty, period=10)

    def test_mom_nan_handling(self, test_data):
        """Test MOM handles NaN values correctly - mirrors check_mom_nan_handling"""
        close = test_data['close']

        result = ta_indicators.mom(close, period=10)
        assert len(result) == len(close)


        if len(result) > 240:
            assert not np.any(np.isnan(result[240:])), "Found unexpected NaN after warmup period"


        assert np.all(np.isnan(result[:10])), "Expected NaN in warmup period"

    def test_mom_streaming(self, test_data):
        """Test MOM streaming matches batch calculation - mirrors check_mom_streaming"""
        close = test_data['close']
        period = 10


        batch_result = ta_indicators.mom(close, period=period)


        stream = ta_indicators.MomStream(period=period)
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
                        msg=f"MOM streaming mismatch at index {i}")

    def test_mom_batch(self, test_data):
        """Test MOM batch processing with multiple periods"""
        close = test_data['close']


        result = ta_indicators.mom_batch(
            close,
            period_range=(5, 15, 5)
        )

        assert 'values' in result
        assert 'periods' in result


        assert result['values'].shape[0] == 3
        assert result['values'].shape[1] == len(close)
        assert len(result['periods']) == 3
        assert list(result['periods']) == [5, 10, 15]


        single_result = ta_indicators.mom(close, period=10)
        assert_close(
            result['values'][1],
            single_result,
            rtol=1e-10,
            msg="MOM batch period=10 mismatch with single calculation"
        )

    def test_mom_batch_single_period(self, test_data):
        """Test MOM batch with single period matches single calculation"""
        close = test_data['close']


        batch_result = ta_indicators.mom_batch(
            close,
            period_range=(10, 10, 0)
        )


        single_result = ta_indicators.mom(close, period=10)

        assert batch_result['values'].shape[0] == 1
        assert batch_result['values'].shape[1] == len(close)


        assert_close(
            batch_result['values'][0],
            single_result,
            rtol=1e-10,
            msg="MOM batch single period mismatch"
        )

    def test_mom_all_nan_input(self):
        """Test MOM with all NaN values"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.mom(all_nan, period=10)

    def test_mom_batch_edge_cases(self, test_data):
        """Test edge cases for batch processing"""
        close = test_data['close'][:50]


        result = ta_indicators.mom_batch(
            close,
            period_range=(10, 12, 10)
        )


        assert result['values'].shape[0] == 1
        assert list(result['periods']) == [10]


        with pytest.raises(ValueError):
            ta_indicators.mom_batch(
                np.array([]),
                period_range=(10, 10, 0)
            )

    def test_mom_warmup_period(self, test_data):
        """Test that MOM correctly handles warmup period"""
        close = test_data['close'][:100]
        period = 10

        result = ta_indicators.mom(close, period=period)


        assert np.all(np.isnan(result[:period])), f"Expected NaN in first {period} values"


        assert not np.any(np.isnan(result[period:])), f"Unexpected NaN after index {period}"



        for i in range(period, min(period + 5, len(close))):
            expected = close[i] - close[i - period]
            assert_close(
                result[i],
                expected,
                rtol=1e-10,
                msg=f"MOM calculation mismatch at index {i}"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
