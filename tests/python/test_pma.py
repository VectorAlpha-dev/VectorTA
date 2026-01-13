"""
Python binding tests for PMA (Predictive Moving Average) indicator.
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

from test_utils import load_test_data, assert_close
from rust_comparison import compare_with_rust


class TestPma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_pma_default_candles(self, test_data):
        """Test PMA with default parameters - mirrors check_pma_default_candles"""
        close = test_data['close']


        predict, trigger = ta_indicators.pma(close)
        assert predict is not None
        assert trigger is not None
        assert len(predict) == len(close)
        assert len(trigger) == len(close)

    def test_pma_accuracy(self, test_data):
        """Test PMA matches expected values from Rust tests - mirrors check_pma_expected_values"""
        hl2 = (test_data['high'] + test_data['low']) / 2

        predict, trigger = ta_indicators.pma(hl2)

        assert len(predict) == len(hl2)
        assert len(trigger) == len(hl2)


        expected_predict = [
            59208.18749999999,
            59233.83609693878,
            59213.19132653061,
            59199.002551020414,
            58993.318877551,
        ]
        expected_trigger = [
            59157.70790816327,
            59208.60076530612,
            59218.6763392857,
            59211.1443877551,
            59123.05019132652,
        ]


        last5_predict = predict[-5:]
        last5_trigger = trigger[-5:]

        for i in range(5):

            assert_close(
                last5_predict[i],
                expected_predict[i],
                rtol=0,
                atol=1e-1,
                msg=f"PMA predict last 5 values mismatch at index {i}"
            )
            assert_close(
                last5_trigger[i],
                expected_trigger[i],
                rtol=0,
                atol=1e-1,
                msg=f"PMA trigger last 5 values mismatch at index {i}"
            )

    def test_pma_with_slice(self, test_data):
        """Test PMA with simple slice data - mirrors check_pma_with_slice"""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=np.float64)

        predict, trigger = ta_indicators.pma(data)
        assert len(predict) == len(data)
        assert len(trigger) == len(data)

    def test_pma_not_enough_data(self):
        """Test PMA fails with not enough data - mirrors check_pma_not_enough_data"""
        data = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        with pytest.raises(Exception, match="Not enough valid data|needed = 7"):
            ta_indicators.pma(data)

    def test_pma_all_values_nan(self):
        """Test PMA fails with all NaN values - mirrors check_pma_all_values_nan"""
        data = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

        with pytest.raises(Exception, match="All values are NaN"):
            ta_indicators.pma(data)

    def test_pma_empty_input(self):
        """Test PMA fails with empty input"""
        empty = np.array([], dtype=np.float64)

        with pytest.raises(Exception, match="Empty data"):
            ta_indicators.pma(empty)

    def test_pma_nan_handling(self, test_data):
        """Test PMA handles NaN values correctly"""
        close = test_data['close']

        predict, trigger = ta_indicators.pma(close)
        assert len(predict) == len(close)
        assert len(trigger) == len(close)



        first_valid = next((i for i, x in enumerate(close) if not np.isnan(x)), 0)
        expected_warmup = first_valid + 6


        for i in range(min(expected_warmup, len(close))):
            assert np.isnan(predict[i]), f"Expected NaN in predict warmup at index {i}"
            assert np.isnan(trigger[i]), f"Expected NaN in trigger warmup at index {i}"


        if len(close) > expected_warmup:

            non_nan_start = max(expected_warmup, 240)
            if len(close) > non_nan_start:
                for i in range(non_nan_start, len(close)):
                    assert not np.isnan(predict[i]), f"Found unexpected NaN in predict at index {i}"
                    assert not np.isnan(trigger[i]), f"Found unexpected NaN in trigger at index {i}"

    def test_pma_mixed_nan_input(self):
        """Test PMA with mixed NaN values"""
        mixed_data = np.array([np.nan, np.nan, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0], dtype=np.float64)

        predict, trigger = ta_indicators.pma(mixed_data)
        assert len(predict) == len(mixed_data)
        assert len(trigger) == len(mixed_data)




        for i in range(len(mixed_data)):
            if i < 8:
                assert np.isnan(predict[i]), f"Expected NaN in predict at index {i} during warmup"
            elif i == 8:
                assert not np.isnan(predict[i]), "Expected valid predict at index 8"

            assert np.isnan(trigger[i]), f"Expected NaN in trigger at index {i} due to insufficient predicts"

    def test_pma_simple_predictable_pattern(self):
        """Test PMA with a simple pattern"""
        simple_data = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5], dtype=np.float64)

        predict, trigger = ta_indicators.pma(simple_data)
        assert len(predict) == len(simple_data)
        assert len(trigger) == len(simple_data)


        for i in range(6):
            assert np.isnan(predict[i]), f"Expected NaN in predict at index {i}"
            assert np.isnan(trigger[i]), f"Expected NaN in trigger at index {i}"


        for i in range(6, min(9, len(simple_data))):
            assert not np.isnan(predict[i]), f"Unexpected NaN in predict at index {i}"
            assert np.isnan(trigger[i]), f"Expected NaN in trigger during warmup at index {i}"


        for i in range(9, len(simple_data)):
            assert not np.isnan(predict[i]), f"Unexpected NaN in predict at index {i}"
            assert not np.isnan(trigger[i]), f"Unexpected NaN in trigger at index {i}"

    def test_pma_batch_single_run(self, test_data):
        """Test PMA batch operation with a single run (no parameter sweep for PMA)"""
        close = test_data['close']


        result = ta_indicators.pma_batch(close)

        assert 'values' in result
        assert 'rows' in result
        assert 'cols' in result


        values = result['values']
        assert result['rows'] == 2
        assert result['cols'] == len(close)
        assert values.shape == (2, len(close))


        single_predict, single_trigger = ta_indicators.pma(close)


        np.testing.assert_array_almost_equal(
            values[0],
            single_predict,
            decimal=10,
            err_msg="Batch predict doesn't match single run"
        )
        np.testing.assert_array_almost_equal(
            values[1],
            single_trigger,
            decimal=10,
            err_msg="Batch trigger doesn't match single run"
        )

    def test_pma_stream(self):
        """Test PMA streaming functionality"""

        stream = ta_indicators.PmaStream()


        test_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0]
        results = []

        for value in test_values:
            result = stream.update(value)
            results.append(result)


        for i in range(6):
            assert results[i] is None, f"Expected None during warmup at index {i}"



        for i in range(6, len(test_values)):
            assert results[i] is not None, f"Expected result after warmup at index {i}"
            assert isinstance(results[i], tuple), "Expected tuple result"
            assert len(results[i]) == 2, "Expected (predict, trigger) tuple"
            predict, trigger = results[i]
            assert isinstance(predict, float), "Expected float predict value"
            assert isinstance(trigger, float), "Expected float trigger value"
            if i < 9:
                assert np.isnan(trigger), f"Expected NaN trigger during warmup at index {i}"
            else:
                assert not np.isnan(trigger), f"Expected valid trigger after warmup at index {i}"

    def test_pma_kernel_parameter(self, test_data):
        """Test PMA with different kernel parameters"""
        close = test_data['close'][:1000]


        kernels = [None, 'auto', 'scalar']
        results = {}

        for kernel in kernels:
            if kernel is None:
                predict, trigger = ta_indicators.pma(close)
            else:
                predict, trigger = ta_indicators.pma(close, kernel=kernel)
            results[kernel or 'default'] = (predict, trigger)


        for kernel_name, (predict, trigger) in results.items():
            assert predict is not None, f"Missing predict for kernel {kernel_name}"
            assert trigger is not None, f"Missing trigger for kernel {kernel_name}"
            assert len(predict) == len(close), f"Wrong predict length for kernel {kernel_name}"
            assert len(trigger) == len(close), f"Wrong trigger length for kernel {kernel_name}"

    def test_rust_parity(self, test_data):
        """Test that Python bindings match Rust implementation"""
        close = test_data['close']
        predict, trigger = ta_indicators.pma(close)


        result = compare_with_rust('pma', predict)
        assert result == True
