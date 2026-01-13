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


        line, signal = ta_indicators.kst(
            close,
            sma_period1=10, sma_period2=10, sma_period3=10, sma_period4=15,
            roc_period1=10, roc_period2=15, roc_period3=20, roc_period4=30,
            signal_period=9
        )
        assert len(line) == len(close)
        assert len(signal) == len(close)

    def test_kst_accuracy(self, test_data):
        """Test KST matches expected values from Rust tests - mirrors check_kst_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['kst']


        line, signal = ta_indicators.kst(
            close,
            sma_period1=10, sma_period2=10, sma_period3=10, sma_period4=15,
            roc_period1=10, roc_period2=15, roc_period3=20, roc_period4=30,
            signal_period=9
        )

        assert len(line) == len(close)
        assert len(signal) == len(close)


        assert_close(
            line[-5:],
            expected['last_5_line'],
            rtol=0,
            atol=1e-1,
            msg="KST line last 5 values mismatch"
        )

        assert_close(
            signal[-5:],
            expected['last_5_signal'],
            rtol=0,
            atol=1e-1,
            msg="KST signal last 5 values mismatch"
        )


        compare_with_rust('kst_line', line)
        compare_with_rust('kst_signal', signal)

    def test_kst_partial_params(self, test_data):
        """Test KST with partial parameters"""
        close = test_data['close']


        line, signal = ta_indicators.kst(
            close,
            sma_period1=12, sma_period2=10, sma_period3=10, sma_period4=15,
            roc_period1=12, roc_period2=15, roc_period3=20, roc_period4=30,
            signal_period=10
        )
        assert len(line) == len(close)
        assert len(signal) == len(close)

    def test_kst_zero_period(self):
        """Test KST fails with zero period"""
        data = np.array([10.0, 20.0, 30.0] * 20)

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
            ta_indicators.kst(small_data, roc_period4=50)

    def test_kst_nan_handling(self, test_data):
        """Test KST handles NaN values correctly - mirrors check_kst_nan_handling"""
        all_nan = np.full(10, np.nan)

        with pytest.raises(ValueError):
            ta_indicators.kst(all_nan)

    def test_kst_all_params(self, test_data):
        """Test KST with all parameters specified"""
        close = test_data['close']

        line, signal = ta_indicators.kst(
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

        assert len(line) == len(close)
        assert len(signal) == len(close)

    def test_kst_kernel_selection(self, test_data):
        """Test KST with different kernel selections"""
        close = test_data['close']


        line_auto, signal_auto = ta_indicators.kst(
            close,
            sma_period1=10, sma_period2=10, sma_period3=10, sma_period4=15,
            roc_period1=10, roc_period2=15, roc_period3=20, roc_period4=30,
            signal_period=9,
            kernel=None
        )
        line_scalar, signal_scalar = ta_indicators.kst(
            close,
            sma_period1=10, sma_period2=10, sma_period3=10, sma_period4=15,
            roc_period1=10, roc_period2=15, roc_period3=20, roc_period4=30,
            signal_period=9,
            kernel='scalar'
        )

        assert len(line_auto) == len(close)
        assert len(line_scalar) == len(close)


        assert_close(line_auto, line_scalar, rtol=1e-10)
        assert_close(signal_auto, signal_scalar, rtol=1e-10)

    def test_kst_streaming(self, test_data):
        """Test KST streaming functionality"""
        close = np.array(test_data['close'])






        first_valid = next((i for i, v in enumerate(close) if not np.isnan(v)), 0)
        valid_close = close[first_valid:]


        stream = ta_indicators.KstStream(10, 10, 10, 15, 10, 15, 20, 30, 9)


        stream_results = []
        for price in valid_close:
            if not np.isnan(price):
                result = stream.update(price)
                stream_results.append(result)


        line_results = []
        signal_results = []
        for r in stream_results:
            if r is None:
                line_results.append(np.nan)
                signal_results.append(np.nan)
            else:
                line_results.append(r[0])
                signal_results.append(r[1])


        batch_line, batch_signal = ta_indicators.kst(
            valid_close,
            sma_period1=10, sma_period2=10, sma_period3=10, sma_period4=15,
            roc_period1=10, roc_period2=15, roc_period3=20, roc_period4=30,
            signal_period=9
        )


        assert_close(line_results, batch_line, rtol=1e-10)











        valid_lines = sum(1 for v in line_results if not np.isnan(v))
        assert valid_lines > 0, "Stream should produce some valid line values"

    def test_kst_batch_single_parameter_set(self, test_data):
        """Test batch processing with single parameter combination"""
        close = test_data['close']


        batch_result = ta_indicators.kst_batch(
            close,
            sma1_range=(10, 10, 0),
            sma2_range=(10, 10, 0),
            sma3_range=(10, 10, 0),
            sma4_range=(15, 15, 0),
            roc1_range=(10, 10, 0),
            roc2_range=(15, 15, 0),
            roc3_range=(20, 20, 0),
            roc4_range=(30, 30, 0),
            sig_range=(9, 9, 0)
        )


        assert 'line' in batch_result
        assert 'signal' in batch_result
        assert 'sma1' in batch_result
        assert 'sma2' in batch_result
        assert 'sma3' in batch_result
        assert 'sma4' in batch_result
        assert 'roc1' in batch_result
        assert 'roc2' in batch_result
        assert 'roc3' in batch_result
        assert 'roc4' in batch_result
        assert 'sig' in batch_result


        assert batch_result['line'].shape == (1, len(close))
        assert batch_result['signal'].shape == (1, len(close))


        single_line, single_signal = ta_indicators.kst(
            close,
            sma_period1=10, sma_period2=10, sma_period3=10, sma_period4=15,
            roc_period1=10, roc_period2=15, roc_period3=20, roc_period4=30,
            signal_period=9
        )
        assert_close(batch_result['line'][0], single_line, rtol=1e-10)
        assert_close(batch_result['signal'][0], single_signal, rtol=1e-10)

    def test_kst_batch_multiple_parameters(self, test_data):
        """Test batch processing with multiple parameter combinations"""
        close = test_data['close'][:200]


        batch_result = ta_indicators.kst_batch(
            close,
            sma1_range=(10, 10, 0),
            sma2_range=(10, 10, 0),
            sma3_range=(10, 10, 0),
            sma4_range=(15, 15, 0),
            roc1_range=(10, 12, 2),
            roc2_range=(15, 15, 0),
            roc3_range=(20, 20, 0),
            roc4_range=(30, 30, 0),
            sig_range=(8, 10, 2)
        )


        assert batch_result['line'].shape[0] == 4
        assert batch_result['line'].shape[1] == len(close)
        assert batch_result['signal'].shape[0] == 4
        assert batch_result['signal'].shape[1] == len(close)


        assert len(batch_result['roc1']) == 4
        assert len(batch_result['sig']) == 4

    def test_kst_batch_custom_ranges(self, test_data):
        """Test batch processing with custom parameter ranges"""
        close = test_data['close'][:100]

        batch_result = ta_indicators.kst_batch(
            close,
            sma1_range=(8, 12, 2),
            sma2_range=(10, 10, 0),
            sma3_range=(10, 10, 0),
            sma4_range=(15, 15, 0),
            roc1_range=(10, 10, 0),
            roc2_range=(15, 15, 0),
            roc3_range=(20, 20, 0),
            roc4_range=(25, 30, 5),
            sig_range=(9, 9, 0)
        )


        assert batch_result['line'].shape[0] == 6
        assert batch_result['signal'].shape[0] == 6

    def test_kst_edge_cases(self):
        """Test KST with edge case inputs"""



        min_data = np.random.rand(53)
        line, signal = ta_indicators.kst(
            min_data,
            sma_period1=10, sma_period2=10, sma_period3=10, sma_period4=15,
            roc_period1=10, roc_period2=15, roc_period3=20, roc_period4=30,
            signal_period=9
        )
        assert len(line) == len(min_data)
        assert len(signal) == len(min_data)



        assert np.isnan(line[0])
        assert np.isnan(signal[0])


        assert not np.isnan(line[-1])
        assert not np.isnan(signal[-1])

    def test_kst_batch_kernel_selection(self, test_data):
        """Test KST batch with different kernels"""
        close = test_data['close'][:100]


        result_auto = ta_indicators.kst_batch(
            close,
            sma1_range=(10, 10, 0),
            sma2_range=(10, 10, 0),
            sma3_range=(10, 10, 0),
            sma4_range=(15, 15, 0),
            roc1_range=(10, 10, 0),
            roc2_range=(15, 15, 0),
            roc3_range=(20, 20, 0),
            roc4_range=(30, 30, 0),
            sig_range=(9, 9, 0),
            kernel=None
        )
        result_scalar = ta_indicators.kst_batch(
            close,
            sma1_range=(10, 10, 0),
            sma2_range=(10, 10, 0),
            sma3_range=(10, 10, 0),
            sma4_range=(15, 15, 0),
            roc1_range=(10, 10, 0),
            roc2_range=(15, 15, 0),
            roc3_range=(20, 20, 0),
            roc4_range=(30, 30, 0),
            sig_range=(9, 9, 0),
            kernel='scalar'
        )


        assert_close(result_auto['line'], result_scalar['line'], rtol=1e-10)
        assert_close(result_auto['signal'], result_scalar['signal'], rtol=1e-10)

    def test_kst_reinput(self, test_data):
        """Test KST applied twice (re-input) - similar to ALMA reinput test"""
        close = test_data['close'][:500]


        first_line, first_signal = ta_indicators.kst(
            close,
            sma_period1=10, sma_period2=10, sma_period3=10, sma_period4=15,
            roc_period1=10, roc_period2=15, roc_period3=20, roc_period4=30,
            signal_period=9
        )
        assert len(first_line) == len(close)


        second_line, second_signal = ta_indicators.kst(
            first_line,
            sma_period1=10, sma_period2=10, sma_period3=10, sma_period4=15,
            roc_period1=10, roc_period2=15, roc_period3=20, roc_period4=30,
            signal_period=9
        )
        assert len(second_line) == len(first_line)




        warmup_first = 44
        warmup_second = warmup_first + 44


        assert np.isnan(first_line[0])
        assert np.isnan(second_line[warmup_second - 1])


        if len(second_line) > warmup_second + 10:
            assert not np.isnan(second_line[warmup_second + 10])

    def test_kst_nan_prefix(self, test_data):
        """Test KST with NaN values at start of data"""
        close = test_data['close'][:200]


        close_with_nan = np.array(close)
        close_with_nan[:20] = np.nan

        line, signal = ta_indicators.kst(
            close_with_nan,
            sma_period1=10, sma_period2=10, sma_period3=10, sma_period4=15,
            roc_period1=10, roc_period2=15, roc_period3=20, roc_period4=30,
            signal_period=9
        )

        assert len(line) == len(close_with_nan)
        assert len(signal) == len(close_with_nan)



        expected_warmup = 44 + 20


        assert np.isnan(line[expected_warmup - 1])


        if len(line) > expected_warmup + 10:
            assert not np.isnan(line[expected_warmup + 10])

    def test_kst_warmup_period(self, test_data):
        """Test KST warmup period calculation"""
        close = test_data['close'][:100]


        line, signal = ta_indicators.kst(
            close,
            sma_period1=10, sma_period2=10, sma_period3=10, sma_period4=15,
            roc_period1=10, roc_period2=15, roc_period3=20, roc_period4=30,
            signal_period=9
        )



        expected_warmup = 44


        for i in range(expected_warmup):
            assert np.isnan(line[i]), f"Expected NaN at index {i} during warmup"


        for i in range(expected_warmup, min(expected_warmup + 5, len(line))):
            assert not np.isnan(line[i]), f"Expected valid value at index {i} after warmup"


        signal_warmup = expected_warmup + 9 - 1
        for i in range(signal_warmup):
            assert np.isnan(signal[i]), f"Expected NaN in signal at index {i} during warmup"

    def test_kst_invalid_parameters(self):
        """Test KST with invalid parameters"""
        data = np.random.rand(100)


        with pytest.raises(ValueError):
            ta_indicators.kst(
                data,
                sma_period1=0, sma_period2=10, sma_period3=10, sma_period4=15,
                roc_period1=10, roc_period2=15, roc_period3=20, roc_period4=30,
                signal_period=9
            )

        with pytest.raises(ValueError):
            ta_indicators.kst(
                data,
                sma_period1=10, sma_period2=10, sma_period3=10, sma_period4=15,
                roc_period1=0, roc_period2=15, roc_period3=20, roc_period4=30,
                signal_period=9
            )

        with pytest.raises(ValueError):
            ta_indicators.kst(
                data,
                sma_period1=10, sma_period2=10, sma_period3=10, sma_period4=15,
                roc_period1=10, roc_period2=15, roc_period3=20, roc_period4=30,
                signal_period=0
            )

    def test_kst_empty_input(self):
        """Test KST with empty input data"""
        empty = np.array([])

        with pytest.raises(ValueError, match="empty|Empty"):
            ta_indicators.kst(
                empty,
                sma_period1=10, sma_period2=10, sma_period3=10, sma_period4=15,
                roc_period1=10, roc_period2=15, roc_period3=20, roc_period4=30,
                signal_period=9
            )

    def test_kst_insufficient_data(self):
        """Test KST with insufficient data for calculation"""

        small_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError):
            ta_indicators.kst(
                small_data,
                sma_period1=10, sma_period2=10, sma_period3=10, sma_period4=15,
                roc_period1=10, roc_period2=15, roc_period3=20, roc_period4=30,
                signal_period=9
            )