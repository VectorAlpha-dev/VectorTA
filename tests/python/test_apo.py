"""
Python binding tests for APO indicator.
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


class TestApo:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_apo_partial_params(self, test_data):
        """Test APO with partial parameters - mirrors check_apo_partial_params"""
        close = test_data['close']


        result = ta_indicators.apo(close, 10, 20)
        assert len(result) == len(close)

    def test_apo_accuracy(self, test_data):
        """Test APO matches expected values from Rust tests - mirrors check_apo_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['apo']

        result = ta_indicators.apo(
            close,
            short_period=expected['default_params']['short_period'],
            long_period=expected['default_params']['long_period']
        )

        assert len(result) == len(close)


        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=0,
            atol=1e-1,
            msg="APO last 5 values mismatch"
        )


        compare_with_rust('apo', result)

    def test_apo_default_params(self, test_data):
        """Test APO with default parameters"""
        close = test_data['close']

        result = ta_indicators.apo(close)
        assert len(result) == len(close)

    def test_apo_zero_period(self):
        """Test APO fails with zero period - mirrors check_apo_zero_period"""
        data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError):
            ta_indicators.apo(data, short_period=0, long_period=20)

    def test_apo_period_invalid(self):
        """Test APO fails when short_period >= long_period - mirrors check_apo_period_invalid"""
        data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError):
            ta_indicators.apo(data, short_period=20, long_period=10)

        with pytest.raises(ValueError):
            ta_indicators.apo(data, short_period=10, long_period=10)

    def test_apo_very_small_dataset(self):
        """Test APO fails with insufficient data - mirrors check_apo_very_small_dataset"""
        single_point = np.array([42.0])

        with pytest.raises(ValueError):
            ta_indicators.apo(single_point, short_period=9, long_period=10)

    def test_apo_reinput(self, test_data):
        """Test APO applied twice (re-input) - mirrors check_apo_reinput"""
        close = test_data['close']


        first_result = ta_indicators.apo(close, short_period=10, long_period=20)
        assert len(first_result) == len(close)


        second_result = ta_indicators.apo(first_result, short_period=5, long_period=15)
        assert len(second_result) == len(first_result)

    def test_apo_nan_handling(self, test_data):
        """Test APO handles NaN values correctly - mirrors check_apo_nan_handling"""
        close = test_data['close']

        result = ta_indicators.apo(close, short_period=10, long_period=20)
        assert len(result) == len(close)


        if len(result) > 30:
            assert not np.any(np.isnan(result[30:])), "Found unexpected NaN after index 30"

    def test_apo_all_nan_input(self):
        """Test APO with all NaN values"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError):
            ta_indicators.apo(all_nan)

    def test_apo_kernel_selection(self, test_data):
        """Test APO with different kernel selections"""
        close = test_data['close']


        result_auto = ta_indicators.apo(close, kernel=None)
        result_scalar = ta_indicators.apo(close, kernel='scalar')

        assert len(result_auto) == len(close)
        assert len(result_scalar) == len(close)


        assert_close(result_auto, result_scalar, rtol=1e-10)

    def test_apo_streaming(self, test_data):
        """Test APO streaming functionality"""
        close = test_data['close']


        stream = ta_indicators.ApoStream(short_period=10, long_period=20)


        stream_results = []
        for price in close:
            result = stream.update(price)
            stream_results.append(result)


        stream_results = [np.nan if x is None else x for x in stream_results]


        batch_results = ta_indicators.apo(close, short_period=10, long_period=20)


        assert_close(stream_results, batch_results, rtol=1e-10)

    def test_apo_batch_single_parameter_set(self, test_data):
        """Test batch processing with single parameter combination"""
        close = test_data['close']


        batch_result = ta_indicators.apo_batch(
            close,
            short_period_range=(10, 10, 0),
            long_period_range=(20, 20, 0)
        )


        assert 'values' in batch_result
        assert 'short_periods' in batch_result
        assert 'long_periods' in batch_result


        assert batch_result['values'].shape == (1, len(close))
        assert len(batch_result['short_periods']) == 1
        assert len(batch_result['long_periods']) == 1


        single_result = ta_indicators.apo(close, short_period=10, long_period=20)
        assert_close(batch_result['values'][0], single_result, rtol=1e-10)

    def test_apo_batch_multiple_parameters(self, test_data):
        """Test batch processing with multiple parameter combinations"""
        close = test_data['close'][:100]


        batch_result = ta_indicators.apo_batch(
            close,
            short_period_range=(5, 15, 5),
            long_period_range=(20, 30, 10)
        )


        expected_combos = [
            (5, 20), (5, 30),
            (10, 20), (10, 30),
            (15, 20), (15, 30)
        ]

        assert batch_result['values'].shape[0] == len(expected_combos)
        assert batch_result['values'].shape[1] == len(close)
        assert len(batch_result['short_periods']) == len(expected_combos)
        assert len(batch_result['long_periods']) == len(expected_combos)


        for i, (short, long) in enumerate(expected_combos):
            single_result = ta_indicators.apo(close, short_period=short, long_period=long)
            assert_close(
                batch_result['values'][i],
                single_result,
                rtol=1e-10,
                msg=f"Batch row {i} (short={short}, long={long}) mismatch"
            )

    def test_apo_batch_invalid_combinations(self, test_data):
        """Test batch processing filters out invalid combinations"""
        close = test_data['close'][:50]


        batch_result = ta_indicators.apo_batch(
            close,
            short_period_range=(10, 30, 10),
            long_period_range=(15, 25, 10)
        )




        assert batch_result['values'].shape[0] == 3


        params = list(zip(batch_result['short_periods'], batch_result['long_periods']))
        assert (10, 15) in params
        assert (10, 25) in params
        assert (20, 25) in params

    def test_apo_batch_empty_result(self):
        """Test batch processing with no valid combinations"""
        close = np.random.rand(50)


        with pytest.raises(ValueError):
            ta_indicators.apo_batch(
                close,
                short_period_range=(20, 30, 10),
                long_period_range=(10, 15, 5)
            )

    def test_apo_edge_cases(self):
        """Test APO with edge case inputs"""

        min_data = np.array([1.0] * 20)
        result = ta_indicators.apo(min_data, short_period=10, long_period=20)
        assert len(result) == len(min_data)


        assert result[0] == 0.0


        assert abs(result[-1]) < 1e-10