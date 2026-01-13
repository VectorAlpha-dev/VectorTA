"""
Python binding tests for StochF indicator.
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


class TestStochF:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_stochf_partial_params(self, test_data):
        """Test StochF with partial parameters (None values) - mirrors check_stochf_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        k, d = ta_indicators.stochf(high, low, close)
        assert len(k) == len(close)
        assert len(d) == len(close)

    def test_stochf_accuracy(self, test_data):
        """Test StochF matches expected values from Rust tests - mirrors check_stochf_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']

        k, d = ta_indicators.stochf(
            high, low, close,
            fastk_period=5,
            fastd_period=3,
            fastd_matype=0
        )

        assert len(k) == len(close)
        assert len(d) == len(close)


        expected_k = [
            80.6987399770905,
            40.88471849865952,
            15.507246376811594,
            36.920529801324506,
            32.1880650994575,
        ]
        expected_d = [
            70.99960994145033,
            61.44725644908976,
            45.696901617520815,
            31.104164892265487,
            28.205280425864817,
        ]



        assert_close(
            k[-5:],
            expected_k,
            rtol=0.0,
            atol=1e-4,
            msg="StochF K last 5 values mismatch"
        )
        assert_close(
            d[-5:],
            expected_d,
            rtol=0.0,
            atol=1e-4,
            msg="StochF D last 5 values mismatch"
        )






    def test_stochf_default_candles(self, test_data):
        """Test StochF with default parameters - mirrors check_stochf_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        k, d = ta_indicators.stochf(high, low, close, 5, 3, 0)
        assert len(k) == len(close)
        assert len(d) == len(close)

    def test_stochf_zero_period(self):
        """Test StochF fails with zero period - mirrors check_stochf_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.stochf(input_data, input_data, input_data, fastk_period=0, fastd_period=3, fastd_matype=0)

    def test_stochf_period_exceeds_length(self):
        """Test StochF fails when period exceeds data length - mirrors check_stochf_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.stochf(data_small, data_small, data_small, fastk_period=10, fastd_period=3, fastd_matype=0)

    def test_stochf_very_small_dataset(self):
        """Test StochF fails with insufficient data - mirrors check_stochf_very_small_dataset"""
        single_point = np.array([42.0])

        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.stochf(single_point, single_point, single_point, fastk_period=9, fastd_period=3, fastd_matype=0)

    def test_stochf_empty_input(self):
        """Test StochF fails with empty input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Input data slice is empty|Empty data"):
            ta_indicators.stochf(empty, empty, empty, fastk_period=5, fastd_period=3, fastd_matype=0)

    def test_stochf_mismatched_lengths(self):
        """Test StochF fails with mismatched input lengths"""
        high = np.array([1.0, 2.0, 3.0])
        low = np.array([1.0, 2.0])
        close = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Input arrays must have the same length"):
            ta_indicators.stochf(high, low, close, fastk_period=2, fastd_period=2, fastd_matype=0)

    def test_stochf_reinput(self, test_data):
        """Test StochF applied twice (re-input) - mirrors check_stochf_slice_reinput"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        k1, d1 = ta_indicators.stochf(high, low, close, 5, 3, 0)
        assert len(k1) == len(close)
        assert len(d1) == len(close)


        k2, d2 = ta_indicators.stochf(k1, k1, k1, 5, 3, 0)
        assert len(k2) == len(k1)
        assert len(d2) == len(d1)

    def test_stochf_nan_handling(self, test_data):
        """Test StochF handles NaN values correctly"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']

        k, d = ta_indicators.stochf(high, low, close, 5, 3, 0)
        assert len(k) == len(close)
        assert len(d) == len(close)




        if len(k) > 10:

            for i in range(10, len(k)):
                assert not np.isnan(k[i]), f"Found unexpected NaN in K at index {i}"


            for i in range(10, len(d)):
                assert not np.isnan(d[i]), f"Found unexpected NaN in D at index {i}"


        for i in range(min(4, len(k))):
            assert np.isnan(k[i]), f"Expected NaN in K warmup period at index {i}"


        for i in range(min(6, len(d))):
            assert np.isnan(d[i]), f"Expected NaN in D warmup period at index {i}"

    def test_stochf_all_nan_input(self):
        """Test StochF with all NaN values"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.stochf(all_nan, all_nan, all_nan, 5, 3, 0)

    def test_stochf_batch_single_parameter_set(self, test_data):
        """Test batch with single parameter combination"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        batch_result = ta_indicators.stochf_batch(
            high, low, close,
            fastk_range=(5, 5, 0),
            fastd_range=(3, 3, 0)
        )


        single_k, single_d = ta_indicators.stochf(high, low, close, 5, 3, 0)

        assert batch_result['k_values'].shape[0] == 1
        assert batch_result['d_values'].shape[0] == 1
        assert batch_result['k_values'].shape[1] == len(close)
        assert batch_result['d_values'].shape[1] == len(close)

        assert_close(
            batch_result['k_values'][0],
            single_k,
            rtol=1e-10,
            msg="Batch vs single K mismatch"
        )
        assert_close(
            batch_result['d_values'][0],
            single_d,
            rtol=1e-10,
            msg="Batch vs single D mismatch"
        )

    def test_stochf_batch_multiple_periods(self, test_data):
        """Test batch with multiple period values"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]


        batch_result = ta_indicators.stochf_batch(
            high, low, close,
            fastk_range=(5, 9, 2),
            fastd_range=(3, 3, 0)
        )


        assert batch_result['k_values'].shape == (3, 100)
        assert batch_result['d_values'].shape == (3, 100)
        assert len(batch_result['fastk_periods']) == 3
        assert len(batch_result['fastd_periods']) == 3


        fastk_periods = [5, 7, 9]
        for i, fastk in enumerate(fastk_periods):
            single_k, single_d = ta_indicators.stochf(high, low, close, fastk, 3, 0)
            assert_close(
                batch_result['k_values'][i],
                single_k,
                rtol=1e-10,
                msg=f"FastK period {fastk} K mismatch"
            )
            assert_close(
                batch_result['d_values'][i],
                single_d,
                rtol=1e-10,
                msg=f"FastK period {fastk} D mismatch"
            )

    def test_stochf_batch_metadata(self, test_data):
        """Test that batch result includes correct parameter combinations"""
        high = test_data['high'][:20]
        low = test_data['low'][:20]
        close = test_data['close'][:20]

        result = ta_indicators.stochf_batch(
            high, low, close,
            fastk_range=(5, 7, 2),
            fastd_range=(3, 4, 1)
        )


        assert len(result['fastk_periods']) == 4
        assert len(result['fastd_periods']) == 4


        expected_combos = [
            (5, 3), (5, 4),
            (7, 3), (7, 4)
        ]

        for i, (k, d) in enumerate(expected_combos):
            assert result['fastk_periods'][i] == k
            assert result['fastd_periods'][i] == d

    def test_stochf_batch_edge_cases(self, test_data):
        """Test edge cases for batch processing"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)


        single_batch = ta_indicators.stochf_batch(
            data, data, data,
            fastk_range=(5, 5, 1),
            fastd_range=(3, 3, 1)
        )

        assert single_batch['k_values'].shape == (1, 10)
        assert single_batch['d_values'].shape == (1, 10)
        assert len(single_batch['fastk_periods']) == 1
        assert len(single_batch['fastd_periods']) == 1


        large_batch = ta_indicators.stochf_batch(
            data, data, data,
            fastk_range=(5, 7, 10),
            fastd_range=(3, 3, 0)
        )


        assert large_batch['k_values'].shape == (1, 10)
        assert large_batch['d_values'].shape == (1, 10)
        assert len(large_batch['fastk_periods']) == 1


        with pytest.raises(ValueError):
            ta_indicators.stochf_batch(
                np.array([]), np.array([]), np.array([]),
                fastk_range=(5, 5, 0),
                fastd_range=(3, 3, 0)
            )


class TestStochFStream:
    def test_stochf_stream_basic(self):
        """Test basic streaming functionality"""
        stream = ta_indicators.StochfStream(5, 3, 0)


        assert stream.update(10.0, 5.0, 7.0) is None
        assert stream.update(12.0, 6.0, 8.0) is None
        assert stream.update(15.0, 7.0, 10.0) is None
        assert stream.update(14.0, 8.0, 12.0) is None


        result = stream.update(16.0, 9.0, 14.0)
        assert result is not None
        k, d = result
        assert isinstance(k, float)
        assert isinstance(d, float)
        assert not np.isnan(k)


    def test_stochf_stream_with_default_params(self):
        """Test stream with default parameters"""
        stream = ta_indicators.StochfStream(5, 3, 0)


        assert stream.update(10.0, 5.0, 7.0) is None

    def test_stochf_stream_invalid_params(self):
        """Test stream creation with invalid parameters"""
        with pytest.raises(ValueError):
            ta_indicators.StochfStream(0, 3, 0)

        with pytest.raises(ValueError):
            ta_indicators.StochfStream(5, 0, 0)
