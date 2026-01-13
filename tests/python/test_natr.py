"""
Python binding tests for NATR indicator.
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


class TestNatr:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_natr_partial_params(self, test_data):
        """Test NATR with different period values"""

        result_14 = ta_indicators.natr(test_data['high'], test_data['low'], test_data['close'], 14)
        assert len(result_14) == len(test_data['close'])


        result_7 = ta_indicators.natr(test_data['high'], test_data['low'], test_data['close'], 7)
        assert len(result_7) == len(test_data['close'])


        assert not np.allclose(result_14[14:], result_7[14:], rtol=1e-9),\
                    "NATR with different periods should produce different results"

    def test_natr_accuracy(self, test_data):
        """Test NATR matches expected values from Rust tests"""
        period = EXPECTED_OUTPUTS['natr']['default_params']['period']


        result = ta_indicators.natr(test_data['high'], test_data['low'], test_data['close'], period)


        assert len(result) == len(test_data['close'])


        result_last_5 = result[-5:]
        expected_last_5 = EXPECTED_OUTPUTS['natr']['last_5_values']


        assert_close(result_last_5, expected_last_5, rtol=0, atol=1e-8,
                     msg="NATR last 5 values don't match expected")

    def test_natr_zero_period(self):
        """Test NATR with zero period"""
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.natr(np.array([10.0, 20.0]), np.array([5.0, 10.0]), np.array([7.0, 15.0]), 0)

    def test_natr_period_exceeds_length(self):
        """Test NATR when period exceeds data length"""
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.natr(np.array([10.0, 20.0]), np.array([5.0, 10.0]), np.array([7.0, 15.0]), 10)

    def test_natr_very_small_dataset(self):
        """Test NATR with very small dataset"""
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.natr(np.array([42.0]), np.array([40.0]), np.array([41.0]), 14)

    def test_natr_empty_input(self):
        """Test NATR with empty input"""
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.natr(np.array([]), np.array([]), np.array([]), 14)

    def test_natr_mismatched_lengths(self):
        """Test NATR with mismatched input lengths"""
        with pytest.raises(ValueError, match="Mismatched lengths"):
            ta_indicators.natr(
                np.array([10.0, 20.0, 30.0]),
                np.array([5.0, 10.0]),
                np.array([7.0, 15.0, 25.0]),
                2
            )

    def test_natr_all_nan_input(self):
        """Test NATR with all NaN values"""
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.natr(
                np.array([np.nan, np.nan, np.nan]),
                np.array([np.nan, np.nan, np.nan]),
                np.array([np.nan, np.nan, np.nan]),
                2
            )

    def test_natr_reinput(self, test_data):
        """Test NATR reinput (using output as input)"""

        result1 = ta_indicators.natr(test_data['high'], test_data['low'], test_data['close'], 14)


        result2 = ta_indicators.natr(result1, result1, result1, 14)

        assert len(result2) == len(result1)

        valid_count = np.sum(~np.isnan(result2[28:]))
        assert valid_count > 0, "Should have valid values after double warmup"

    def test_natr_streaming(self, test_data):
        """Test NATR streaming functionality"""
        period = 14


        batch_result = ta_indicators.natr(test_data['high'], test_data['low'], test_data['close'], period)


        stream = ta_indicators.NatrStream(period)
        stream_results = []

        for i in range(len(test_data['high'])):
            result = stream.update(test_data['high'][i], test_data['low'][i], test_data['close'][i])
            if result is not None:
                stream_results.append(result)
            else:
                stream_results.append(np.nan)


        stream_results = np.array(stream_results)


        assert_close(batch_result, stream_results, rtol=0, atol=1e-9,
                     msg="NATR streaming results don't match batch results")

    def test_natr_batch(self, test_data):
        """Test NATR batch functionality with multiple periods"""

        batch_result = ta_indicators.natr_batch(
            test_data['high'], test_data['low'], test_data['close'],
            (10, 20, 5)
        )

        assert batch_result['values'].shape == (3, len(test_data['close']))
        assert len(batch_result['periods']) == 3
        assert list(batch_result['periods']) == [10, 15, 20]


        for i, period in enumerate([10, 15, 20]):
            single = ta_indicators.natr(test_data['high'], test_data['low'], test_data['close'], period)

            assert_close(batch_result['values'][i], single, rtol=0, atol=1e-9,
                         msg=f"NATR batch row {i} (period {period}) doesn't match single calculation")

    def test_natr_batch_single_parameter(self, test_data):
        """Test NATR batch with single parameter set"""

        batch_result = ta_indicators.natr_batch(
            test_data['high'], test_data['low'], test_data['close'],
            (14, 14, 0)
        )


        assert 'values' in batch_result
        assert 'periods' in batch_result
        assert batch_result['values'].shape == (1, len(test_data['close']))
        assert len(batch_result['periods']) == 1
        assert batch_result['periods'][0] == 14


        single_result = ta_indicators.natr(test_data['high'], test_data['low'], test_data['close'], 14)
        assert_close(batch_result['values'][0], single_result, rtol=0, atol=1e-9,
                     msg="NATR batch single doesn't match single calculation")

    def test_natr_batch_edge_cases(self, test_data):
        """Test NATR batch edge cases"""

        batch_result = ta_indicators.natr_batch(
            test_data['high'][:100], test_data['low'][:100], test_data['close'][:100],
            (10, 15, 1)
        )

        assert batch_result['values'].shape == (6, 100)
        assert len(batch_result['periods']) == 6
        assert list(batch_result['periods']) == [10, 11, 12, 13, 14, 15]


        batch_result_large = ta_indicators.natr_batch(
            test_data['high'][:100], test_data['low'][:100], test_data['close'][:100],
            (5, 50, 15)
        )

        assert batch_result_large['values'].shape == (4, 100)
        assert list(batch_result_large['periods']) == [5, 20, 35, 50]

    def test_natr_with_nans(self, test_data):
        """Test NATR handles NaN values correctly"""

        high_with_nans = test_data['high'].copy()
        low_with_nans = test_data['low'].copy()
        close_with_nans = test_data['close'].copy()


        high_with_nans[10:15] = np.nan
        low_with_nans[10:15] = np.nan
        close_with_nans[10:15] = np.nan


        result = ta_indicators.natr(high_with_nans, low_with_nans, close_with_nans, 14)
        assert len(result) == len(test_data['close'])


        assert np.all(np.isnan(result[:13]))




        assert np.isnan(result[13])




    def test_natr_handles_zero_close(self):
        """Test NATR handles zero close price"""

        high = np.array([100.0, 110.0, 105.0, 108.0])
        low = np.array([95.0, 100.0, 98.0, 102.0])
        close = np.array([98.0, 105.0, 0.0, 106.0])

        result = ta_indicators.natr(high, low, close, 2)


        assert np.isnan(result[2]), "Result should be NaN when close price is zero"

    def test_natr_handles_infinite_values(self):
        """Test NATR handles infinite values"""

        high = np.array([100.0, 110.0, np.inf, 108.0, 105.0])
        low = np.array([95.0, 100.0, 98.0, 102.0, 99.0])
        close = np.array([98.0, 105.0, 100.0, 106.0, 102.0])

        result = ta_indicators.natr(high, low, close, 2)


        assert np.any(np.isfinite(result)), "Should have some valid finite values"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
