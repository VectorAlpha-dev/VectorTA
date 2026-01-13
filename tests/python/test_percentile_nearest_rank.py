"""
Python binding tests for Percentile Nearest Rank indicator.
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


class TestPercentileNearestRank:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_percentile_nearest_rank_partial_params(self, test_data):
        """Test PNR with partial parameters - mirrors check_pnr_partial_params"""
        data = np.array(EXPECTED_OUTPUTS['percentile_nearest_rank']['basic_test']['data'])


        result = ta_indicators.percentile_nearest_rank(data, length=5)
        assert len(result) == len(data)
        assert result[4] == EXPECTED_OUTPUTS['percentile_nearest_rank']['basic_test']['expected_at_4']

    def test_percentile_nearest_rank_accuracy(self, test_data):
        """Test PNR accuracy with default parameters - mirrors check_pnr_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['percentile_nearest_rank']

        result = ta_indicators.percentile_nearest_rank(
            close,
            length=expected['default_params']['length'],
            percentage=expected['default_params']['percentage']
        )

        assert len(result) == len(close)


        warmup = expected['warmup_period']
        assert np.all(np.isnan(result[:warmup])), f"Expected NaN in first {warmup} values"


        assert not np.isnan(result[warmup]), f"Expected valid value at index {warmup}"



        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=0,
            atol=1e-6,
            msg="PNR last 5 values mismatch"
        )





    def test_percentile_nearest_rank_default_candles(self, test_data):
        """Test PNR with default parameters - mirrors check_pnr_default_candles"""
        close = test_data['close']


        result = ta_indicators.percentile_nearest_rank(close)
        assert len(result) == len(close)


        assert np.all(np.isnan(result[:14]))
        assert not np.isnan(result[14])

    def test_percentile_nearest_rank_zero_period(self):
        """Test PNR fails with zero period - mirrors check_pnr_zero_period"""
        data = np.array([1.0] * 10)

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.percentile_nearest_rank(data, length=0, percentage=50.0)

    def test_percentile_nearest_rank_period_exceeds_length(self):
        """Test PNR fails when period exceeds data length - mirrors check_pnr_period_exceeds_length"""
        data = np.array([1.0] * 5)

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.percentile_nearest_rank(data, length=10, percentage=50.0)

    def test_percentile_nearest_rank_very_small_dataset(self):
        """Test PNR with single data point - mirrors check_pnr_very_small_dataset"""
        data = np.array([5.0])

        result = ta_indicators.percentile_nearest_rank(data, length=1, percentage=50.0)
        assert len(result) == 1
        assert result[0] == 5.0

    def test_percentile_nearest_rank_empty_input(self):
        """Test PNR fails with empty input - mirrors check_pnr_empty_input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Input data is empty"):
            ta_indicators.percentile_nearest_rank(empty)

    def test_percentile_nearest_rank_invalid_percentage(self):
        """Test PNR fails with invalid percentage - mirrors check_pnr_invalid_percentage"""
        data = np.array([1.0] * 20)


        with pytest.raises(ValueError, match="Percentage must be between"):
            ta_indicators.percentile_nearest_rank(data, length=5, percentage=150.0)


        with pytest.raises(ValueError, match="Percentage must be between"):
            ta_indicators.percentile_nearest_rank(data, length=5, percentage=-10.0)

    def test_percentile_nearest_rank_nan_handling(self, test_data):
        """Test PNR handles NaN values correctly - mirrors check_pnr_nan_handling"""
        data = np.array([
            1.0, 2.0, np.nan, 4.0, 5.0,
            np.nan, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, np.nan, 15.0,
        ])

        result = ta_indicators.percentile_nearest_rank(data, length=5, percentage=50.0)
        assert len(result) == len(data)


        assert not np.isnan(result[6])

    def test_percentile_nearest_rank_basic(self):
        """Test basic functionality with simple data"""
        data = np.array(EXPECTED_OUTPUTS['percentile_nearest_rank']['basic_test']['data'])
        expected_test = EXPECTED_OUTPUTS['percentile_nearest_rank']['basic_test']

        result = ta_indicators.percentile_nearest_rank(
            data,
            length=expected_test['length'],
            percentage=expected_test['percentage']
        )

        assert len(result) == len(data)


        assert np.all(np.isnan(result[:4]))


        assert result[4] == expected_test['expected_at_4']
        assert result[5] == expected_test['expected_at_5']

    def test_percentile_nearest_rank_different_percentiles(self):
        """Test with different percentile values"""
        data = np.array(EXPECTED_OUTPUTS['percentile_nearest_rank']['percentile_tests']['data'])
        expected_test = EXPECTED_OUTPUTS['percentile_nearest_rank']['percentile_tests']
        length = expected_test['length']


        result_25 = ta_indicators.percentile_nearest_rank(data, length=length, percentage=25.0)
        assert result_25[4] == expected_test['p25_at_4']


        result_75 = ta_indicators.percentile_nearest_rank(data, length=length, percentage=75.0)
        assert result_75[4] == expected_test['p75_at_4']


        result_100 = ta_indicators.percentile_nearest_rank(data, length=length, percentage=100.0)
        assert result_100[4] == expected_test['p100_at_4']

    def test_percentile_nearest_rank_streaming(self, test_data):
        """Test PNR streaming matches batch calculation - mirrors check_pnr_streaming"""
        close = test_data['close'][:100]
        length = 15
        percentage = 50.0


        batch_result = ta_indicators.percentile_nearest_rank(
            close, length=length, percentage=percentage
        )


        stream = ta_indicators.PercentileNearestRankStream(length=length, percentage=percentage)
        stream_values = []

        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)

        stream_values = np.array(stream_values)


        assert len(batch_result) == len(stream_values)


        for i, (b, s) in enumerate(zip(batch_result, stream_values)):
            if np.isnan(b) and np.isnan(s):
                continue

            assert_close(b, s, rtol=0, atol=1e-9,
                         msg=f"PNR streaming mismatch at index {i}")

    def test_percentile_nearest_rank_batch(self, test_data):
        """Test PNR batch processing - mirrors check_batch_default_row"""
        close = test_data['close'][:100]


        result = ta_indicators.percentile_nearest_rank_batch(
            close,
            length_range=(15, 15, 0),
            percentage_range=(50.0, 50.0, 0.0)
        )

        assert 'values' in result
        assert 'lengths' in result
        assert 'percentages' in result


        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)


        batch_row = result['values'][0]
        single_result = ta_indicators.percentile_nearest_rank(close, length=15, percentage=50.0)


        assert_close(
            batch_row,
            single_result,
            rtol=0,
            atol=1e-9,
            msg="PNR batch default row mismatch"
        )

    def test_percentile_nearest_rank_batch_sweep(self, test_data):
        """Test PNR batch with parameter sweep"""
        close = test_data['close'][:50]


        result = ta_indicators.percentile_nearest_rank_batch(
            close,
            length_range=(10, 20, 10),
            percentage_range=(25.0, 75.0, 25.0)
        )


        assert result['values'].shape[0] == 6
        assert result['values'].shape[1] == len(close)
        assert len(result['lengths']) == 6
        assert len(result['percentages']) == 6


        assert result['lengths'][0] == 10
        assert result['percentages'][0] == 25.0


        assert result['lengths'][5] == 20
        assert result['percentages'][5] == 75.0


        for i in range(6):
            length = result['lengths'][i]
            row = result['values'][i]


            warmup_values = row[:length-1]

            assert np.all(np.isnan(warmup_values) | (np.abs(warmup_values) < 1e-100))

            if length < len(close):
                assert not np.isnan(row[length-1]) and np.abs(row[length-1]) > 1e-10

    def test_percentile_nearest_rank_batch_metadata(self, test_data):
        """Test batch result includes correct metadata"""
        close = test_data['close'][:30]

        result = ta_indicators.percentile_nearest_rank_batch(
            close,
            length_range=(5, 10, 5),
            percentage_range=(50.0, 100.0, 50.0)
        )


        assert len(result['lengths']) == 4
        assert len(result['percentages']) == 4


        expected_combos = [
            (5, 50.0), (5, 100.0),
            (10, 50.0), (10, 100.0)
        ]

        for i, (length, percentage) in enumerate(expected_combos):
            assert result['lengths'][i] == length
            assert_close(result['percentages'][i], percentage, rtol=1e-10)

    def test_percentile_nearest_rank_all_nan_input(self):
        """Test PNR with all NaN values"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.percentile_nearest_rank(all_nan)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
