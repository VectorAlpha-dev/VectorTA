"""
Python binding tests for ASO (Average Sentiment Oscillator) indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
import os
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import my_project
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS

class TestAso:
    @pytest.fixture(scope='class')
    def test_data(self):
        """Load test data from CSV file"""
        return load_test_data()

    def test_aso_accuracy(self, test_data):
        """Test ASO matches expected values from Rust tests - mirrors check_aso_accuracy"""
        expected = EXPECTED_OUTPUTS['aso']


        bulls, bears = my_project.aso(
            test_data['open'],
            test_data['high'],
            test_data['low'],
            test_data['close'],
            period=expected['default_params']['period'],
            mode=expected['default_params']['mode']
        )

        assert len(bulls) == len(test_data['close'])
        assert len(bears) == len(test_data['close'])


        assert_close(
            bulls[-5:],
            expected['last_5_bulls'],
            rtol=0.0,
            atol=1e-6,
            msg="ASO Bulls last 5 values mismatch"
        )

        assert_close(
            bears[-5:],
            expected['last_5_bears'],
            rtol=0.0,
            atol=1e-6,
            msg="ASO Bears last 5 values mismatch"
        )

    def test_aso_partial_params(self, test_data):
        """Test ASO with default parameters - mirrors check_aso_partial_params"""

        bulls, bears = my_project.aso(
            test_data['open'],
            test_data['high'],
            test_data['low'],
            test_data['close']
        )

        assert len(bulls) == len(test_data['close'])
        assert len(bears) == len(test_data['close'])

    def test_aso_zero_period(self):
        """Test ASO fails with zero period - mirrors check_aso_zero_period"""
        open_data = np.array([10.0, 20.0, 30.0])
        high_data = np.array([15.0, 25.0, 35.0])
        low_data = np.array([8.0, 18.0, 28.0])
        close_data = np.array([12.0, 22.0, 32.0])

        with pytest.raises(ValueError, match="Invalid period"):
            my_project.aso(open_data, high_data, low_data, close_data, period=0)

    def test_aso_period_exceeds_length(self):
        """Test ASO fails when period exceeds data length - mirrors check_aso_period_exceeds_length"""
        open_data = np.array([10.0, 20.0, 30.0])
        high_data = np.array([15.0, 25.0, 35.0])
        low_data = np.array([8.0, 18.0, 28.0])
        close_data = np.array([12.0, 22.0, 32.0])

        with pytest.raises(ValueError, match="Invalid period"):
            my_project.aso(open_data, high_data, low_data, close_data, period=10)

    def test_aso_very_small_dataset(self):
        """Test ASO fails with insufficient data - mirrors check_aso_very_small_dataset"""
        single_point = np.array([42.0])

        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            my_project.aso(single_point, single_point, single_point, single_point, period=10)

    def test_aso_empty_input(self):
        """Test ASO fails with empty input - mirrors check_aso_empty_input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Input data slice is empty"):
            my_project.aso(empty, empty, empty, empty)

    def test_aso_all_nan(self):
        """Test ASO fails with all NaN values - mirrors check_aso_all_nan"""
        nan_data = np.full(3, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            my_project.aso(nan_data, nan_data, nan_data, nan_data)

    def test_aso_mismatched_lengths(self):
        """Test ASO fails with mismatched array lengths - mirrors MissingData error"""
        open_data = np.array([10.0, 20.0, 30.0])
        high_data = np.array([15.0, 25.0])
        low_data = np.array([8.0, 18.0, 28.0])
        close_data = np.array([12.0, 22.0, 32.0])

        with pytest.raises(ValueError, match="All OHLC arrays must have the same length"):
            my_project.aso(open_data, high_data, low_data, close_data)

    def test_aso_invalid_mode(self, test_data):
        """Test ASO fails with invalid mode - mirrors check_aso_invalid_mode"""
        with pytest.raises(ValueError, match="Invalid mode"):
            my_project.aso(
                test_data['open'][:100],
                test_data['high'][:100],
                test_data['low'][:100],
                test_data['close'][:100],
                mode=3
            )

    def test_aso_different_modes(self, test_data):
        """Test ASO produces different results for different modes"""
        open_data = test_data['open'][:100]
        high_data = test_data['high'][:100]
        low_data = test_data['low'][:100]
        close_data = test_data['close'][:100]


        bulls0, bears0 = my_project.aso(open_data, high_data, low_data, close_data, mode=0)


        bulls1, bears1 = my_project.aso(open_data, high_data, low_data, close_data, mode=1)


        bulls2, bears2 = my_project.aso(open_data, high_data, low_data, close_data, mode=2)



        check_idx = 50
        assert bulls0[check_idx] != bulls1[check_idx] or bulls0[check_idx] != bulls2[check_idx], \
            "Different modes should produce different results"



        for i in range(20, 100):
            if not np.isnan(bulls1[i]) and not np.isnan(bears1[i]):
                sum1 = bulls1[i] + bears1[i]
                assert abs(sum1 - 100.0) < 1e-9, f"Mode 1: bulls + bears != 100 at index {i}: {sum1}"

            if not np.isnan(bulls2[i]) and not np.isnan(bears2[i]):
                sum2 = bulls2[i] + bears2[i]
                assert abs(sum2 - 100.0) < 1e-9, f"Mode 2: bulls + bears != 100 at index {i}: {sum2}"

    def test_aso_nan_handling(self, test_data):
        """Test ASO handles NaN values correctly - mirrors check_aso_nan_handling"""
        bulls, bears = my_project.aso(
            test_data['open'],
            test_data['high'],
            test_data['low'],
            test_data['close'],
            period=10,
            mode=0
        )

        assert len(bulls) == len(test_data['close'])
        assert len(bears) == len(test_data['close'])


        for i in range(9):
            assert np.isnan(bulls[i]), f"Expected NaN in bulls warmup at index {i}"
            assert np.isnan(bears[i]), f"Expected NaN in bears warmup at index {i}"


        if len(bulls) > 240:
            for i in range(240, len(bulls)):
                assert not np.isnan(bulls[i]), f"Found unexpected NaN in bulls at index {i}"
                assert not np.isnan(bears[i]), f"Found unexpected NaN in bears at index {i}"

    def test_aso_streaming(self, test_data):
        """Test ASO streaming functionality - mirrors check_aso_streaming"""
        period = 10
        mode = 0


        batch_bulls, batch_bears = my_project.aso(
            test_data['open'],
            test_data['high'],
            test_data['low'],
            test_data['close'],
            period=period,
            mode=mode
        )


        stream = my_project.AsoStream(period=period, mode=mode)
        stream_bulls = []
        stream_bears = []

        for i in range(len(test_data['close'])):
            result = stream.update(
                test_data['open'][i],
                test_data['high'][i],
                test_data['low'][i],
                test_data['close'][i]
            )

            if result is None:
                stream_bulls.append(np.nan)
                stream_bears.append(np.nan)
            else:
                bulls, bears = result
                stream_bulls.append(bulls)
                stream_bears.append(bears)

        stream_bulls = np.array(stream_bulls)
        stream_bears = np.array(stream_bears)






        assert len(batch_bulls) == len(stream_bulls)
        assert len(batch_bears) == len(stream_bears)


        for i in range(period - 1):
            assert np.isnan(stream_bulls[i]), f"Expected NaN in streaming bulls at warmup index {i}"
            assert np.isnan(stream_bears[i]), f"Expected NaN in streaming bears at warmup index {i}"


        for i in range(period, min(100, len(stream_bulls))):
            if not np.isnan(stream_bulls[i]):
                assert -1e-9 <= stream_bulls[i] <= 100.0 + 1e-9, \
                    f"Streaming bulls out of range at index {i}: {stream_bulls[i]}"
            if not np.isnan(stream_bears[i]):
                assert -1e-9 <= stream_bears[i] <= 100.0 + 1e-9, \
                    f"Streaming bears out of range at index {i}: {stream_bears[i]}"


            if mode != 0 and not np.isnan(stream_bulls[i]) and not np.isnan(stream_bears[i]):
                total = stream_bulls[i] + stream_bears[i]
                assert abs(total - 100.0) < 1e-9, \
                    f"Mode {mode}: bulls + bears != 100 at index {i}: {total}"

    def test_aso_batch(self, test_data):
        """Test ASO batch processing - mirrors check_aso_batch"""

        open_data = test_data['open'][:100]
        high_data = test_data['high'][:100]
        low_data = test_data['low'][:100]
        close_data = test_data['close'][:100]


        result = my_project.aso_batch(
            open_data,
            high_data,
            low_data,
            close_data,
            period_range=(8, 12, 2),
            mode_range=(0, 2, 1)
        )

        assert 'bulls' in result
        assert 'bears' in result
        assert 'periods' in result
        assert 'modes' in result


        expected_combos = 9
        assert result['bulls'].shape[0] == expected_combos
        assert result['bears'].shape[0] == expected_combos
        assert result['bulls'].shape[1] == 100
        assert result['bears'].shape[1] == 100
        assert len(result['periods']) == expected_combos
        assert len(result['modes']) == expected_combos


        combo_idx = 0
        for period in [8, 10, 12]:
            for mode in [0, 1, 2]:
                assert result['periods'][combo_idx] == period
                assert result['modes'][combo_idx] == mode


                batch_bulls = result['bulls'][combo_idx]
                batch_bears = result['bears'][combo_idx]


                single_bulls, single_bears = my_project.aso(
                    open_data, high_data, low_data, close_data,
                    period=period, mode=mode
                )


                assert_close(
                    batch_bulls, single_bulls, rtol=1e-10,
                    msg=f"Batch bulls mismatch for period={period}, mode={mode}"
                )
                assert_close(
                    batch_bears, single_bears, rtol=1e-10,
                    msg=f"Batch bears mismatch for period={period}, mode={mode}"
                )

                combo_idx += 1

    def test_aso_batch_single_params(self, test_data):
        """Test ASO batch with single parameter set"""
        open_data = test_data['open'][:50]
        high_data = test_data['high'][:50]
        low_data = test_data['low'][:50]
        close_data = test_data['close'][:50]


        result = my_project.aso_batch(
            open_data,
            high_data,
            low_data,
            close_data,
            period_range=(10, 10, 0),
            mode_range=(0, 0, 0)
        )


        assert result['bulls'].shape[0] == 1
        assert result['bears'].shape[0] == 1
        assert len(result['periods']) == 1
        assert len(result['modes']) == 1
        assert result['periods'][0] == 10
        assert result['modes'][0] == 0


        single_bulls, single_bears = my_project.aso(
            open_data, high_data, low_data, close_data,
            period=10, mode=0
        )

        assert_close(
            result['bulls'][0], single_bulls, rtol=1e-10,
            msg="Batch single params bulls mismatch"
        )
        assert_close(
            result['bears'][0], single_bears, rtol=1e-10,
            msg="Batch single params bears mismatch"
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
