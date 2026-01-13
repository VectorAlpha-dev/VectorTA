"""
Python binding tests for CKSP indicator.
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

from test_utils import load_test_data, assert_close, assert_all_nan, assert_no_nan, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust


class TestCksp:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_cksp_partial_params(self, test_data):
        """Test CKSP with partial parameters (None values) - mirrors check_cksp_partial_params"""

        long_result, short_result = ta_indicators.cksp(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            p=10, x=1.0, q=9
        )
        assert len(long_result) == len(test_data['close'])
        assert len(short_result) == len(test_data['close'])

    def test_cksp_accuracy(self, test_data):
        """Test CKSP matches expected values from Rust tests - mirrors check_cksp_accuracy"""
        expected = EXPECTED_OUTPUTS['cksp']
        params = expected['default_params']


        long_result, short_result = ta_indicators.cksp(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            p=params['p'],
            x=params['x'],
            q=params['q']
        )


        assert len(long_result) == len(test_data['close'])
        assert len(short_result) == len(test_data['close'])


        expected_long_last_5 = [
            60306.66197802568,
            60306.66197802568,
            60306.66197802568,
            60203.29578022311,
            60201.57958198072,
        ]
        assert_close(
            long_result[-5:],
            expected_long_last_5,
            rtol=1e-8,
            atol=1e-5,
            msg="CKSP long values mismatch"
        )


        expected_short_last_5 = [
            58757.826484736055,
            58701.74383626245,
            58656.36945263621,
            58611.03250737258,
            58611.03250737258,
        ]
        assert_close(
            short_result[-5:],
            expected_short_last_5,
            rtol=1e-8,
            atol=1e-5,
            msg="CKSP short values mismatch"
        )






    def test_cksp_default_candles(self, test_data):
        """Test CKSP with default parameters - mirrors check_cksp_default_candles"""

        long_result, short_result = ta_indicators.cksp(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            10, 1.0, 9
        )
        assert len(long_result) == len(test_data['close'])
        assert len(short_result) == len(test_data['close'])

    def test_cksp_with_kernel(self, test_data):
        """Test CKSP with different kernel options - mirrors kernel tests"""
        params = EXPECTED_OUTPUTS['cksp']['default_params']


        long_scalar, short_scalar = ta_indicators.cksp(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            p=params['p'],
            x=params['x'],
            q=params['q'],
            kernel='scalar'
        )


        long_auto, short_auto = ta_indicators.cksp(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            p=params['p'],
            x=params['x'],
            q=params['q']
        )


        assert_close(long_scalar, long_auto, rtol=1e-10, msg="CKSP long kernel mismatch")
        assert_close(short_scalar, short_auto, rtol=1e-10, msg="CKSP short kernel mismatch")

    def test_cksp_zero_period(self):
        """Test CKSP fails with zero period - mirrors check_cksp_zero_period"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 10.5])
        close = np.array([9.5, 10.5, 11.0])

        with pytest.raises(ValueError, match="Invalid param"):
            ta_indicators.cksp(high, low, close, p=0, x=1.0, q=9)

    def test_cksp_invalid_x(self):
        """Test CKSP fails with invalid x multiplier"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 10.5])
        close = np.array([9.5, 10.5, 11.0])


        with pytest.raises(ValueError, match="Invalid param"):
            ta_indicators.cksp(high, low, close, p=10, x=float('nan'), q=9)





        with pytest.raises(ValueError, match="Invalid param"):
            ta_indicators.cksp(high, low, close, p=10, x=float('inf'), q=9)

    def test_cksp_invalid_q(self):
        """Test CKSP fails with invalid q parameter"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 10.5])
        close = np.array([9.5, 10.5, 11.0])

        with pytest.raises(ValueError, match="Invalid param"):
            ta_indicators.cksp(high, low, close, p=10, x=1.0, q=0)

    def test_cksp_period_exceeds_length(self):
        """Test CKSP fails when period exceeds data length - mirrors check_cksp_period_exceeds_length"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 10.5])
        close = np.array([9.5, 10.5, 11.0])

        with pytest.raises(ValueError, match="Not enough data"):
            ta_indicators.cksp(high, low, close, p=10, x=1.0, q=9)

    def test_cksp_errors(self):
        """Test error handling - additional edge cases"""

        with pytest.raises(ValueError, match="Data is empty"):
            ta_indicators.cksp(
                np.array([]),
                np.array([]),
                np.array([]),
                p=10, x=1.0, q=9
            )


        with pytest.raises(ValueError, match="Inconsistent"):
            ta_indicators.cksp(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0, 3.0]),
                p=10, x=1.0, q=9
            )


        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 10.5])
        close = np.array([9.5, 10.5, 11.0])
        with pytest.raises(ValueError, match="Invalid param"):
            ta_indicators.cksp(high, low, close, p=0, x=1.0, q=9)


        with pytest.raises(ValueError, match="Not enough data"):
            ta_indicators.cksp(high, low, close, p=10, x=1.0, q=9)

    def test_cksp_very_small_dataset(self):
        """Test CKSP with single data point - mirrors check_cksp_very_small_dataset"""
        high = np.array([42.0])
        low = np.array([41.0])
        close = np.array([41.5])

        with pytest.raises(ValueError, match="Not enough data"):
            ta_indicators.cksp(high, low, close, p=10, x=1.0, q=9)

    def test_cksp_all_nan_input(self):
        """Test CKSP with all NaN values"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="Data is empty|No data"):
            ta_indicators.cksp(all_nan, all_nan, all_nan, p=10, x=1.0, q=9)

    def test_cksp_nan_handling(self, test_data):
        """Test CKSP handles NaN values correctly - mirrors check_cksp_nan_handling"""
        params = EXPECTED_OUTPUTS['cksp']['default_params']

        long_result, short_result = ta_indicators.cksp(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            p=params['p'],
            x=params['x'],
            q=params['q']
        )


        warmup_period = params['p'] + params['q'] - 1
        assert warmup_period == 18, f"Expected warmup period of 18, got {warmup_period}"
        assert_all_nan(long_result[:warmup_period], "Expected NaN in warmup period for long")
        assert_all_nan(short_result[:warmup_period], "Expected NaN in warmup period for short")


        if len(long_result) > 240:
            assert_no_nan(long_result[240:], "Found unexpected NaN in long values after warmup")
            assert_no_nan(short_result[240:], "Found unexpected NaN in short values after warmup")

    def test_cksp_streaming(self, test_data):
        """Test CKSP streaming functionality - mirrors check_cksp_streaming"""
        params = EXPECTED_OUTPUTS['cksp']['default_params']


        stream = ta_indicators.CkspStream(
            p=params['p'],
            x=params['x'],
            q=params['q']
        )


        stream_long = []
        stream_short = []
        for i in range(len(test_data['close'])):
            result = stream.update(
                test_data['high'][i],
                test_data['low'][i],
                test_data['close'][i]
            )
            if result is None:
                stream_long.append(np.nan)
                stream_short.append(np.nan)
            else:
                long_val, short_val = result
                stream_long.append(long_val)
                stream_short.append(short_val)


        batch_long, batch_short = ta_indicators.cksp(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            p=params['p'],
            x=params['x'],
            q=params['q']
        )


        assert_close(
            stream_long,
            batch_long,
            rtol=1e-8,
            msg="CKSP streaming long values mismatch"
        )
        assert_close(
            stream_short,
            batch_short,
            rtol=1e-8,
            msg="CKSP streaming short values mismatch"
        )

    def test_cksp_batch(self, test_data):
        """Test CKSP batch processing - mirrors batch tests"""

        result = ta_indicators.cksp_batch(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            p_range=(10, 10, 0),
            x_range=(1.0, 1.0, 0.0),
            q_range=(9, 9, 0)
        )


        assert 'long_values' in result
        assert 'short_values' in result
        assert 'p' in result
        assert 'x' in result
        assert 'q' in result


        assert result['long_values'].shape == (1, len(test_data['close']))
        assert result['short_values'].shape == (1, len(test_data['close']))


        assert result['p'][0] == 10
        assert result['x'][0] == 1.0
        assert result['q'][0] == 9


        result = ta_indicators.cksp_batch(
            test_data['high'][:100],
            test_data['low'][:100],
            test_data['close'][:100],
            p_range=(5, 15, 5),
            x_range=(0.5, 1.5, 0.5),
            q_range=(5, 10, 5)
        )


        assert result['long_values'].shape == (18, 100)
        assert result['short_values'].shape == (18, 100)
        assert len(result['p']) == 18
        assert len(result['x']) == 18
        assert len(result['q']) == 18



        assert result['p'][0] == 5
        assert result['x'][0] == 0.5
        assert result['q'][0] == 5


        assert result['p'][-1] == 15
        assert result['x'][-1] == 1.5
        assert result['q'][-1] == 10


        for i, (p, q) in enumerate(zip(result['p'], result['q'])):
            expected_warmup = p + q - 1
            row_data = result['long_values'][i]

            nan_count = 0
            for val in row_data:
                if np.isnan(val):
                    nan_count += 1
                else:
                    break

            assert nan_count >= min(expected_warmup, 100), f"Row {i}: Expected at least {expected_warmup} NaN values, got {nan_count}"

    def test_cksp_batch_metadata(self, test_data):
        """Test CKSP batch metadata and edge cases"""
        close = test_data['close'][:50]
        high = test_data['high'][:50]
        low = test_data['low'][:50]


        result = ta_indicators.cksp_batch(
            high, low, close,
            p_range=(10, 10, 1),
            x_range=(1.0, 1.0, 0.1),
            q_range=(9, 9, 1)
        )

        assert result['long_values'].shape == (1, 50)
        assert len(result['p']) == 1


        result = ta_indicators.cksp_batch(
            high, low, close,
            p_range=(10, 12, 10),
            x_range=(1.0, 1.0, 0),
            q_range=(9, 9, 0)
        )


        assert result['long_values'].shape == (1, 50)
        assert result['p'][0] == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
