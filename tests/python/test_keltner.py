"""
Python binding tests for KELTNER indicator.
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


class TestKeltner:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_keltner_accuracy(self, test_data):
        """Test KELTNER matches expected values from Rust tests - mirrors check_keltner_accuracy"""
        expected = EXPECTED_OUTPUTS['keltner']


        upper_band, middle_band, lower_band = ta_indicators.keltner(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            test_data['close'],
            expected['default_params']['period'],
            expected['default_params']['multiplier'],
            expected['default_params']['ma_type']
        )


        assert len(upper_band) == len(test_data['close'])
        assert len(middle_band) == len(test_data['close'])
        assert len(lower_band) == len(test_data['close'])


        assert_close(
            upper_band[-5:],
            expected['last_5_upper'],
            rtol=1e-8,
            msg="Upper band mismatch"
        )

        assert_close(
            middle_band[-5:],
            expected['last_5_middle'],
            rtol=1e-8,
            msg="Middle band mismatch"
        )

        assert_close(
            lower_band[-5:],
            expected['last_5_lower'],
            rtol=1e-8,
            msg="Lower band mismatch"
        )


        try:
            compare_with_rust('keltner', upper_band, 'keltner', expected['default_params'])
        except:
            pass

    def test_keltner_default_params(self, test_data):
        """Test KELTNER with default parameters - mirrors check_keltner_default_params"""

        upper, middle, lower = ta_indicators.keltner(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            test_data['close'],
            20, 2.0, "ema"
        )

        assert len(upper) == len(test_data['close'])
        assert len(middle) == len(test_data['close'])
        assert len(lower) == len(test_data['close'])

    def test_keltner_zero_period(self, test_data):
        """Test KELTNER fails with zero period - mirrors check_keltner_zero_period"""
        with pytest.raises(ValueError, match="invalid period"):
            ta_indicators.keltner(
                test_data['high'],
                test_data['low'],
                test_data['close'],
                test_data['close'],
                0, 2.0, "ema"
            )

    def test_keltner_large_period(self, test_data):
        """Test KELTNER fails when period exceeds data length - mirrors check_keltner_large_period"""
        with pytest.raises(ValueError, match="invalid period"):
            ta_indicators.keltner(
                test_data['high'],
                test_data['low'],
                test_data['close'],
                test_data['close'],
                999999, 2.0, "ema"
            )

    def test_keltner_empty_input(self):
        """Test KELTNER fails with empty input - similar to ALMA's check_alma_empty_input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="empty data"):
            ta_indicators.keltner(empty, empty, empty, empty, 20, 2.0, "ema")

    def test_keltner_very_small_dataset(self):
        """Test KELTNER with insufficient data - similar to ALMA's check_alma_very_small_dataset"""
        small_data = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="invalid period|not enough valid data"):
            ta_indicators.keltner(
                small_data, small_data, small_data, small_data,
                20, 2.0, "ema"
            )

    def test_keltner_all_nan_input(self):
        """Test KELTNER with all NaN values"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="[Aa]ll values are"):
            ta_indicators.keltner(
                all_nan, all_nan, all_nan, all_nan,
                20, 2.0, "ema"
            )

    def test_keltner_invalid_multiplier(self):
        """Test KELTNER with invalid multiplier values"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


        upper, middle, lower = ta_indicators.keltner(
            data, data, data, data,
            2, -2.0, "ema"
        )
        assert len(upper) == len(data)


        upper, middle, lower = ta_indicators.keltner(
            data, data, data, data,
            2, 0.0, "ema"
        )
        assert len(upper) == len(data)



        try:
            upper, middle, lower = ta_indicators.keltner(
                data, data, data, data,
                2, float('nan'), "ema"
            )

            assert np.all(np.isnan(upper[2:])), "Expected NaN output for NaN multiplier"
        except ValueError:

            pass

    def test_keltner_different_ma_types(self, test_data):
        """Test KELTNER with different MA types"""
        ma_types = ["ema", "sma", "wma", "rma"]

        for ma_type in ma_types:
            try:
                upper, middle, lower = ta_indicators.keltner(
                    test_data['high'][:100],
                    test_data['low'][:100],
                    test_data['close'][:100],
                    test_data['close'][:100],
                    20, 2.0, ma_type
                )
                assert len(upper) == 100, f"Failed for MA type: {ma_type}"
                assert len(middle) == 100, f"Failed for MA type: {ma_type}"
                assert len(lower) == 100, f"Failed for MA type: {ma_type}"
            except ValueError as e:

                if "unsupported" not in str(e).lower():
                    raise

    def test_keltner_nan_handling(self, test_data):
        """Test KELTNER handles NaN values correctly - mirrors check_keltner_nan_handling"""
        period = 20
        upper, middle, lower = ta_indicators.keltner(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            test_data['close'],
            period, 2.0, "ema"
        )

        assert len(upper) == len(test_data['close'])
        assert len(middle) == len(test_data['close'])
        assert len(lower) == len(test_data['close'])


        if len(upper) > 240:
            assert not np.any(np.isnan(upper[240:])), "Found unexpected NaN in upper band after warmup"
            assert not np.any(np.isnan(middle[240:])), "Found unexpected NaN in middle band after warmup"
            assert not np.any(np.isnan(lower[240:])), "Found unexpected NaN in lower band after warmup"


        warmup = period - 1
        assert np.all(np.isnan(upper[:warmup])), f"Expected NaN in upper band warmup period (first {warmup} values)"
        assert np.all(np.isnan(middle[:warmup])), f"Expected NaN in middle band warmup period (first {warmup} values)"
        assert np.all(np.isnan(lower[:warmup])), f"Expected NaN in lower band warmup period (first {warmup} values)"

    def test_keltner_streaming(self, test_data):
        """Test streaming functionality - mirrors check_keltner_streaming"""
        expected = EXPECTED_OUTPUTS['keltner']


        batch_upper, batch_middle, batch_lower = ta_indicators.keltner(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            test_data['close'],
            expected['default_params']['period'],
            expected['default_params']['multiplier'],
            expected['default_params']['ma_type']
        )


        stream = ta_indicators.KeltnerStream(
            expected['default_params']['period'],
            expected['default_params']['multiplier'],
            expected['default_params']['ma_type']
        )


        stream_upper = []
        stream_middle = []
        stream_lower = []

        for i in range(len(test_data['close'])):
            result = stream.update(
                test_data['high'][i],
                test_data['low'][i],
                test_data['close'][i],
                test_data['close'][i]
            )
            if result is not None:
                stream_upper.append(result[0])
                stream_middle.append(result[1])
                stream_lower.append(result[2])
            else:
                stream_upper.append(np.nan)
                stream_middle.append(np.nan)
                stream_lower.append(np.nan)

        stream_upper = np.array(stream_upper)
        stream_middle = np.array(stream_middle)
        stream_lower = np.array(stream_lower)


        for i, (b_u, s_u, b_m, s_m, b_l, s_l) in enumerate(zip(
            batch_upper, stream_upper, batch_middle, stream_middle, batch_lower, stream_lower
        )):
            if np.isnan(b_u) and np.isnan(s_u):
                continue
            assert_close(b_u, s_u, rtol=1e-8, atol=1e-8,
                        msg=f"Upper band streaming mismatch at index {i}")
            assert_close(b_m, s_m, rtol=1e-8, atol=1e-8,
                        msg=f"Middle band streaming mismatch at index {i}")
            assert_close(b_l, s_l, rtol=1e-8, atol=1e-8,
                        msg=f"Lower band streaming mismatch at index {i}")

    def test_keltner_batch_default_row(self, test_data):
        """Test batch with default parameters matches single calculation - mirrors check_batch_default_row"""
        expected = EXPECTED_OUTPUTS['keltner']


        single_upper, single_middle, single_lower = ta_indicators.keltner(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            test_data['close'],
            expected['default_params']['period'],
            expected['default_params']['multiplier'],
            expected['default_params']['ma_type']
        )


        result = ta_indicators.keltner_batch(
            test_data['high'],
            test_data['low'],
            test_data['close'],
            test_data['close'],
            (20, 20, 0),
            (2.0, 2.0, 0.0)
        )

        assert 'upper' in result
        assert 'middle' in result
        assert 'lower' in result
        assert 'periods' in result
        assert 'multipliers' in result


        assert result['upper'].shape[0] == 1
        assert result['middle'].shape[0] == 1
        assert result['lower'].shape[0] == 1
        assert result['upper'].shape[1] == len(test_data['close'])


        batch_upper_row = result['upper'][0]
        batch_middle_row = result['middle'][0]
        batch_lower_row = result['lower'][0]


        assert_close(batch_upper_row, single_upper, rtol=1e-8,
                    msg="Batch upper band doesn't match single calculation")
        assert_close(batch_middle_row, single_middle, rtol=1e-8,
                    msg="Batch middle band doesn't match single calculation")
        assert_close(batch_lower_row, single_lower, rtol=1e-8,
                    msg="Batch lower band doesn't match single calculation")

    def test_keltner_batch_multiple_params(self, test_data):
        """Test batch functionality with multiple parameter combinations"""

        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]

        result = ta_indicators.keltner_batch(
            high, low, close, close,
            (10, 30, 10),
            (1.0, 3.0, 1.0)
        )


        expected_rows = 3 * 3
        assert result['upper'].shape == (expected_rows, 100)
        assert result['middle'].shape == (expected_rows, 100)
        assert result['lower'].shape == (expected_rows, 100)


        assert len(result['periods']) == expected_rows
        assert len(result['multipliers']) == expected_rows


        assert result['periods'][0] == 10
        assert result['multipliers'][0] == 1.0
        assert result['periods'][-1] == 30
        assert result['multipliers'][-1] == 3.0



        single_upper, single_middle, single_lower = ta_indicators.keltner(
            high, low, close, close, 10, 1.0, "ema"
        )

        assert_close(result['upper'][0], single_upper, rtol=1e-8,
                    msg="First batch row doesn't match single calculation")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])