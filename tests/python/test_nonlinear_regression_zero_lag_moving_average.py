import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    try:
        import my_project as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


class TestNonlinearRegressionZeroLagMovingAverage:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = np.asarray(test_data["close"][:256], dtype=np.float64)
        value, signal, long_signal, short_signal = (
            mp.nonlinear_regression_zero_lag_moving_average(close)
        )

        assert len(value) == len(close)
        assert len(signal) == len(close)
        assert len(long_signal) == len(close)
        assert len(short_signal) == len(close)
        assert int(np.flatnonzero(~np.isnan(value))[0]) == 42
        assert int(np.flatnonzero(~np.isnan(signal))[0]) == 43

    def test_invalid_params(self, test_data):
        close = np.asarray(test_data["close"][:128], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid zlma_period"):
            mp.nonlinear_regression_zero_lag_moving_average(close, zlma_period=0)

        with pytest.raises(ValueError, match="Invalid regression_period"):
            mp.nonlinear_regression_zero_lag_moving_average(close, regression_period=0)

    def test_stream_matches_batch(self, test_data):
        close = np.asarray(test_data["close"][:320], dtype=np.float64)
        batch = mp.nonlinear_regression_zero_lag_moving_average(
            close,
            zlma_period=17,
            regression_period=11,
        )

        stream = mp.NonlinearRegressionZeroLagMovingAverageStream(
            zlma_period=17,
            regression_period=11,
        )

        value = []
        signal = []
        long_signal = []
        short_signal = []
        for sample in close:
            out = stream.update(float(sample))
            if out is None:
                value.append(np.nan)
                signal.append(np.nan)
                long_signal.append(0.0)
                short_signal.append(0.0)
            else:
                v, s, lng, sht = out
                value.append(v)
                signal.append(s)
                long_signal.append(lng)
                short_signal.append(sht)

        np.testing.assert_allclose(value, batch[0], rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(signal, batch[1], rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(long_signal, batch[2], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(short_signal, batch[3], rtol=1e-12, atol=1e-12)

    def test_batch_single_param_matches_single(self, test_data):
        close = np.asarray(test_data["close"][:280], dtype=np.float64)
        batch = mp.nonlinear_regression_zero_lag_moving_average_batch(
            close,
            zlma_period_range=(15, 15, 0),
            regression_period_range=(15, 15, 0),
        )
        single = mp.nonlinear_regression_zero_lag_moving_average(close)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_allclose(
            batch["value"][0], single[0], rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            batch["signal"][0], single[1], rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_multiple_ranges(self, test_data):
        close = np.asarray(test_data["close"][:220], dtype=np.float64)
        batch = mp.nonlinear_regression_zero_lag_moving_average_batch(
            close,
            zlma_period_range=(14, 16, 1),
            regression_period_range=(10, 12, 1),
        )

        assert batch["rows"] == 9
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(
            batch["zlma_periods"],
            np.array([14, 14, 14, 15, 15, 15, 16, 16, 16], dtype=np.uint64),
        )
        np.testing.assert_array_equal(
            batch["regression_periods"],
            np.array([10, 11, 12, 10, 11, 12, 10, 11, 12], dtype=np.uint64),
        )
