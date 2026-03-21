import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    try:
        import vector_ta as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


class TestVelocityAccelerationConvergenceDivergenceIndicator:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = np.asarray(test_data["high"][:256], dtype=np.float64)
        low = np.asarray(test_data["low"][:256], dtype=np.float64)
        close = np.asarray(test_data["close"][:256], dtype=np.float64)
        hlcc4 = (high + low + 2.0 * close) / 4.0
        vacd, signal = mp.velocity_acceleration_convergence_divergence_indicator(
            hlcc4, 21, 5
        )

        assert len(vacd) == len(hlcc4)
        assert len(signal) == len(hlcc4)
        assert int(np.flatnonzero(~np.isnan(vacd))[0]) == 4
        assert int(np.flatnonzero(~np.isnan(signal))[0]) == 4

    def test_invalid_params(self, test_data):
        close = np.asarray(test_data["close"][:64], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid length"):
            mp.velocity_acceleration_convergence_divergence_indicator(close, 1, 5)

    def test_stream_matches_batch(self, test_data):
        high = np.asarray(test_data["high"][:320], dtype=np.float64)
        low = np.asarray(test_data["low"][:320], dtype=np.float64)
        close = np.asarray(test_data["close"][:320], dtype=np.float64)
        hlcc4 = (high + low + 2.0 * close) / 4.0
        batch_vacd, batch_signal = (
            mp.velocity_acceleration_convergence_divergence_indicator(hlcc4, 21, 5)
        )

        stream = mp.VelocityAccelerationConvergenceDivergenceIndicatorStream(21, 5)
        vacd_stream = []
        signal_stream = []
        for value in hlcc4:
            out = stream.update(float(value))
            if out is None:
                vacd_stream.append(np.nan)
                signal_stream.append(np.nan)
            else:
                a, b = out
                vacd_stream.append(a)
                signal_stream.append(b)

        np.testing.assert_allclose(
            vacd_stream, batch_vacd, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            signal_stream, batch_signal, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        high = np.asarray(test_data["high"][:300], dtype=np.float64)
        low = np.asarray(test_data["low"][:300], dtype=np.float64)
        close = np.asarray(test_data["close"][:300], dtype=np.float64)
        hlcc4 = (high + low + 2.0 * close) / 4.0
        batch = mp.velocity_acceleration_convergence_divergence_indicator_batch(
            hlcc4,
            length_range=(21, 21, 0),
            smooth_length_range=(5, 5, 0),
        )
        single = mp.velocity_acceleration_convergence_divergence_indicator(hlcc4, 21, 5)

        assert batch["rows"] == 1
        assert batch["cols"] == len(hlcc4)
        np.testing.assert_allclose(
            batch["vacd"][0], single[0], rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            batch["signal"][0], single[1], rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_multiple_ranges(self, test_data):
        high = np.asarray(test_data["high"][:256], dtype=np.float64)
        low = np.asarray(test_data["low"][:256], dtype=np.float64)
        close = np.asarray(test_data["close"][:256], dtype=np.float64)
        hlcc4 = (high + low + 2.0 * close) / 4.0
        batch = mp.velocity_acceleration_convergence_divergence_indicator_batch(
            hlcc4,
            length_range=(21, 25, 2),
            smooth_length_range=(3, 5, 2),
        )

        assert batch["rows"] == 6
        assert batch["cols"] == len(hlcc4)
        np.testing.assert_array_equal(
            batch["lengths"], np.array([21, 21, 23, 23, 25, 25], dtype=np.uint64)
        )
        np.testing.assert_array_equal(
            batch["smooth_lengths"], np.array([3, 5, 3, 5, 3, 5], dtype=np.uint64)
        )
