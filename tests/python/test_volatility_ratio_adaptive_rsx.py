import numpy as np
import pytest

from test_utils import load_test_data

try:
    import my_project as mp
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python' first",
        allow_module_level=True,
    )


class TestVolatilityRatioAdaptiveRsx:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = np.asarray(test_data["close"][:256], dtype=np.float64)

        line, signal = mp.volatility_ratio_adaptive_rsx(close, 6, 0.5)

        assert len(line) == len(close)
        assert len(signal) == len(close)
        assert np.isnan(line[:10]).all()
        assert np.isnan(signal[:11]).all()
        assert np.isfinite(line[10:]).any()
        assert np.isfinite(signal[11:]).any()

    def test_kernel_parity(self, test_data):
        close = np.asarray(test_data["close"][:256], dtype=np.float64)

        auto_line, auto_signal = mp.volatility_ratio_adaptive_rsx(close, 6, 0.5)
        scalar_line, scalar_signal = mp.volatility_ratio_adaptive_rsx(
            close, 6, 0.5, kernel="scalar"
        )

        np.testing.assert_allclose(
            auto_line, scalar_line, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            auto_signal, scalar_signal, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_invalid_speed(self, test_data):
        close = np.asarray(test_data["close"][:128], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid speed"):
            mp.volatility_ratio_adaptive_rsx(close, 6, 1.5)

    def test_stream_matches_batch(self, test_data):
        close = np.asarray(test_data["close"][:256], dtype=np.float64)

        batch_line, batch_signal = mp.volatility_ratio_adaptive_rsx(close, 6, 0.5)
        stream = mp.VolatilityRatioAdaptiveRsxStream(6, 0.5)
        stream_line = []
        stream_signal = []

        for value in close:
            out = stream.update(float(value))
            if out is None:
                stream_line.append(np.nan)
                stream_signal.append(np.nan)
            else:
                v0, v1 = out
                stream_line.append(v0)
                stream_signal.append(v1)

        np.testing.assert_allclose(
            stream_line, batch_line, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_signal, batch_signal, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = np.asarray(test_data["close"][:192], dtype=np.float64)

        result = mp.volatility_ratio_adaptive_rsx_batch(
            close,
            period_range=(6, 6, 0),
            speed_range=(0.5, 0.5, 0.0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(close)
        assert result["line"].shape == (1, len(close))
        assert result["signal"].shape == (1, len(close))
        np.testing.assert_array_equal(result["periods"], np.array([6], dtype=np.uint64))
        np.testing.assert_allclose(result["speeds"], np.array([0.5]), atol=1e-12)

        single_line, single_signal = mp.volatility_ratio_adaptive_rsx(close, 6, 0.5)
        np.testing.assert_allclose(
            result["line"][0], single_line, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["signal"][0], single_signal, rtol=1e-10, atol=1e-10, equal_nan=True
        )
