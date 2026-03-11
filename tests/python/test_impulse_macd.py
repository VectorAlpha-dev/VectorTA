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


class TestImpulseMacd:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = np.asarray(test_data["high"][:720], dtype=np.float64)
        low = np.asarray(test_data["low"][:720], dtype=np.float64)
        close = np.asarray(test_data["close"][:720], dtype=np.float64)

        impulse_macd, impulse_histo, signal = mp.impulse_macd(high, low, close, 34, 9)

        assert len(impulse_macd) == len(close)
        assert len(impulse_histo) == len(close)
        assert len(signal) == len(close)
        assert np.isfinite(impulse_macd[:8]).any()
        assert np.isnan(impulse_histo[:8]).all()
        assert np.isnan(signal[:8]).all()
        assert np.isfinite(impulse_macd).any()
        assert np.isfinite(impulse_histo[8:]).any()
        assert np.isfinite(signal[8:]).any()

    def test_kernel_parity(self, test_data):
        high = np.asarray(test_data["high"][:512], dtype=np.float64)
        low = np.asarray(test_data["low"][:512], dtype=np.float64)
        close = np.asarray(test_data["close"][:512], dtype=np.float64)

        auto_md, auto_hist, auto_signal = mp.impulse_macd(high, low, close, 34, 9)
        scalar_md, scalar_hist, scalar_signal = mp.impulse_macd(
            high, low, close, 34, 9, kernel="scalar"
        )

        np.testing.assert_allclose(
            auto_md, scalar_md, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            auto_hist, scalar_hist, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            auto_signal, scalar_signal, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_invalid_length_signal(self, test_data):
        high = np.asarray(test_data["high"][:128], dtype=np.float64)
        low = np.asarray(test_data["low"][:128], dtype=np.float64)
        close = np.asarray(test_data["close"][:128], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid length_signal"):
            mp.impulse_macd(high, low, close, 34, 0)

    def test_stream_matches_batch(self, test_data):
        high = np.asarray(test_data["high"][:320], dtype=np.float64)
        low = np.asarray(test_data["low"][:320], dtype=np.float64)
        close = np.asarray(test_data["close"][:320], dtype=np.float64)

        batch_md, batch_hist, batch_signal = mp.impulse_macd(high, low, close, 34, 9)
        stream = mp.ImpulseMacdStream(34, 9)
        stream_md = []
        stream_hist = []
        stream_signal = []

        for h, l, c in zip(high, low, close):
            out = stream.update(float(h), float(l), float(c))
            if out is None:
                stream_md.append(np.nan)
                stream_hist.append(np.nan)
                stream_signal.append(np.nan)
            else:
                a, b, d = out
                stream_md.append(a)
                stream_hist.append(b)
                stream_signal.append(d)

        np.testing.assert_allclose(
            stream_md, batch_md, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_hist, batch_hist, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_signal, batch_signal, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        high = np.asarray(test_data["high"][:320], dtype=np.float64)
        low = np.asarray(test_data["low"][:320], dtype=np.float64)
        close = np.asarray(test_data["close"][:320], dtype=np.float64)

        result = mp.impulse_macd_batch(
            high,
            low,
            close,
            length_ma_range=(34, 34, 0),
            length_signal_range=(9, 9, 0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(close)
        assert result["impulse_macd"].shape == (1, len(close))
        assert result["impulse_histo"].shape == (1, len(close))
        assert result["signal"].shape == (1, len(close))
        np.testing.assert_array_equal(result["length_mas"], np.array([34], dtype=np.uint64))
        np.testing.assert_array_equal(
            result["length_signals"], np.array([9], dtype=np.uint64)
        )

        single_md, single_hist, single_signal = mp.impulse_macd(high, low, close, 34, 9)
        np.testing.assert_allclose(
            result["impulse_macd"][0],
            single_md,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            result["impulse_histo"][0],
            single_hist,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            result["signal"][0],
            single_signal,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
