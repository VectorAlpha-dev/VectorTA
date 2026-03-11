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


class TestKeltnerChannelWidthOscillator:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = np.asarray(test_data["high"][:256], dtype=np.float64)
        low = np.asarray(test_data["low"][:256], dtype=np.float64)
        close = np.asarray(test_data["close"][:256], dtype=np.float64)
        source = np.asarray((high + low + close) / 3.0, dtype=np.float64)

        kbw, kbw_sma = mp.keltner_channel_width_oscillator(
            high, low, close, source, 20, 2.0, True, "Average True Range", 10
        )

        assert len(kbw) == len(source)
        assert len(kbw_sma) == len(source)
        assert np.isnan(kbw[:19]).all()
        assert np.isnan(kbw_sma[:38]).all()
        assert np.isfinite(kbw[19:]).any()
        assert np.isfinite(kbw_sma[38:]).any()

    def test_kernel_parity(self, test_data):
        high = np.asarray(test_data["high"][:192], dtype=np.float64)
        low = np.asarray(test_data["low"][:192], dtype=np.float64)
        close = np.asarray(test_data["close"][:192], dtype=np.float64)
        source = np.asarray(test_data["close"][:192], dtype=np.float64)

        auto_kbw, auto_kbw_sma = mp.keltner_channel_width_oscillator(
            high, low, close, source, 20, 2.0, True, "Average True Range", 10
        )
        scalar_kbw, scalar_kbw_sma = mp.keltner_channel_width_oscillator(
            high,
            low,
            close,
            source,
            20,
            2.0,
            True,
            "Average True Range",
            10,
            kernel="scalar",
        )

        np.testing.assert_allclose(
            auto_kbw, scalar_kbw, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            auto_kbw_sma, scalar_kbw_sma, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_invalid_bands_style(self, test_data):
        high = np.asarray(test_data["high"][:64], dtype=np.float64)
        low = np.asarray(test_data["low"][:64], dtype=np.float64)
        close = np.asarray(test_data["close"][:64], dtype=np.float64)
        source = np.asarray(test_data["close"][:64], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid bands style"):
            mp.keltner_channel_width_oscillator(
                high, low, close, source, 20, 2.0, True, "bogus", 10
            )

    def test_stream_matches_batch(self, test_data):
        high = np.asarray(test_data["high"][:192], dtype=np.float64)
        low = np.asarray(test_data["low"][:192], dtype=np.float64)
        close = np.asarray(test_data["close"][:192], dtype=np.float64)
        source = np.asarray((high + low + close) / 3.0, dtype=np.float64)

        batch_kbw, batch_kbw_sma = mp.keltner_channel_width_oscillator(
            high, low, close, source, 16, 1.8, False, "Range", 10
        )
        stream = mp.KeltnerChannelWidthOscillatorStream(16, 1.8, False, "Range", 10)
        stream_kbw = []
        stream_kbw_sma = []

        for h, l, c, s in zip(high, low, close, source):
            out = stream.update(float(h), float(l), float(c), float(s))
            if out is None:
                stream_kbw.append(np.nan)
                stream_kbw_sma.append(np.nan)
            else:
                w, sm = out
                stream_kbw.append(w)
                stream_kbw_sma.append(sm)

        np.testing.assert_allclose(
            stream_kbw, batch_kbw, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_kbw_sma, batch_kbw_sma, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        high = np.asarray(test_data["high"][:192], dtype=np.float64)
        low = np.asarray(test_data["low"][:192], dtype=np.float64)
        close = np.asarray(test_data["close"][:192], dtype=np.float64)
        source = np.asarray(test_data["close"][:192], dtype=np.float64)

        result = mp.keltner_channel_width_oscillator_batch(
            high,
            low,
            close,
            source,
            length_range=(20, 20, 0),
            multiplier_range=(2.0, 2.0, 0.0),
            atr_length_range=(10, 10, 0),
            use_exponential=True,
            bands_style="Average True Range",
        )

        assert result["rows"] == 1
        assert result["cols"] == len(source)
        assert result["kbw"].shape == (1, len(source))
        assert result["kbw_sma"].shape == (1, len(source))
        np.testing.assert_array_equal(result["lengths"], np.array([20], dtype=np.uint64))
        np.testing.assert_allclose(result["multipliers"], np.array([2.0]), atol=1e-12)
        np.testing.assert_array_equal(result["atr_lengths"], np.array([10], dtype=np.uint64))
        assert result["use_exponential"] == [True]
        assert result["bands_styles"] == ["Average True Range"]

        single_kbw, single_kbw_sma = mp.keltner_channel_width_oscillator(
            high, low, close, source, 20, 2.0, True, "Average True Range", 10
        )
        np.testing.assert_allclose(
            result["kbw"][0], single_kbw, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["kbw_sma"][0],
            single_kbw_sma,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
