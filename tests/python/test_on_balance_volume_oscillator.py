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


class TestOnBalanceVolumeOscillator:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        source = np.asarray(test_data["close"][:256], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:256], dtype=np.float64)

        line, signal = mp.on_balance_volume_oscillator(source, volume, 20, 9)

        assert len(line) == len(source)
        assert len(signal) == len(source)
        assert np.isnan(line[:19]).all()
        assert np.isnan(signal[:19]).all()
        assert np.isfinite(line[19:]).any()
        assert np.isfinite(signal[19:]).any()

    def test_kernel_parity(self, test_data):
        source = np.asarray(test_data["close"][:256], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:256], dtype=np.float64)

        auto_line, auto_signal = mp.on_balance_volume_oscillator(source, volume, 20, 9)
        scalar_line, scalar_signal = mp.on_balance_volume_oscillator(
            source, volume, 20, 9, kernel="scalar"
        )

        np.testing.assert_allclose(
            auto_line, scalar_line, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            auto_signal, scalar_signal, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_stream_matches_batch(self, test_data):
        source = np.asarray(test_data["close"][:256], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:256], dtype=np.float64)

        batch_line, batch_signal = mp.on_balance_volume_oscillator(source, volume, 20, 9)
        stream = mp.OnBalanceVolumeOscillatorStream(20, 9)
        stream_line = []
        stream_signal = []

        for s, v in zip(source, volume):
            out = stream.update(float(s), float(v))
            if out is None:
                stream_line.append(np.nan)
                stream_signal.append(np.nan)
            else:
                stream_line.append(out[0])
                stream_signal.append(out[1])

        np.testing.assert_allclose(
            stream_line, batch_line, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_signal, batch_signal, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        source = np.asarray(test_data["close"][:192], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:192], dtype=np.float64)

        result = mp.on_balance_volume_oscillator_batch(
            source,
            volume,
            obv_length_range=(20, 20, 0),
            ema_length_range=(9, 9, 0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(source)
        assert result["line"].shape == (1, len(source))
        assert result["signal"].shape == (1, len(source))
        np.testing.assert_array_equal(result["obv_lengths"], np.array([20], dtype=np.uint64))
        np.testing.assert_array_equal(result["ema_lengths"], np.array([9], dtype=np.uint64))

        single_line, single_signal = mp.on_balance_volume_oscillator(source, volume, 20, 9)
        np.testing.assert_allclose(
            result["line"][0], single_line, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            result["signal"][0], single_signal, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_invalid_obv_length(self, test_data):
        source = np.asarray(test_data["close"][:64], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:64], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid OBV length"):
            mp.on_balance_volume_oscillator(source, volume, 0, 9)
