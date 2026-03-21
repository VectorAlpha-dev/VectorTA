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


class TestGmmaOscillator:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = np.asarray(test_data["close"][:256], dtype=np.float64)
        oscillator, signal = mp.gmma_oscillator(close, "guppy", 3, 13)

        assert len(oscillator) == len(close)
        assert len(signal) == len(close)
        assert int(np.flatnonzero(~np.isnan(oscillator))[0]) == 2
        assert int(np.flatnonzero(~np.isnan(signal))[0]) == 0

    def test_invalid_params(self, test_data):
        close = np.asarray(test_data["close"][:64], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid smooth_length"):
            mp.gmma_oscillator(close, "guppy", 0, 13)

        with pytest.raises(ValueError, match="Invalid GMMA type"):
            mp.gmma_oscillator(close, "bad_mode", 1, 13)

    def test_stream_matches_batch(self, test_data):
        close = np.asarray(test_data["close"][:320], dtype=np.float64)
        batch_oscillator, batch_signal = mp.gmma_oscillator(close, "super_guppy", 4, 9)

        stream = mp.GmmaOscillatorStream("super_guppy", 4, 9)
        osc_stream = []
        signal_stream = []
        for value in close:
            out = stream.update(float(value))
            if out is None:
                osc_stream.append(np.nan)
                signal_stream.append(np.nan)
            else:
                osc_value, sig_value = out
                osc_stream.append(osc_value)
                signal_stream.append(sig_value)

        np.testing.assert_allclose(
            osc_stream, batch_oscillator, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            signal_stream, batch_signal, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = np.asarray(test_data["close"][:280], dtype=np.float64)
        batch = mp.gmma_oscillator_batch(
            close,
            "guppy",
            smooth_length_range=(3, 3, 0),
            signal_length_range=(13, 13, 0),
        )
        single = mp.gmma_oscillator(close, "guppy", 3, 13)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_allclose(
            batch["oscillator"][0], single[0], rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            batch["signal"][0], single[1], rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_multiple_ranges(self, test_data):
        close = np.asarray(test_data["close"][:220], dtype=np.float64)
        batch = mp.gmma_oscillator_batch(
            close,
            "super_guppy",
            smooth_length_range=(1, 3, 1),
            signal_length_range=(9, 13, 2),
        )

        assert batch["rows"] == 9
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(
            batch["smooth_lengths"],
            np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.uint64),
        )
        np.testing.assert_array_equal(
            batch["signal_lengths"],
            np.array([9, 11, 13, 9, 11, 13, 9, 11, 13], dtype=np.uint64),
        )
        assert batch["gmma_types"] == ["super_guppy"] * 9
