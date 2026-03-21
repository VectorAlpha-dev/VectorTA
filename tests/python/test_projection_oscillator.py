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


class TestProjectionOscillator:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]
        pbo, signal = mp.projection_oscillator(high, low, close, 14, 4)

        assert len(pbo) == len(close)
        assert len(signal) == len(close)
        pbo_valid = np.flatnonzero(~np.isnan(pbo))
        signal_valid = np.flatnonzero(~np.isnan(signal))
        assert pbo_valid.size > 0
        assert signal_valid.size > 0
        assert int(pbo_valid[0]) == 29
        assert int(signal_valid[0]) == 32

    def test_kernel_parity(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]
        auto_pbo, auto_signal = mp.projection_oscillator(high, low, close, 14, 4)
        scalar_pbo, scalar_signal = mp.projection_oscillator(
            high, low, close, 14, 4, kernel="scalar"
        )
        np.testing.assert_allclose(auto_pbo, scalar_pbo, rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            auto_signal, scalar_signal, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_invalid_params(self, test_data):
        high = test_data["high"][:64]
        low = test_data["low"][:64]
        close = test_data["close"][:64]

        with pytest.raises(ValueError, match="Invalid length"):
            mp.projection_oscillator(high, low, close, 0, 4)

        with pytest.raises(ValueError, match="Invalid smooth_length"):
            mp.projection_oscillator(high, low, close, 14, 0)

    def test_stream_matches_batch(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]
        batch_pbo, batch_signal = mp.projection_oscillator(high, low, close, 14, 4)

        stream = mp.ProjectionOscillatorStream(14, 4)
        stream_pbo = []
        stream_signal = []
        for h, l, c in zip(high, low, close):
            out = stream.update(float(h), float(l), float(c))
            if out is None:
                stream_pbo.append(np.nan)
                stream_signal.append(np.nan)
            else:
                pbo, signal = out
                stream_pbo.append(pbo)
                stream_signal.append(signal)

        np.testing.assert_allclose(
            stream_pbo, batch_pbo, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_signal, batch_signal, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]
        batch = mp.projection_oscillator_batch(
            high,
            low,
            close,
            length_range=(14, 14, 0),
            smooth_length_range=(4, 4, 0),
        )
        single_pbo, single_signal = mp.projection_oscillator(high, low, close, 14, 4)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        assert batch["pbo"].shape == (1, len(close))
        assert batch["signal"].shape == (1, len(close))
        np.testing.assert_allclose(
            batch["pbo"][0], single_pbo, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            batch["signal"][0], single_signal, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_multiple_ranges(self, test_data):
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]
        batch = mp.projection_oscillator_batch(
            high,
            low,
            close,
            length_range=(10, 14, 2),
            smooth_length_range=(4, 4, 0),
        )

        assert batch["rows"] == 3
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(
            batch["lengths"], np.array([10, 12, 14], dtype=np.uint64)
        )
        np.testing.assert_array_equal(
            batch["smooth_lengths"], np.array([4, 4, 4], dtype=np.uint64)
        )
