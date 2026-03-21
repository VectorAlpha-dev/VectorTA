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


def manual_andean(open_, close, length=50, signal_length=9):
    bull = np.full(open_.shape[0], np.nan, dtype=np.float64)
    bear = np.full(open_.shape[0], np.nan, dtype=np.float64)
    signal = np.full(open_.shape[0], np.nan, dtype=np.float64)
    alpha = 2.0 / (length + 1.0)
    signal_alpha = 2.0 / (signal_length + 1.0)

    initialized = False
    up1 = up2 = dn1 = dn2 = sig = np.nan

    for i in range(open_.shape[0]):
        o = open_[i]
        c = close[i]
        if not np.isfinite(o) or not np.isfinite(c):
            continue

        c2 = c * c
        o2 = o * o

        if not initialized:
            up1 = c
            up2 = c2
            dn1 = c
            dn2 = c2
            sig = 0.0
            bull[i] = 0.0
            bear[i] = 0.0
            signal[i] = 0.0
            initialized = True
            continue

        up1 = max(c, o, up1 - (up1 - c) * alpha)
        up2 = max(c2, o2, up2 - (up2 - c2) * alpha)
        dn1 = min(c, o, dn1 + (c - dn1) * alpha)
        dn2 = min(c2, o2, dn2 + (c2 - dn2) * alpha)

        bull[i] = np.sqrt(max(0.0, dn2 - dn1 * dn1))
        bear[i] = np.sqrt(max(0.0, up2 - up1 * up1))
        src = max(bull[i], bear[i])
        sig = signal_alpha * src + (1.0 - signal_alpha) * sig
        signal[i] = sig

    return bull, bear, signal


class TestAndeanOscillator:
    def test_manual_reference_matches_binding(self):
        data = load_test_data()
        open_ = np.asarray(data["open"][:160], dtype=np.float64)
        close = np.asarray(data["close"][:160], dtype=np.float64)
        expected_bull, expected_bear, expected_signal = manual_andean(open_, close, 50, 9)
        result = mp.andean_oscillator(open_, close, length=50, signal_length=9)
        np.testing.assert_allclose(result["bull"], expected_bull, rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(result["bear"], expected_bear, rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            result["signal"], expected_signal, rtol=0.0, atol=1e-12, equal_nan=True
        )

    def test_stream_matches_batch(self):
        data = load_test_data()
        open_ = np.asarray(data["open"][:144], dtype=np.float64)
        close = np.asarray(data["close"][:144], dtype=np.float64)
        batch = mp.andean_oscillator(open_, close, length=50, signal_length=9)
        stream = mp.AndeanOscillatorStream(50, 9)
        streamed = np.array([stream.update(float(o), float(c)) for o, c in zip(open_, close)], dtype=np.float64)
        np.testing.assert_allclose(streamed[:, 0], batch["bull"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(streamed[:, 1], batch["bear"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(streamed[:, 2], batch["signal"], rtol=0.0, atol=1e-12, equal_nan=True)

    def test_batch_metadata_and_first_row_match_single(self):
        data = load_test_data()
        open_ = np.asarray(data["open"][:128], dtype=np.float64)
        close = np.asarray(data["close"][:128], dtype=np.float64)
        result = mp.andean_oscillator_batch(
            open_,
            close,
            length_range=(50, 52, 2),
            signal_length_range=(9, 11, 2),
        )
        assert result["rows"] == 4
        assert result["cols"] == open_.shape[0]
        np.testing.assert_allclose(result["lengths"], np.array([50, 50, 52, 52]), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            result["signal_lengths"], np.array([9, 11, 9, 11]), rtol=0.0, atol=0.0
        )
        single = mp.andean_oscillator(open_, close, length=50, signal_length=9)
        np.testing.assert_allclose(result["bull"][0], single["bull"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(result["bear"][0], single["bear"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            result["signal"][0], single["signal"], rtol=0.0, atol=1e-12, equal_nan=True
        )

    def test_invalid_parameters(self):
        data = load_test_data()
        open_ = np.asarray(data["open"][:64], dtype=np.float64)
        close = np.asarray(data["close"][:64], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid length"):
            mp.andean_oscillator(open_, close, length=0, signal_length=9)
        with pytest.raises(ValueError, match="Invalid signal_length"):
            mp.andean_oscillator(open_, close, length=50, signal_length=0)
