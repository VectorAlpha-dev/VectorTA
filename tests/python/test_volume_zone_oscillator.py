import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python' first",
        allow_module_level=True,
    )


def manual_vzo(close: np.ndarray, volume: np.ndarray, length: int, intraday_smoothing: bool, noise_filter: int) -> np.ndarray:
    alpha = 2.0 / (length + 1.0)
    beta = 1.0 - alpha
    smooth_alpha = 2.0 / (noise_filter + 1.0)
    smooth_beta = 1.0 - smooth_alpha

    out = np.full(close.shape[0], np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(volume))
    if valid.size == 0:
        return out

    first_valid = int(valid[0])
    prev_close = np.nan
    ema_direction = 0.0
    ema_total = 0.0
    smooth = 0.0
    smooth_valid = False

    for i in range(first_valid, close.shape[0]):
        raw = np.nan
        if np.isfinite(volume[i]):
            directed = volume[i] if np.isfinite(close[i]) and np.isfinite(prev_close) and close[i] > prev_close else -volume[i]
            ema_direction = beta * ema_direction + alpha * directed
            ema_total = beta * ema_total + alpha * volume[i]
            if ema_total != 0.0:
                raw = 100.0 * ema_direction / ema_total
        elif ema_total != 0.0:
            raw = 100.0 * ema_direction / ema_total

        if np.isfinite(close[i]):
            prev_close = close[i]

        if intraday_smoothing:
            if np.isfinite(raw):
                smooth = smooth_beta * smooth + smooth_alpha * raw
                smooth_valid = True
                out[i] = smooth
            elif smooth_valid:
                out[i] = smooth
        else:
            out[i] = raw

    return out


class TestVolumeZoneOscillator:
    def test_manual_reference_matches_binding(self):
        data = load_test_data()
        close = np.asarray(data["close"][:160], dtype=np.float64)
        volume = np.asarray(data["volume"][:160], dtype=np.float64)
        expected = manual_vzo(close, volume, 14, True, 4)
        result = mp.volume_zone_oscillator(
            close,
            volume,
            length=14,
            intraday_smoothing=True,
            noise_filter=4,
        )
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_stream_matches_batch(self):
        data = load_test_data()
        close = np.asarray(data["close"][:128], dtype=np.float64)
        volume = np.asarray(data["volume"][:128], dtype=np.float64)
        batch = mp.volume_zone_oscillator(
            close,
            volume,
            length=10,
            intraday_smoothing=False,
            noise_filter=4,
        )
        stream = mp.VolumeZoneOscillatorStream(10, False, 4)
        streamed = np.array([stream.update(float(c), float(v)) for c, v in zip(close, volume)])
        np.testing.assert_allclose(streamed, batch, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_batch_metadata_and_first_row_match_single(self):
        data = load_test_data()
        close = np.asarray(data["close"][:128], dtype=np.float64)
        volume = np.asarray(data["volume"][:128], dtype=np.float64)
        result = mp.volume_zone_oscillator_batch(
            close,
            volume,
            length_range=(14, 16, 2),
            intraday_smoothing=True,
            noise_filter_range=(4, 4, 0),
        )
        assert result["rows"] == 2
        assert result["cols"] == close.shape[0]
        np.testing.assert_array_equal(result["lengths"], np.array([14, 16], dtype=np.uint64))
        np.testing.assert_array_equal(result["noise_filters"], np.array([4, 4], dtype=np.uint64))
        assert result["intraday_smoothing"] == [True, True]

        single = mp.volume_zone_oscillator(
            close,
            volume,
            length=14,
            intraday_smoothing=True,
            noise_filter=4,
        )
        np.testing.assert_allclose(
            result["values"][0],
            single,
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )

    def test_kernel_parity(self):
        data = load_test_data()
        close = np.asarray(data["close"][:144], dtype=np.float64)
        volume = np.asarray(data["volume"][:144], dtype=np.float64)
        auto = mp.volume_zone_oscillator(close, volume, length=14, intraday_smoothing=True, noise_filter=4)
        scalar = mp.volume_zone_oscillator(
            close,
            volume,
            length=14,
            intraday_smoothing=True,
            noise_filter=4,
            kernel="scalar",
        )
        np.testing.assert_allclose(auto, scalar, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_invalid_length(self):
        data = load_test_data()
        close = np.asarray(data["close"][:32], dtype=np.float64)
        volume = np.asarray(data["volume"][:32], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid length"):
            mp.volume_zone_oscillator(close, volume, length=1, intraday_smoothing=True, noise_filter=4)

    def test_length_mismatch(self):
        data = load_test_data()
        close = np.asarray(data["close"][:32], dtype=np.float64)
        volume = np.asarray(data["volume"][:31], dtype=np.float64)
        with pytest.raises(ValueError, match="slice length"):
            mp.volume_zone_oscillator(close, volume, length=14, intraday_smoothing=True, noise_filter=4)
