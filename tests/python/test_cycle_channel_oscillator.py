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


def manual_rma(values: np.ndarray, length: int) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    total = 0.0
    count = 0
    current = np.nan
    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue
        if count < length:
            total += value
            count += 1
            if count == length:
                current = total / float(length)
                out[i] = current
        else:
            current = current + (value - current) / float(length)
            count += 1
            out[i] = current
    return out


def manual_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    tr = np.full(close.shape[0], np.nan, dtype=np.float64)
    prev_close = None
    for i in range(close.shape[0]):
        if not (np.isfinite(high[i]) and np.isfinite(low[i]) and np.isfinite(close[i])):
            continue
        if prev_close is None:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - prev_close),
                abs(low[i] - prev_close),
            )
        prev_close = close[i]
    return manual_rma(tr, length)


def delayed_or_current(values: np.ndarray, idx: int, delay: int, current: float) -> float:
    if idx >= delay and np.isfinite(values[idx - delay]):
        return float(values[idx - delay])
    return current


def manual_cycle_channel_oscillator(
    source: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    short_cycle_length: int,
    medium_cycle_length: int,
    medium_multiplier: float,
):
    short_period = short_cycle_length // 2
    medium_period = medium_cycle_length // 2
    short_delay = short_period // 2
    medium_delay = medium_period // 2

    short_ma = manual_rma(source, short_period)
    medium_ma = manual_rma(source, medium_period)
    medium_atr = manual_atr(high, low, close, medium_period)

    fast = np.full(source.shape[0], np.nan, dtype=np.float64)
    slow = np.full(source.shape[0], np.nan, dtype=np.float64)
    for i in range(source.shape[0]):
        if not (
            np.isfinite(source[i])
            and np.isfinite(high[i])
            and np.isfinite(low[i])
            and np.isfinite(close[i])
        ):
            continue
        medium_center = delayed_or_current(medium_ma, i, medium_delay, float(source[i]))
        short_center = delayed_or_current(short_ma, i, short_delay, float(source[i]))
        offset = medium_multiplier * medium_atr[i]
        denom = 2.0 * offset
        if np.isfinite(denom) and denom != 0.0:
            medium_bottom = medium_center - offset
            fast[i] = (source[i] - medium_bottom) / denom
            slow[i] = (short_center - medium_bottom) / denom
    return fast, slow


class TestCycleChannelOscillator:
    def test_manual_reference_matches_binding(self):
        data = load_test_data()
        n = 160
        source = np.asarray(data["close"][:n], dtype=np.float64)
        high = np.asarray(data["high"][:n], dtype=np.float64)
        low = np.asarray(data["low"][:n], dtype=np.float64)
        close = np.asarray(data["close"][:n], dtype=np.float64)
        expected_fast, expected_slow = manual_cycle_channel_oscillator(
            source, high, low, close, 10, 30, 3.0
        )
        result = mp.cycle_channel_oscillator(
            source,
            high,
            low,
            close,
            short_cycle_length=10,
            medium_cycle_length=30,
            short_multiplier=1.0,
            medium_multiplier=3.0,
        )
        np.testing.assert_allclose(result["fast"], expected_fast, rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(result["slow"], expected_slow, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_stream_matches_batch(self):
        data = load_test_data()
        n = 128
        source = np.asarray(data["close"][:n], dtype=np.float64)
        high = np.asarray(data["high"][:n], dtype=np.float64)
        low = np.asarray(data["low"][:n], dtype=np.float64)
        close = np.asarray(data["close"][:n], dtype=np.float64)
        batch = mp.cycle_channel_oscillator(source, high, low, close)
        stream = mp.CycleChannelOscillatorStream()
        streamed = np.array(
            [stream.update(float(s), float(h), float(l), float(c)) for s, h, l, c in zip(source, high, low, close)]
        )
        np.testing.assert_allclose(streamed[:, 0], batch["fast"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(streamed[:, 1], batch["slow"], rtol=0.0, atol=1e-12, equal_nan=True)

    def test_batch_metadata_and_first_row_match_single(self):
        data = load_test_data()
        n = 96
        source = np.asarray(data["close"][:n], dtype=np.float64)
        high = np.asarray(data["high"][:n], dtype=np.float64)
        low = np.asarray(data["low"][:n], dtype=np.float64)
        close = np.asarray(data["close"][:n], dtype=np.float64)
        result = mp.cycle_channel_oscillator_batch(
            source,
            high,
            low,
            close,
            short_cycle_length_range=(10, 10, 0),
            medium_cycle_length_range=(30, 32, 2),
            short_multiplier_range=(1.0, 1.0, 0.0),
            medium_multiplier_range=(3.0, 3.0, 0.0),
        )
        assert result["rows"] == 2
        assert result["cols"] == n
        np.testing.assert_array_equal(result["short_cycle_lengths"], np.array([10, 10], dtype=np.uint64))
        np.testing.assert_array_equal(result["medium_cycle_lengths"], np.array([30, 32], dtype=np.uint64))

        single = mp.cycle_channel_oscillator(source, high, low, close)
        np.testing.assert_allclose(result["fast"][0], single["fast"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(result["slow"][0], single["slow"], rtol=0.0, atol=1e-12, equal_nan=True)

    def test_kernel_parity(self):
        data = load_test_data()
        n = 144
        source = np.asarray(data["close"][:n], dtype=np.float64)
        high = np.asarray(data["high"][:n], dtype=np.float64)
        low = np.asarray(data["low"][:n], dtype=np.float64)
        close = np.asarray(data["close"][:n], dtype=np.float64)
        auto = mp.cycle_channel_oscillator(source, high, low, close)
        scalar = mp.cycle_channel_oscillator(source, high, low, close, kernel="scalar")
        np.testing.assert_allclose(auto["fast"], scalar["fast"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(auto["slow"], scalar["slow"], rtol=0.0, atol=1e-12, equal_nan=True)

    def test_invalid_length(self):
        data = load_test_data()
        source = np.asarray(data["close"][:32], dtype=np.float64)
        high = np.asarray(data["high"][:32], dtype=np.float64)
        low = np.asarray(data["low"][:32], dtype=np.float64)
        close = np.asarray(data["close"][:32], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid cycle length"):
            mp.cycle_channel_oscillator(source, high, low, close, short_cycle_length=1)
