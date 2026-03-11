import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    try:
        import my_project as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


def assert_close(a: np.ndarray, b: np.ndarray):
    np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-12, equal_nan=True)


def first_valid_index(high: np.ndarray, low: np.ndarray, close: np.ndarray, source: np.ndarray) -> int:
    valid = np.flatnonzero(
        np.isfinite(high) & np.isfinite(low) & np.isfinite(close) & np.isfinite(source)
    )
    return int(valid[0]) if valid.size else -1


def avg_finite(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(a.shape[0], np.nan, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    out[mask] = 0.5 * (a[mask] + b[mask])
    return out


def diff_finite(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(a.shape[0], np.nan, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    out[mask] = a[mask] - b[mask]
    return out


def rolling_midpoint(high: np.ndarray, low: np.ndarray, length: int, first: int) -> np.ndarray:
    out = np.full(high.shape[0], np.nan, dtype=np.float64)
    warm = first + length - 1
    for i in range(first, high.shape[0]):
        if not np.isfinite(high[i]) or not np.isfinite(low[i]):
            continue
        start = max(first, i + 1 - length)
        out[i] = 0.5 * (np.max(high[start : i + 1]) + np.min(low[start : i + 1]))
    out[:warm] = np.nan
    return out


def chebyshev_series(data: np.ndarray, length: int, ripple: float) -> np.ndarray:
    out = np.full(data.shape[0], np.nan, dtype=np.float64)
    a = np.cosh((1.0 / length) * np.arccosh(1.0 / (1.0 - ripple)))
    b = np.sinh((1.0 / length) * np.arcsinh(1.0 / ripple))
    c = (a - b) / (a + b)
    one_minus_c = 1.0 - c
    for i, value in enumerate(data):
        if not np.isfinite(value):
            continue
        prev = out[i - 1] if i > 0 and np.isfinite(out[i - 1]) else 0.0
        out[i] = one_minus_c * value + c * prev
    return out


def gaussian_value(x: float, bandwidth: float) -> float:
    return np.exp(-0.5 * (x / bandwidth) ** 2) / np.sqrt(2.0 * np.pi)


def gaussian_kernel_series(data: np.ndarray, size: int, h: float, r: float) -> np.ndarray:
    out = np.full(data.shape[0], np.nan, dtype=np.float64)
    weights = np.array([gaussian_value((i * i) / (h * h * r), r) for i in range(size + 1)])
    for i in range(size, data.shape[0]):
        window = data[i - size : i + 1][::-1]
        if np.all(np.isfinite(window)):
            out[i] = np.dot(window, weights) / np.sum(weights)
    return out


def smooth_series(data: np.ndarray, length: int, extra: bool) -> np.ndarray:
    cheb = chebyshev_series(data, length, 0.5)
    return gaussian_kernel_series(cheb, 4, 2.0, 1.0) if extra else cheb


def wma_series(data: np.ndarray, length: int) -> np.ndarray:
    out = np.full(data.shape[0], np.nan, dtype=np.float64)
    weights = np.arange(1, length + 1, dtype=np.float64)
    denom = np.sum(weights)
    for i in range(length - 1, data.shape[0]):
        window = data[i + 1 - length : i + 1]
        if np.all(np.isfinite(window)):
            out[i] = np.dot(window, weights) / denom
    return out


def shift_back(data: np.ndarray, shift: int) -> np.ndarray:
    out = np.full(data.shape[0], np.nan, dtype=np.float64)
    if shift == 0:
        out[:] = data
        return out
    out[shift:] = data[:-shift]
    return out


def rolling_rms_window(data: np.ndarray, window: int) -> np.ndarray:
    out = np.full(data.shape[0], np.nan, dtype=np.float64)
    queue = []
    sum_sq = 0.0
    for i, value in enumerate(data):
        if not np.isfinite(value):
            queue.clear()
            sum_sq = 0.0
            continue
        sq = value * value
        queue.append(sq)
        sum_sq += sq
        if len(queue) > window:
            sum_sq -= queue.pop(0)
        if len(queue) == window and window > 1:
            out[i] = np.sqrt(sum_sq / (window - 1))
    return out


def rms_all(signal: np.ndarray, gate: np.ndarray) -> np.ndarray:
    out = np.full(signal.shape[0], np.nan, dtype=np.float64)
    sum_sq = 0.0
    count = 0
    for i, value in enumerate(signal):
        if np.isfinite(value):
            sum_sq += value * value
            count += 1
        if np.isfinite(gate[i]) and count:
            out[i] = np.sqrt(sum_sq / count)
    return out


def normalize_value(value: float, min_level: float, max_level: float, mode: str, clamp: bool) -> float:
    if mode == "disabled":
        return value
    if (
        not np.isfinite(value)
        or not np.isfinite(min_level)
        or not np.isfinite(max_level)
        or min_level == max_level
    ):
        return np.nan
    scaled = (value - min_level) / (max_level - min_level)
    if clamp:
        scaled = min(max(scaled, 0.0), 1.0)
    return (scaled - 0.5) * 200.0


def manual_ichimoku(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    source: np.ndarray,
    conversion_periods: int = 9,
    base_periods: int = 26,
    lagging_span_periods: int = 52,
    displacement: int = 26,
    ma_length: int = 12,
    smoothing_length: int = 3,
    extra_smoothing: bool = True,
    normalize: str = "window",
    window_size: int = 20,
    clamp: bool = True,
    top_band: float = 2.0,
    mid_band: float = 1.5,
):
    first = first_valid_index(high, low, close, source)
    if first < 0:
        raise ValueError("No valid data")

    conversion_raw = rolling_midpoint(high, low, conversion_periods, first)
    base_raw = rolling_midpoint(high, low, base_periods, first)
    span_b_raw = rolling_midpoint(high, low, lagging_span_periods, first)

    kumo_a_input = avg_finite(conversion_raw, base_raw)
    kumo_a = smooth_series(kumo_a_input, smoothing_length, extra_smoothing)
    kumo_b = smooth_series(span_b_raw, smoothing_length, extra_smoothing)
    kumo_center = avg_finite(kumo_a, kumo_b)
    kumo_a_centered = diff_finite(kumo_a, kumo_center)
    kumo_b_centered = diff_finite(kumo_b, kumo_center)

    shift = displacement - 1
    kumo_center_offset = shift_back(kumo_center, shift)
    kumo_a_offset = shift_back(kumo_a, shift)
    kumo_b_offset = shift_back(kumo_b, shift)
    kumo_a_offset_centered = diff_finite(kumo_a_offset, kumo_center_offset)
    kumo_b_offset_centered = diff_finite(kumo_b_offset, kumo_center_offset)

    chikou_raw = shift_back(source, displacement + 1)
    chikou_input = diff_finite(source, chikou_raw)
    signal_input = diff_finite(source, kumo_center_offset)
    conversion_input = diff_finite(conversion_raw, kumo_center_offset)
    base_input = diff_finite(base_raw, kumo_center_offset)

    chikou = smooth_series(chikou_input, smoothing_length, extra_smoothing)
    signal = smooth_series(signal_input, smoothing_length, extra_smoothing)
    conversion = smooth_series(conversion_input, smoothing_length, extra_smoothing)
    base = smooth_series(base_input, smoothing_length, extra_smoothing)
    ma = wma_series(signal, ma_length)

    if normalize == "all":
        dev = rms_all(signal, kumo_a_offset)
    elif normalize == "window":
        dev = rolling_rms_window(signal, window_size)
    else:
        dev = np.zeros(close.shape[0], dtype=np.float64)

    max_level = np.where(np.isfinite(dev), dev * top_band, np.nan)
    min_level = np.where(np.isfinite(dev), -dev * top_band, np.nan)
    high_level = np.where(np.isfinite(dev), dev * mid_band, np.nan)
    low_level = np.where(np.isfinite(dev), -dev * mid_band, np.nan)

    out = {key: np.full(close.shape[0], np.nan, dtype=np.float64) for key in [
        "signal", "ma", "conversion", "base", "chikou",
        "current_kumo_a", "current_kumo_b", "future_kumo_a", "future_kumo_b",
        "max_level", "high_level", "low_level", "min_level",
    ]}
    for i in range(close.shape[0]):
        out["signal"][i] = normalize_value(signal[i], min_level[i], max_level[i], normalize, clamp)
        out["ma"][i] = normalize_value(ma[i], min_level[i], max_level[i], normalize, clamp)
        out["conversion"][i] = normalize_value(conversion[i], min_level[i], max_level[i], normalize, clamp)
        out["base"][i] = normalize_value(base[i], min_level[i], max_level[i], normalize, clamp)
        out["chikou"][i] = normalize_value(chikou[i], min_level[i], max_level[i], normalize, clamp)
        out["future_kumo_a"][i] = normalize_value(kumo_a_centered[i], min_level[i], max_level[i], normalize, clamp)
        out["future_kumo_b"][i] = normalize_value(kumo_b_centered[i], min_level[i], max_level[i], normalize, clamp)
        if i >= shift:
            out["current_kumo_a"][i] = normalize_value(
                kumo_a_offset_centered[i], min_level[i - shift], max_level[i - shift], normalize, clamp
            )
            out["current_kumo_b"][i] = normalize_value(
                kumo_b_offset_centered[i], min_level[i - shift], max_level[i - shift], normalize, clamp
            )
        out["max_level"][i] = normalize_value(max_level[i], min_level[i], max_level[i], normalize, clamp)
        out["high_level"][i] = normalize_value(high_level[i], min_level[i], max_level[i], normalize, clamp)
        out["low_level"][i] = normalize_value(low_level[i], min_level[i], max_level[i], normalize, clamp)
        out["min_level"][i] = normalize_value(min_level[i], min_level[i], max_level[i], normalize, clamp)
    return out


class TestIchimokuOscillator:
    def test_manual_reference_matches_binding(self):
        data = load_test_data()
        n = 192
        high = np.asarray(data["high"][:n], dtype=np.float64)
        low = np.asarray(data["low"][:n], dtype=np.float64)
        close = np.asarray(data["close"][:n], dtype=np.float64)
        source = close.copy()

        expected = manual_ichimoku(high, low, close, source)
        result = mp.ichimoku_oscillator(high, low, close, source=source)

        for key in [
            "signal",
            "ma",
            "conversion",
            "base",
            "chikou",
            "current_kumo_a",
            "future_kumo_b",
            "max_level",
        ]:
            np.testing.assert_allclose(
                np.asarray(result[key], dtype=np.float64),
                expected[key],
                rtol=0.0,
                atol=5e-12,
                equal_nan=True,
            )

    def test_batch_first_row_matches_single(self):
        data = load_test_data()
        n = 192
        high = np.asarray(data["high"][:n], dtype=np.float64)
        low = np.asarray(data["low"][:n], dtype=np.float64)
        close = np.asarray(data["close"][:n], dtype=np.float64)
        source = np.asarray(data["hlc3"][:n], dtype=np.float64) if "hlc3" in data else close

        single = mp.ichimoku_oscillator(high, low, close, source=source)
        batch = mp.ichimoku_oscillator_batch(
            high,
            low,
            close,
            source=source,
            conversion_periods_range=(9, 11, 2),
        )

        assert batch["rows"] == 2
        assert batch["cols"] == n
        assert_close(batch["signal"][0], single["signal"])
        assert_close(batch["current_kumo_a"][0], single["current_kumo_a"])

    def test_stream_last_value_matches_single(self):
        data = load_test_data()
        n = 160
        high = np.asarray(data["high"][:n], dtype=np.float64)
        low = np.asarray(data["low"][:n], dtype=np.float64)
        close = np.asarray(data["close"][:n], dtype=np.float64)
        source = close.copy()

        single = mp.ichimoku_oscillator(high, low, close, source=source)
        stream = mp.IchimokuOscillatorStream()
        last = None
        for h, l, c, s in zip(high, low, close, source):
            last = stream.update(float(h), float(l), float(c), float(s))

        assert last is not None
        assert np.isclose(last["signal"], single["signal"][-1], equal_nan=True)
        assert np.isclose(last["future_kumo_b"], single["future_kumo_b"][-1], equal_nan=True)

    def test_kernel_parity(self):
        data = load_test_data()
        n = 192
        high = np.asarray(data["high"][:n], dtype=np.float64)
        low = np.asarray(data["low"][:n], dtype=np.float64)
        close = np.asarray(data["close"][:n], dtype=np.float64)
        source = close.copy()

        auto = mp.ichimoku_oscillator(high, low, close, source=source)
        scalar = mp.ichimoku_oscillator(high, low, close, source=source, kernel="scalar")
        for key in ["signal", "current_kumo_a", "future_kumo_b"]:
            assert_close(np.asarray(auto[key]), np.asarray(scalar[key]))

    def test_invalid_period(self):
        data = load_test_data()
        high = np.asarray(data["high"][:64], dtype=np.float64)
        low = np.asarray(data["low"][:64], dtype=np.float64)
        close = np.asarray(data["close"][:64], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid period"):
            mp.ichimoku_oscillator(high, low, close, conversion_periods=0)

    def test_length_mismatch(self):
        data = load_test_data()
        high = np.asarray(data["high"][:64], dtype=np.float64)
        low = np.asarray(data["low"][:63], dtype=np.float64)
        close = np.asarray(data["close"][:64], dtype=np.float64)
        with pytest.raises(ValueError, match="Inconsistent slice lengths"):
            mp.ichimoku_oscillator(high, low, close)
