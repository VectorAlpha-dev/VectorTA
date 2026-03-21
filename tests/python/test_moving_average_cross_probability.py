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


def sma(data, period):
    out = np.full(data.shape[0], np.nan, dtype=np.float64)
    total = 0.0
    for i, value in enumerate(data):
        total += value
        if i + 1 >= period:
            if i + 1 > period:
                total -= data[i - period]
            out[i] = total / period
    return out


def ema(data, period):
    out = np.full(data.shape[0], np.nan, dtype=np.float64)
    alpha = 2.0 / (period + 1.0)
    seed = 0.0
    current = np.nan
    for i, value in enumerate(data):
        seed += value
        if i + 1 < period:
            continue
        if i + 1 == period:
            current = seed / period
        else:
            current = alpha * value + (1.0 - alpha) * current
        out[i] = current
    return out


def wma_window(values):
    weights = np.arange(1.0, values.shape[0] + 1.0, dtype=np.float64)
    return np.dot(values, weights) / weights.sum()


def hma(data, period):
    n = data.shape[0]
    half = period // 2
    sqrt_len = int(np.sqrt(period))
    diff = np.full(n, np.nan, dtype=np.float64)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if i + 1 >= period and i + 1 >= half:
            diff[i] = 2.0 * wma_window(data[i + 1 - half : i + 1]) - wma_window(data[i + 1 - period : i + 1])
        if i + 1 >= period + sqrt_len - 1:
            window = diff[i + 1 - sqrt_len : i + 1]
            if np.all(np.isfinite(window)):
                out[i] = wma_window(window)
    return out


def stddev4(data, period):
    out = np.full(data.shape[0], np.nan, dtype=np.float64)
    for i in range(period - 1, data.shape[0]):
        window = data[i + 1 - period : i + 1]
        mean = np.mean(window)
        out[i] = np.sqrt(np.mean((window - mean) ** 2)) * 4.0
    return out


def array_ema(temp, period):
    alpha = 2.0 / (period + 1.0)
    current = temp[-1]
    for idx in range(temp.shape[0] - 2, -1, -1):
        current = alpha * temp[idx] + (1.0 - alpha) * current
    return current


def array_sma(temp, period):
    return np.mean(temp[:period])


def manual_reference(
    data,
    ma_type="ema",
    smoothing_window=7,
    slow_length=30,
    fast_length=14,
    resolution=50,
):
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    slow_ma = ema(data, slow_length) if ma_type == "ema" else sma(data, slow_length)
    fast_ma = ema(data, fast_length) if ma_type == "ema" else sma(data, fast_length)
    price = hma(data, smoothing_window)
    stdev = stddev4(data, smoothing_window)
    value = np.full(n, np.nan, dtype=np.float64)
    forecast = np.full(n, np.nan, dtype=np.float64)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    direction = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if np.isfinite(slow_ma[i]) and np.isfinite(fast_ma[i]):
            direction[i] = -1.0 if fast_ma[i] > slow_ma[i] else 1.0
        if i > 0 and np.isfinite(price[i]) and np.isfinite(price[i - 1]) and np.isfinite(stdev[i]):
            forecast[i] = price[i] + (price[i] - price[i - 1])
            upper[i] = forecast[i] + stdev[i]
            lower[i] = forecast[i] - stdev[i]
        if i < 2 * slow_length or not (np.isfinite(direction[i]) and np.isfinite(upper[i]) and np.isfinite(lower[i])):
            continue
        memory = np.array([data[i - j] for j in range(2 * slow_length + 1)], dtype=np.float64)
        step = (upper[i] - lower[i]) / (resolution - 1.0)
        hits = 0
        for k in range(resolution):
            possibility = lower[i] + step * k
            temp = np.concatenate(([possibility], memory))
            slow_future = array_ema(temp, slow_length) if ma_type == "ema" else array_sma(temp, slow_length)
            fast_future = array_ema(temp, fast_length) if ma_type == "ema" else array_sma(temp, fast_length)
            crossed = slow_future > fast_future if direction[i] < 0 else slow_future <= fast_future
            hits += 1 if crossed else 0
        value[i] = 100.0 * hits / resolution

    return {
        "value": value,
        "slow_ma": slow_ma,
        "fast_ma": fast_ma,
        "forecast": forecast,
        "upper": upper,
        "lower": lower,
        "direction": direction,
    }


class TestMovingAverageCrossProbability:
    def test_manual_reference_matches_binding(self):
        data = np.asarray(load_test_data()["close"][:220], dtype=np.float64)
        expected = manual_reference(data)
        result = mp.moving_average_cross_probability(data)
        for key in expected:
            np.testing.assert_allclose(result[key], expected[key], rtol=0.0, atol=1e-8, equal_nan=True)

    def test_stream_matches_batch(self):
        data = np.asarray(load_test_data()["close"][:200], dtype=np.float64)
        batch = mp.moving_average_cross_probability(
            data,
            ma_type="sma",
            smoothing_window=8,
            slow_length=26,
            fast_length=11,
            resolution=40,
        )
        stream = mp.MovingAverageCrossProbabilityStream(
            ma_type="sma",
            smoothing_window=8,
            slow_length=26,
            fast_length=11,
            resolution=40,
        )
        streamed = np.array([stream.update(float(v)) for v in data], dtype=np.float64)
        keys = ["value", "slow_ma", "fast_ma", "forecast", "upper", "lower", "direction"]
        for idx, key in enumerate(keys):
            np.testing.assert_allclose(streamed[:, idx], batch[key], rtol=0.0, atol=1e-8, equal_nan=True)

    def test_batch_metadata_and_first_row_match_single(self):
        data = np.asarray(load_test_data()["close"][:180], dtype=np.float64)
        result = mp.moving_average_cross_probability_batch(
            data,
            smoothing_window_range=(7, 8, 1),
            slow_length_range=(30, 30, 0),
            fast_length_range=(14, 14, 0),
            resolution_range=(50, 50, 0),
        )
        single = mp.moving_average_cross_probability(data)
        assert result["rows"] == 2
        assert result["cols"] == data.shape[0]
        np.testing.assert_allclose(result["smoothing_windows"], np.array([7, 8]), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(result["slow_lengths"], np.array([30, 30]), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(result["fast_lengths"], np.array([14, 14]), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(result["resolutions"], np.array([50, 50]), rtol=0.0, atol=0.0)
        for key in ["value", "slow_ma", "fast_ma", "forecast", "upper", "lower", "direction"]:
            np.testing.assert_allclose(result[key][0], single[key], rtol=0.0, atol=1e-8, equal_nan=True)

    def test_invalid_length_order(self):
        data = np.asarray(load_test_data()["close"][:96], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid length order"):
            mp.moving_average_cross_probability(data, slow_length=10, fast_length=14)
