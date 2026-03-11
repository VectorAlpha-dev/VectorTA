import numpy as np
import pytest

from rust_comparison import compare_with_rust
from test_utils import load_test_data

try:
    import vector_ta as ta
except ImportError:
    try:
        import my_project as ta
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


def manual_range_oscillator(high, low, close, length, mult):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = close.shape[0]
    out = {
        "oscillator": np.full(n, np.nan, dtype=np.float64),
        "ma": np.full(n, np.nan, dtype=np.float64),
        "upper_band": np.full(n, np.nan, dtype=np.float64),
        "lower_band": np.full(n, np.nan, dtype=np.float64),
        "range_width": np.full(n, np.nan, dtype=np.float64),
        "in_range": np.full(n, np.nan, dtype=np.float64),
        "trend": np.full(n, np.nan, dtype=np.float64),
        "break_up": np.full(n, np.nan, dtype=np.float64),
        "break_down": np.full(n, np.nan, dtype=np.float64),
    }

    atr200 = None
    atr2000 = None
    count200 = 0
    count2000 = 0
    sum200 = 0.0
    sum2000 = 0.0
    prev_close_200 = np.nan
    prev_close_2000 = np.nan
    closes = []
    trend = 0.0

    for i in range(n):
        h = high[i]
        l = low[i]
        c = close[i]
        if not np.isfinite(h) or not np.isfinite(l) or not np.isfinite(c):
            atr200 = None
            atr2000 = None
            count200 = 0
            count2000 = 0
            sum200 = 0.0
            sum2000 = 0.0
            prev_close_200 = np.nan
            prev_close_2000 = np.nan
            closes = []
            trend = 0.0
            continue

        tr200 = h - l if not np.isfinite(prev_close_200) else max(h - l, abs(h - prev_close_200), abs(l - prev_close_200))
        prev_close_200 = c
        if atr200 is None:
            count200 += 1
            sum200 += tr200
            if count200 == 200:
                atr200 = sum200 / 200.0
        else:
            atr200 = ((atr200 * 199.0) + tr200) / 200.0

        tr2000 = h - l if not np.isfinite(prev_close_2000) else max(h - l, abs(h - prev_close_2000), abs(l - prev_close_2000))
        prev_close_2000 = c
        if atr2000 is None:
            count2000 += 1
            sum2000 += tr2000
            if count2000 == 2000:
                atr2000 = sum2000 / 2000.0
        else:
            atr2000 = ((atr2000 * 1999.0) + tr2000) / 2000.0

        closes.append(c)
        if len(closes) > length + 1:
            closes.pop(0)

        atr_raw = atr2000 if atr2000 is not None else atr200
        if atr_raw is None or len(closes) < length + 1:
            continue

        sum_weighted = 0.0
        sum_weights = 0.0
        for j in range(length):
            curr = closes[-1 - j]
            prev = closes[-2 - j]
            if abs(prev) <= 1e-12:
                continue
            w = abs(curr - prev) / prev
            sum_weighted += curr * w
            sum_weights += w
        if abs(sum_weights) <= 1e-12:
            continue

        ma = sum_weighted / sum_weights
        width = atr_raw * mult
        max_dist = 0.0
        for j in range(length):
            max_dist = max(max_dist, abs(closes[-1 - j] - ma))

        if c > ma:
            trend = 1.0
        elif c < ma:
            trend = -1.0

        upper = ma + width
        lower = ma - width
        osc = np.nan if abs(width) <= 1e-12 else 100.0 * (c - ma) / width

        out["oscillator"][i] = osc
        out["ma"][i] = ma
        out["upper_band"][i] = upper
        out["lower_band"][i] = lower
        out["range_width"][i] = width
        out["in_range"][i] = 1.0 if max_dist <= width else 0.0
        out["trend"][i] = trend
        out["break_up"][i] = 1.0 if c > upper else 0.0
        out["break_down"][i] = 1.0 if c < lower else 0.0

    return out


class TestRangeOscillator:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_manual_reference_matches_binding(self, test_data):
        high = np.asarray(test_data["high"][:320], dtype=np.float64)
        low = np.asarray(test_data["low"][:320], dtype=np.float64)
        close = np.asarray(test_data["close"][:320], dtype=np.float64)
        params = {"length": 50, "mult": 2.0}
        result = ta.range_oscillator(high, low, close, **params)
        expected = manual_range_oscillator(high, low, close, **params)

        for key, values in expected.items():
            np.testing.assert_allclose(
                np.asarray(result[key], dtype=np.float64),
                values,
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )

    def test_matches_rust_reference_generator(self, test_data):
        high = np.asarray(test_data["high"], dtype=np.float64)
        low = np.asarray(test_data["low"], dtype=np.float64)
        close = np.asarray(test_data["close"], dtype=np.float64)
        params = {"length": 50, "mult": 2.0}
        result = ta.range_oscillator(high, low, close, **params)
        compare_with_rust("range_oscillator", result, "ohlc", params, rtol=0.0, atol=1e-12)

    def test_stream_matches_batch(self, test_data):
        high = np.asarray(test_data["high"][:320], dtype=np.float64)
        low = np.asarray(test_data["low"][:320], dtype=np.float64)
        close = np.asarray(test_data["close"][:320], dtype=np.float64)
        params = {"length": 50, "mult": 2.0}
        batch = ta.range_oscillator(high, low, close, **params)
        stream = ta.RangeOscillatorStream(**params)
        keys = list(batch.keys())
        collected = {key: [] for key in keys}

        for h, l, c in zip(high, low, close):
            point = stream.update(float(h), float(l), float(c))
            if point is None:
                for key in keys:
                    collected[key].append(np.nan)
            else:
                for idx, key in enumerate(keys):
                    collected[key].append(point[idx])

        for key in keys:
            np.testing.assert_allclose(
                np.asarray(collected[key], dtype=np.float64),
                np.asarray(batch[key], dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )

    def test_batch_metadata_and_first_row_match_single(self, test_data):
        high = np.asarray(test_data["high"][:320], dtype=np.float64)
        low = np.asarray(test_data["low"][:320], dtype=np.float64)
        close = np.asarray(test_data["close"][:320], dtype=np.float64)
        result = ta.range_oscillator_batch(
            high,
            low,
            close,
            length_range=(50, 52, 2),
            mult_range=(2.0, 2.5, 0.5),
        )

        assert result["rows"] == 4
        assert result["cols"] == close.shape[0]
        np.testing.assert_array_equal(result["lengths"], np.array([50, 50, 52, 52], dtype=np.uint64))
        np.testing.assert_allclose(result["mults"], np.array([2.0, 2.5, 2.0, 2.5], dtype=np.float64))

        single = ta.range_oscillator(high, low, close, length=50, mult=2.0)
        for key in single.keys():
            np.testing.assert_allclose(
                np.asarray(result[key][0], dtype=np.float64),
                np.asarray(single[key], dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )

    def test_invalid_inputs_raise(self, test_data):
        high = np.asarray(test_data["high"][:128], dtype=np.float64)
        low = np.asarray(test_data["low"][:128], dtype=np.float64)
        close = np.asarray(test_data["close"][:128], dtype=np.float64)

        with pytest.raises(ValueError, match="invalid length|Invalid length"):
            ta.range_oscillator(high, low, close, length=0)
        with pytest.raises(ValueError, match="invalid mult|Invalid mult"):
            ta.range_oscillator(high, low, close, mult=0.0)
