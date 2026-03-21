import numpy as np
import pytest

from rust_comparison import compare_with_rust
from test_utils import load_test_data

try:
    import vector_ta as ta
except ImportError:
    try:
        import vector_ta as ta
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


def _rma_update(state, count, total, period, value):
    if state is not None:
        return ((state * (period - 1.0)) + value) / period, count, total
    count += 1
    total += value
    if count == period:
        return total / period, count, total
    return None, count, total


def manual_volume_weighted_relative_strength_index(source, volume, rsi_length, range_length, ma_length, ma_type):
    source = np.asarray(source, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    n = source.shape[0]
    out = {
        "rsi": np.full(n, np.nan, dtype=np.float64),
        "consolidation_strength": np.full(n, np.nan, dtype=np.float64),
        "rsi_ma": np.full(n, np.nan, dtype=np.float64),
        "bearish_tp": np.full(n, np.nan, dtype=np.float64),
        "bullish_tp": np.full(n, np.nan, dtype=np.float64),
    }

    from collections import deque

    prev_source = None
    prev_rsi = None
    prev_dir = None
    valid_rsi_count = 0

    up_state = down_state = vol_state = None
    up_count = down_count = vol_count = 0
    up_sum = down_sum = vol_sum = 0.0

    ma_queue = deque()
    ma_sum = 0.0
    ma_pv_queue = deque()
    ma_pv_sum = 0.0
    ma_vol_queue = deque()
    ma_vol_sum = 0.0
    ema_state = None
    rma_ma_state = None
    rma_ma_count = 0
    rma_ma_sum = 0.0

    x_queue = deque()
    x_sum = 0.0
    short = max(1, range_length // 2)
    short_window = deque()
    long_window = deque()
    short_wma_half = deque()
    short_wma_full = deque()
    long_wma_half = deque()
    long_wma_full = deque()

    def wma(window, period, value):
        window.append(value)
        if len(window) > period:
            window.popleft()
        if len(window) < period:
            return None
        arr = np.asarray(window, dtype=np.float64)
        weights = np.arange(1.0, period + 1.0)
        return float(np.dot(arr, weights) / weights.sum())

    def hma(step_window, half_window, full_window, period, value):
        h = wma(half_window, max(1, period // 2), value)
        f = wma(full_window, period, value)
        if h is None or f is None:
            return None
        diff = 2.0 * h - f
        sqrt_len = max(1, int(np.sqrt(period)))
        return wma(step_window, sqrt_len, diff)

    for i in range(n):
        s = source[i]
        v = volume[i]
        if not np.isfinite(s) or not np.isfinite(v):
            prev_source = None
            prev_rsi = None
            prev_dir = None
            valid_rsi_count = 0
            up_state = down_state = vol_state = None
            up_count = down_count = vol_count = 0
            up_sum = down_sum = vol_sum = 0.0
            ma_queue.clear()
            ma_sum = 0.0
            ma_pv_queue.clear()
            ma_pv_sum = 0.0
            ma_vol_queue.clear()
            ma_vol_sum = 0.0
            ema_state = None
            rma_ma_state = None
            rma_ma_count = 0
            rma_ma_sum = 0.0
            x_queue.clear()
            x_sum = 0.0
            short_window.clear()
            long_window.clear()
            short_wma_half.clear()
            short_wma_full.clear()
            long_wma_half.clear()
            long_wma_full.clear()
            continue

        if prev_source is None:
            prev_source = s
            continue

        delta = s - prev_source
        prev_source = s
        up_state, up_count, up_sum = _rma_update(up_state, up_count, up_sum, rsi_length, max(delta, 0.0) * v)
        down_state, down_count, down_sum = _rma_update(down_state, down_count, down_sum, rsi_length, max(-delta, 0.0) * v)
        vol_state, vol_count, vol_sum = _rma_update(vol_state, vol_count, vol_sum, rsi_length, v)
        if up_state is None or down_state is None or vol_state is None or abs(vol_state) <= 1e-12:
            continue

        up = up_state / vol_state
        down = down_state / vol_state
        if abs(down) <= 1e-12:
            rsi = 100.0
        elif abs(up) <= 1e-12:
            rsi = 0.0
        else:
            rsi = 100.0 - (100.0 / (1.0 + up / down))
        out["rsi"][i] = rsi

        ma_type_norm = ma_type.lower()
        if ma_type_norm == "ema":
            ema_state = rsi if ema_state is None else (2.0 / (ma_length + 1.0)) * rsi + (1.0 - 2.0 / (ma_length + 1.0)) * ema_state
            out["rsi_ma"][i] = ema_state
        elif ma_type_norm == "sma":
            ma_queue.append(rsi)
            ma_sum += rsi
            if len(ma_queue) > ma_length:
                ma_sum -= ma_queue.popleft()
            if len(ma_queue) == ma_length:
                out["rsi_ma"][i] = ma_sum / ma_length
        elif ma_type_norm == "wma":
            value = wma(ma_queue, ma_length, rsi)
            if value is not None:
                out["rsi_ma"][i] = value
        elif ma_type_norm == "hma":
            value = hma(ma_queue, ma_pv_queue, ma_vol_queue, ma_length, rsi)
            if value is not None:
                out["rsi_ma"][i] = value
        elif ma_type_norm in ("smma (rma)", "smma", "rma"):
            rma_ma_state, rma_ma_count, rma_ma_sum = _rma_update(rma_ma_state, rma_ma_count, rma_ma_sum, ma_length, rsi)
            if rma_ma_state is not None:
                out["rsi_ma"][i] = rma_ma_state
        elif ma_type_norm == "vwma":
            ma_pv_queue.append(rsi * v)
            ma_vol_queue.append(v)
            ma_pv_sum += rsi * v
            ma_vol_sum += v
            if len(ma_pv_queue) > ma_length:
                ma_pv_sum -= ma_pv_queue.popleft()
                ma_vol_sum -= ma_vol_queue.popleft()
            if len(ma_pv_queue) == ma_length and abs(ma_vol_sum) > 1e-12:
                out["rsi_ma"][i] = ma_pv_sum / ma_vol_sum

        if prev_rsi is not None and prev_rsi >= 80.0 and rsi < 80.0:
            out["bearish_tp"][i] = 95.0
        if prev_rsi is not None and prev_rsi <= 20.0 and rsi > 20.0:
            out["bullish_tp"][i] = 5.0

        valid_rsi_count += 1
        direction = 1 if rsi > 50.0 else -1
        transition = 1.0 if prev_dir is not None and prev_dir + direction == 0 else 0.0
        prev_dir = direction
        prev_rsi = rsi

        x = transition / min(valid_rsi_count, range_length)
        x_queue.append(x)
        x_sum += x
        if len(x_queue) > range_length:
            x_sum -= x_queue.popleft()
        if len(x_queue) == range_length:
            p = x_sum / range_length
            short_hma = hma(short_window, short_wma_half, short_wma_full, short, p)
            if short_hma is not None:
                f = short_hma * (short * (5.0 / 3.0))
                long_hma = hma(long_window, long_wma_half, long_wma_full, range_length * 2, f)
                if long_hma is not None:
                    out["consolidation_strength"][i] = max(long_hma, 0.0)

    return out


class TestVolumeWeightedRelativeStrengthIndex:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_manual_reference_matches_binding(self, test_data):
        source = np.asarray(test_data["close"][:320], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:320], dtype=np.float64)
        params = {
            "rsi_length": 14,
            "range_length": 10,
            "ma_length": 14,
            "ma_type": "EMA",
        }
        result = ta.volume_weighted_relative_strength_index(source, volume, **params)
        expected = manual_volume_weighted_relative_strength_index(source, volume, **params)
        for key, values in expected.items():
            np.testing.assert_allclose(
                np.asarray(result[key], dtype=np.float64),
                values,
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )

    def test_matches_rust_reference_generator(self, test_data):
        source = np.asarray(test_data["close"], dtype=np.float64)
        volume = np.asarray(test_data["volume"], dtype=np.float64)
        params = {
            "rsi_length": 14,
            "range_length": 10,
            "ma_length": 14,
            "ma_type": "EMA",
        }
        result = ta.volume_weighted_relative_strength_index(source, volume, **params)
        compare_with_rust(
            "volume_weighted_relative_strength_index",
            result,
            "close_volume",
            params,
            rtol=0.0,
            atol=1e-12,
        )

    def test_stream_matches_batch(self, test_data):
        source = np.asarray(test_data["close"][:320], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:320], dtype=np.float64)
        params = {
            "rsi_length": 14,
            "range_length": 10,
            "ma_length": 14,
            "ma_type": "EMA",
        }
        batch = ta.volume_weighted_relative_strength_index(source, volume, **params)
        stream = ta.VolumeWeightedRelativeStrengthIndexStream(**params)
        keys = list(batch.keys())
        collected = {key: [] for key in keys}

        for s, v in zip(source, volume):
            point = stream.update(float(s), float(v))
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
        source = np.asarray(test_data["close"][:320], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:320], dtype=np.float64)
        result = ta.volume_weighted_relative_strength_index_batch(
            source,
            volume,
            rsi_length_range=(14, 16, 2),
            range_length_range=(10, 10, 0),
            ma_length_range=(14, 14, 0),
            ma_type="EMA",
        )

        assert result["rows"] == 2
        assert result["cols"] == source.shape[0]
        np.testing.assert_array_equal(result["rsi_lengths"], np.array([14, 16], dtype=np.uint64))
        np.testing.assert_array_equal(result["range_lengths"], np.array([10, 10], dtype=np.uint64))
        np.testing.assert_array_equal(result["ma_lengths"], np.array([14, 14], dtype=np.uint64))
        assert result["ma_types"] == ["EMA", "EMA"]

        single = ta.volume_weighted_relative_strength_index(
            source, volume, rsi_length=14, range_length=10, ma_length=14, ma_type="EMA"
        )
        for key in single.keys():
            np.testing.assert_allclose(
                np.asarray(result[key][0], dtype=np.float64),
                np.asarray(single[key], dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )

    def test_invalid_inputs_raise(self, test_data):
        source = np.asarray(test_data["close"][:128], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:128], dtype=np.float64)

        with pytest.raises(ValueError, match="invalid rsi_length|Invalid rsi_length"):
            ta.volume_weighted_relative_strength_index(source, volume, rsi_length=0)
        with pytest.raises(ValueError, match="invalid range_length|Invalid range_length"):
            ta.volume_weighted_relative_strength_index(source, volume, range_length=0)
        with pytest.raises(ValueError, match="invalid ma_type|Invalid ma_type"):
            ta.volume_weighted_relative_strength_index(source, volume, ma_type="BAD")
