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


def manual_qqe_weighted_oscillator(data, length, factor, smooth, weight):
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    rsi = np.full(n, np.nan, dtype=np.float64)
    trailing_stop = np.full(n, np.nan, dtype=np.float64)

    valid = np.flatnonzero(np.isfinite(data))
    if valid.size == 0:
        return {"rsi": rsi, "trailing_stop": trailing_stop}
    if valid.size < length + 1:
        return {"rsi": rsi, "trailing_stop": trailing_stop}

    alpha = 2.0 / (float(smooth) + 1.0)
    prev_src = data[int(valid[0])]
    prev_rsi = np.nan
    prev_ts = np.nan

    num_count = 0
    den_count = 0
    diff_count = 0
    num_sum = 0.0
    den_sum = 0.0
    diff_sum = 0.0
    num_state = np.nan
    den_state = np.nan
    diff_state = np.nan
    ema_state = np.nan

    for i in range(int(valid[0]) + 1, n):
        current = data[i]
        if not np.isfinite(current):
            prev_src = np.nan
            continue
        if not np.isfinite(prev_src):
            prev_src = current
            continue

        delta = current - prev_src
        w = weight if np.isfinite(prev_rsi) and np.isfinite(prev_ts) and delta * (prev_rsi - prev_ts) > 0.0 else 1.0
        weighted_delta = delta * w
        abs_weighted_delta = abs(weighted_delta)

        num_count += 1
        if num_count < length:
            num_sum += weighted_delta
        elif num_count == length:
            num_sum += weighted_delta
            num_state = num_sum / float(length)
        else:
            num_state = ((num_state * (length - 1.0)) + weighted_delta) / float(length)

        den_count += 1
        if den_count < length:
            den_sum += abs_weighted_delta
        elif den_count == length:
            den_sum += abs_weighted_delta
            den_state = den_sum / float(length)
        else:
            den_state = ((den_state * (length - 1.0)) + abs_weighted_delta) / float(length)

        if np.isfinite(num_state) and np.isfinite(den_state) and den_state != 0.0:
            ratio = num_state / den_state
            ema_state = ratio if not np.isfinite(ema_state) else alpha * ratio + (1.0 - alpha) * ema_state
            curr_rsi = 50.0 * ema_state + 50.0
            rsi[i] = curr_rsi

            if np.isfinite(prev_rsi):
                diff_input = abs(curr_rsi - prev_rsi)
                diff_count += 1
                if diff_count < length:
                    diff_sum += diff_input
                    diff = np.nan
                elif diff_count == length:
                    diff_sum += diff_input
                    diff_state = diff_sum / float(length)
                    diff = diff_state
                else:
                    diff_state = ((diff_state * (length - 1.0)) + diff_input) / float(length)
                    diff = diff_state
            else:
                diff = np.nan

            if np.isfinite(diff):
                crossover = np.isfinite(prev_ts) and curr_rsi > prev_ts and prev_rsi <= prev_ts
                crossunder = np.isfinite(prev_ts) and curr_rsi < prev_ts and prev_rsi >= prev_ts
                if crossover:
                    curr_ts = curr_rsi - diff * factor
                elif crossunder:
                    curr_ts = curr_rsi + diff * factor
                elif np.isfinite(prev_ts):
                    if curr_rsi > prev_ts:
                        curr_ts = max(curr_rsi - diff * factor, prev_ts)
                    else:
                        curr_ts = min(curr_rsi + diff * factor, prev_ts)
                else:
                    curr_ts = curr_rsi
            else:
                curr_ts = curr_rsi

            trailing_stop[i] = curr_ts
            prev_rsi = curr_rsi
            prev_ts = curr_ts

        prev_src = current

    return {"rsi": rsi, "trailing_stop": trailing_stop}


class TestQqeWeightedOscillator:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_manual_reference_matches_binding(self, test_data):
        close = np.asarray(test_data["close"][:160], dtype=np.float64)
        params = {"length": 14, "factor": 4.236, "smooth": 5, "weight": 2.4}
        result = ta.qqe_weighted_oscillator(close, **params)
        expected = manual_qqe_weighted_oscillator(close, **params)

        np.testing.assert_allclose(
            np.asarray(result["rsi"], dtype=np.float64),
            expected["rsi"],
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            np.asarray(result["trailing_stop"], dtype=np.float64),
            expected["trailing_stop"],
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )

    def test_matches_rust_reference_generator(self, test_data):
        close = np.asarray(test_data["close"], dtype=np.float64)
        params = {"length": 14, "factor": 4.236, "smooth": 5, "weight": 2.0}
        result = ta.qqe_weighted_oscillator(close, **params)
        compare_with_rust(
            "qqe_weighted_oscillator",
            result,
            "close",
            params,
            rtol=0.0,
            atol=1e-12,
        )

    def test_stream_matches_batch(self, test_data):
        close = np.asarray(test_data["close"][:144], dtype=np.float64)
        params = {"length": 14, "factor": 4.236, "smooth": 5, "weight": 2.0}
        batch = ta.qqe_weighted_oscillator(close, **params)
        stream = ta.QqeWeightedOscillatorStream(**params)

        out_rsi = []
        out_ts = []
        for value in close:
            point = stream.update(float(value))
            if point is None:
                out_rsi.append(np.nan)
                out_ts.append(np.nan)
            else:
                out_rsi.append(point[0])
                out_ts.append(point[1])

        np.testing.assert_allclose(
            np.asarray(out_rsi, dtype=np.float64),
            np.asarray(batch["rsi"], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            np.asarray(out_ts, dtype=np.float64),
            np.asarray(batch["trailing_stop"], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )

    def test_batch_metadata_and_first_row_match_single(self, test_data):
        close = np.asarray(test_data["close"][:128], dtype=np.float64)
        result = ta.qqe_weighted_oscillator_batch(
            close,
            length_range=(14, 16, 2),
            factor_range=(4.236, 4.736, 0.5),
            smooth_range=(5, 5, 0),
            weight_range=(2.0, 2.0, 0.0),
        )

        assert result["rows"] == 4
        assert result["cols"] == close.shape[0]
        np.testing.assert_array_equal(result["lengths"], np.array([14, 14, 16, 16], dtype=np.uint64))
        np.testing.assert_allclose(result["factors"], np.array([4.236, 4.736, 4.236, 4.736]))
        np.testing.assert_array_equal(result["smooths"], np.array([5, 5, 5, 5], dtype=np.uint64))
        np.testing.assert_allclose(result["weights"], np.array([2.0, 2.0, 2.0, 2.0]))

        single = ta.qqe_weighted_oscillator(close, length=14, factor=4.236, smooth=5, weight=2.0)
        np.testing.assert_allclose(
            np.asarray(result["rsi"][0], dtype=np.float64),
            np.asarray(single["rsi"], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            np.asarray(result["trailing_stop"][0], dtype=np.float64),
            np.asarray(single["trailing_stop"], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )

    def test_invalid_inputs_raise(self, test_data):
        close = np.asarray(test_data["close"][:64], dtype=np.float64)

        with pytest.raises(ValueError, match="invalid length|Invalid length"):
            ta.qqe_weighted_oscillator(close, length=0)
        with pytest.raises(ValueError, match="invalid factor|Invalid factor"):
            ta.qqe_weighted_oscillator(close, factor=-1.0)
        with pytest.raises(ValueError, match="invalid smooth|Invalid smooth"):
            ta.qqe_weighted_oscillator(close, smooth=0)
