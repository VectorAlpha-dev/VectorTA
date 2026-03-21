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


def manual_reference(data, min_range=10, max_range=100, step=5, signal_line=7):
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    lengths = list(range(min_range, max_range + 1, step))
    out = {
        "value": np.full(n, np.nan, dtype=np.float64),
        "signal": np.full(n, np.nan, dtype=np.float64),
        "bullish_reversal": np.full(n, np.nan, dtype=np.float64),
        "bearish_reversal": np.full(n, np.nan, dtype=np.float64),
    }

    for idx in range(max_range - 1, n):
        slopes = []
        valid = True
        for length in lengths:
            start = idx + 1 - length
            window = data[start : idx + 1]
            if not np.all(np.isfinite(window)) or np.any(window <= 0):
                valid = False
                break
            y = np.log(window)
            x = np.arange(1.0, length + 1.0, dtype=np.float64)
            sum_x = x.sum()
            sum_y = y.sum()
            sum_x_sqr = np.dot(x, x)
            sum_xy = np.dot(x, y)
            slopes.append((length * sum_xy - sum_x * sum_y) / (length * sum_x_sqr - sum_x * sum_x))
        if valid:
            out["value"][idx] = float(np.mean(slopes))

    queue = []
    running_sum = 0.0
    for idx in range(n):
        value = out["value"][idx]
        if np.isfinite(value):
            queue.append(float(value))
            running_sum += value
            if len(queue) > signal_line:
                running_sum -= queue.pop(0)
        if len(queue) == signal_line:
            out["signal"][idx] = running_sum / signal_line
        if np.isfinite(value) and np.isfinite(out["signal"][idx]):
            prev_value = out["value"][idx - 1] if idx > 0 else np.nan
            prev_signal = out["signal"][idx - 1] if idx > 0 else np.nan
            out["bullish_reversal"][idx] = (
                1.0
                if np.isfinite(prev_value)
                and np.isfinite(prev_signal)
                and value > out["signal"][idx]
                and prev_value <= prev_signal
                and value < 0.0
                else 0.0
            )
            out["bearish_reversal"][idx] = (
                1.0
                if np.isfinite(prev_value)
                and np.isfinite(prev_signal)
                and value < out["signal"][idx]
                and prev_value >= prev_signal
                and value > 0.0
                else 0.0
            )

    return out


class TestRegressionSlopeOscillator:
    def test_manual_reference_matches_binding(self):
        data = np.asarray(load_test_data()["close"][:220], dtype=np.float64)
        expected = manual_reference(data, 10, 30, 5, 7)
        result = mp.regression_slope_oscillator(data, 10, 30, 5, 7)
        for key in expected:
            np.testing.assert_allclose(result[key], expected[key], rtol=0.0, atol=1e-10, equal_nan=True)

    def test_stream_matches_batch(self):
        data = np.asarray(load_test_data()["close"][:220], dtype=np.float64)
        batch = mp.regression_slope_oscillator(data, 10, 40, 5, 6)
        stream = mp.RegressionSlopeOscillatorStream(10, 40, 5, 6)
        streamed = np.array([stream.update(float(v)) for v in data], dtype=np.float64)
        np.testing.assert_allclose(streamed[:, 0], batch["value"], rtol=0.0, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(streamed[:, 1], batch["signal"], rtol=0.0, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(streamed[:, 2], batch["bullish_reversal"], rtol=0.0, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(streamed[:, 3], batch["bearish_reversal"], rtol=0.0, atol=1e-10, equal_nan=True)

    def test_batch_metadata_and_first_row_match_single(self):
        data = np.asarray(load_test_data()["close"][:180], dtype=np.float64)
        result = mp.regression_slope_oscillator_batch(
            data,
            min_range_range=(10, 15, 5),
            max_range_range=(30, 30, 0),
            step_range=(5, 5, 0),
            signal_line_range=(7, 7, 0),
        )
        single = mp.regression_slope_oscillator(data, 10, 30, 5, 7)
        assert result["rows"] == 2
        assert result["cols"] == data.shape[0]
        np.testing.assert_allclose(result["min_ranges"], np.array([10, 15]), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(result["max_ranges"], np.array([30, 30]), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(result["steps"], np.array([5, 5]), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(result["signal_lines"], np.array([7, 7]), rtol=0.0, atol=0.0)
        for key in ["value", "signal", "bullish_reversal", "bearish_reversal"]:
            np.testing.assert_allclose(result[key][0], single[key], rtol=0.0, atol=1e-10, equal_nan=True)

    def test_invalid_range_config(self):
        data = np.asarray(load_test_data()["close"][:96], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid range config"):
            mp.regression_slope_oscillator(data, min_range=20, max_range=10, step=5, signal_line=7)
