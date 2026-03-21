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


def manual_forward_backward_exponential_oscillator(data, length, smooth):
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    forward_backward = np.full(n, np.nan, dtype=np.float64)
    backward = np.full(n, np.nan, dtype=np.float64)
    histogram = np.full(n, np.nan, dtype=np.float64)

    alpha = 2.0 / (float(smooth) + 1.0)
    ema1_state = np.nan
    ema2_state = np.nan
    prev_ema2 = np.nan
    ema1_window = []
    diffs = []

    for i in range(n):
        value = data[i]
        if not np.isfinite(value):
            ema1_state = np.nan
            ema2_state = np.nan
            prev_ema2 = np.nan
            ema1_window.clear()
            diffs.clear()
            continue

        ema1_state = value if not np.isfinite(ema1_state) else alpha * value + (1.0 - alpha) * ema1_state
        ema1_window.append(ema1_state)
        if len(ema1_window) > length:
            ema1_window.pop(0)

        if len(ema1_window) == length:
            ema2 = ema1_window[-1]
            prev = ema2
            num = 0.0
            den = 0.0
            for hist_value in reversed(ema1_window[:-1]):
                ema2 += alpha * (hist_value - ema2)
                dt = prev - ema2
                num += dt
                den += abs(dt)
                prev = ema2
            if den != 0.0:
                forward_backward[i] = num / den * 50.0 + 50.0

        ema2_state = ema1_state if not np.isfinite(ema2_state) else alpha * ema1_state + (1.0 - alpha) * ema2_state
        if np.isfinite(prev_ema2):
            diffs.append(ema2_state - prev_ema2)
            if len(diffs) > length:
                diffs.pop(0)
            if len(diffs) == length:
                den = float(np.sum(np.abs(diffs)))
                if den != 0.0:
                    backward[i] = float(np.sum(diffs)) / den * 50.0 + 50.0
                    if np.isfinite(forward_backward[i]):
                        histogram[i] = (forward_backward[i] - backward[i]) / 4.0 + 50.0
        prev_ema2 = ema2_state

    return {
        "forward_backward": forward_backward,
        "backward": backward,
        "histogram": histogram,
    }


class TestForwardBackwardExponentialOscillator:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_manual_reference_matches_binding(self, test_data):
        close = np.asarray(test_data["close"][:180], dtype=np.float64)
        params = {"length": 20, "smooth": 10}
        result = ta.forward_backward_exponential_oscillator(close, **params)
        expected = manual_forward_backward_exponential_oscillator(close, **params)

        np.testing.assert_allclose(
            np.asarray(result["forward_backward"], dtype=np.float64),
            expected["forward_backward"],
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            np.asarray(result["backward"], dtype=np.float64),
            expected["backward"],
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            np.asarray(result["histogram"], dtype=np.float64),
            expected["histogram"],
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )

    def test_matches_rust_reference_generator(self, test_data):
        close = np.asarray(test_data["close"], dtype=np.float64)
        params = {"length": 20, "smooth": 10}
        result = ta.forward_backward_exponential_oscillator(close, **params)
        compare_with_rust(
            "forward_backward_exponential_oscillator",
            result,
            "close",
            params,
            rtol=0.0,
            atol=1e-12,
        )

    def test_stream_matches_batch(self, test_data):
        close = np.asarray(test_data["close"][:144], dtype=np.float64)
        params = {"length": 20, "smooth": 10}
        batch = ta.forward_backward_exponential_oscillator(close, **params)
        stream = ta.ForwardBackwardExponentialOscillatorStream(**params)

        out_fb = []
        out_bw = []
        out_hist = []
        for value in close:
            point = stream.update(float(value))
            if point is None:
                out_fb.append(np.nan)
                out_bw.append(np.nan)
                out_hist.append(np.nan)
            else:
                out_fb.append(point[0])
                out_bw.append(point[1])
                out_hist.append(point[2])

        np.testing.assert_allclose(
            np.asarray(out_fb, dtype=np.float64),
            np.asarray(batch["forward_backward"], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            np.asarray(out_bw, dtype=np.float64),
            np.asarray(batch["backward"], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            np.asarray(out_hist, dtype=np.float64),
            np.asarray(batch["histogram"], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )

    def test_batch_metadata_and_first_row_match_single(self, test_data):
        close = np.asarray(test_data["close"][:128], dtype=np.float64)
        result = ta.forward_backward_exponential_oscillator_batch(
            close,
            length_range=(20, 22, 2),
            smooth_range=(10, 12, 2),
        )

        assert result["rows"] == 4
        assert result["cols"] == close.shape[0]
        np.testing.assert_array_equal(result["lengths"], np.array([20, 20, 22, 22], dtype=np.uint64))
        np.testing.assert_array_equal(result["smooths"], np.array([10, 12, 10, 12], dtype=np.uint64))

        single = ta.forward_backward_exponential_oscillator(close, length=20, smooth=10)
        np.testing.assert_allclose(
            np.asarray(result["forward_backward"][0], dtype=np.float64),
            np.asarray(single["forward_backward"], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            np.asarray(result["backward"][0], dtype=np.float64),
            np.asarray(single["backward"], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            np.asarray(result["histogram"][0], dtype=np.float64),
            np.asarray(single["histogram"], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )

    def test_invalid_inputs_raise(self, test_data):
        close = np.asarray(test_data["close"][:64], dtype=np.float64)

        with pytest.raises(ValueError, match="invalid length|Invalid length"):
            ta.forward_backward_exponential_oscillator(close, length=0)
        with pytest.raises(ValueError, match="invalid smooth|Invalid smooth"):
            ta.forward_backward_exponential_oscillator(close, smooth=0)
