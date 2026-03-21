import numpy as np
import pytest

try:
    import vector_ta as ta_indicators
except ImportError:
    try:
        import vector_ta as ta_indicators
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )

from test_utils import load_test_data, assert_close


def naive_adaptive_momentum_oscillator(data, length, smoothing_length):
    data = np.asarray(data, dtype=float)
    amo = np.full(data.shape[0], np.nan)
    ama = np.full(data.shape[0], np.nan)
    raw = np.full(data.shape[0], np.nan)

    for idx, value in enumerate(data):
        if not np.isfinite(value) or idx < length:
            continue
        best_abs = -1.0
        best_delta = np.nan
        valid = True
        for lag in range(1, length + 1):
            past = data[idx - lag]
            if not np.isfinite(past):
                valid = False
                break
            delta = value - past
            abs_delta = abs(delta)
            if abs_delta >= best_abs:
                best_abs = abs_delta
                best_delta = delta
        if valid:
            raw[idx] = best_delta

    pf = float(smoothing_length)
    x_sum = pf * (pf + 1.0) * 0.5
    x2_sum = pf * (pf + 1.0) * (2.0 * pf + 1.0) / 6.0
    denom = pf * x2_sum - x_sum * x_sum
    for idx in range(smoothing_length - 1, data.shape[0]):
        window = raw[idx + 1 - smoothing_length : idx + 1]
        if np.all(np.isfinite(window)):
            y_sum = float(np.sum(window))
            xy_sum = float(np.dot(window, np.arange(1, smoothing_length + 1, dtype=float)))
            b = (pf * xy_sum - x_sum * y_sum) / denom
            a = (y_sum - b * x_sum) / pf
            amo[idx] = a + b * pf

    ama_state = 0.0
    prev = np.nan
    have_prev = False
    change_ring = np.zeros(length, dtype=float)
    head = 0
    count = 0
    change_sum = 0.0
    for idx, current in enumerate(amo):
        change = abs(current - prev) if have_prev and np.isfinite(current) and np.isfinite(prev) else 0.0
        if count < length:
            change_ring[head] = change
            change_sum += change
            count += 1
        else:
            old = change_ring[head]
            change_ring[head] = change
            change_sum += change - old
        head = (head + 1) % length

        if np.isfinite(current):
            if change_sum > 0.0:
                efficiency_ratio = abs(current) / change_sum
                delta = efficiency_ratio * (current - ama_state)
                if np.isfinite(delta):
                    ama_state += delta
            ama[idx] = ama_state

        prev = current
        have_prev = True

    return amo, ama


class TestAdaptiveMomentumOscillator:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_basic_output_shape(self, test_data):
        result = ta_indicators.adaptive_momentum_oscillator(test_data["close"])
        assert len(result) == 2
        for series in result:
            assert len(series) == len(test_data["close"])
            assert np.any(np.isfinite(series[32:]))

    def test_matches_naive_reference(self):
        data = np.array([10.0, 10.4, 10.2, 10.9, 11.3, 11.0, 11.8, 11.5, 12.0, 11.7, 12.4, 12.1], dtype=float)
        actual = ta_indicators.adaptive_momentum_oscillator(data, length=3, smoothing_length=2)
        expected = naive_adaptive_momentum_oscillator(data, 3, 2)
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-9, atol=1e-9, equal_nan=True)

    def test_stream_matches_single(self, test_data):
        expected = ta_indicators.adaptive_momentum_oscillator(
            test_data["close"], length=14, smoothing_length=9
        )
        stream = ta_indicators.AdaptiveMomentumOscillatorStream(length=14, smoothing_length=9)

        amo = np.full(len(test_data["close"]), np.nan)
        ama = np.full(len(test_data["close"]), np.nan)
        for idx, value in enumerate(test_data["close"]):
            result = stream.update(float(value))
            if result is not None:
                amo[idx] = result[0]
                ama[idx] = result[1]

        np.testing.assert_allclose(amo, expected[0], rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(ama, expected[1], rtol=1e-12, atol=1e-12, equal_nan=True)

        stream.reset()
        assert stream.update(float("nan")) is None

    def test_batch_matches_single(self, test_data):
        single = ta_indicators.adaptive_momentum_oscillator(
            test_data["close"], length=14, smoothing_length=9
        )
        batch = ta_indicators.adaptive_momentum_oscillator_batch(
            test_data["close"],
            length_range=(14, 14, 0),
            smoothing_length_range=(9, 9, 0),
        )

        assert batch["rows"] == 1
        assert batch["cols"] == len(test_data["close"])
        assert list(batch["lengths"]) == [14]
        assert list(batch["smoothing_lengths"]) == [9]
        assert batch["amo"].shape == (1, len(test_data["close"]))
        assert batch["ama"].shape == (1, len(test_data["close"]))
        assert_close(batch["amo"][0], single[0], rtol=1e-12, atol=1e-12)
        assert_close(batch["ama"][0], single[1], rtol=1e-12, atol=1e-12)

    def test_errors(self):
        sample = np.arange(8, dtype=float) + 10.0
        with pytest.raises(ValueError, match="invalid length|Invalid length"):
            ta_indicators.adaptive_momentum_oscillator(sample, length=0, smoothing_length=2)
        with pytest.raises(ValueError, match="invalid smoothing_length|Invalid smoothing_length"):
            ta_indicators.adaptive_momentum_oscillator(sample, length=3, smoothing_length=0)
        with pytest.raises(ValueError, match="empty|Empty"):
            ta_indicators.adaptive_momentum_oscillator(np.array([], dtype=float))
        with pytest.raises(ValueError, match="all values are NaN|All values are NaN"):
            ta_indicators.adaptive_momentum_oscillator(np.full(16, np.nan), length=3, smoothing_length=2)
        with pytest.raises(ValueError, match="not enough valid data|Not enough valid data"):
            ta_indicators.adaptive_momentum_oscillator(np.array([1.0, 2.0, 3.0], dtype=float), length=2, smoothing_length=2)
