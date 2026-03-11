import numpy as np
import pytest

try:
    import my_project as ta_indicators
except ImportError:
    try:
        import vector_ta as ta_indicators
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )

from test_utils import load_test_data


def hlcc4_series(test_data):
    return (test_data["high"] + test_data["low"] + 2.0 * test_data["close"]) * 0.25


def naive_eudma(data, fast_length, slow_length, sample_length):
    data = np.asarray(data, dtype=float)
    fast = np.full(data.shape[0], np.nan)
    slow = np.full(data.shape[0], np.nan)
    finite = np.flatnonzero(np.isfinite(data))
    if finite.size == 0:
        return fast, slow

    first_valid = int(finite[0])
    sampled = np.full(data.shape[0], np.nan)
    for idx in range(data.shape[0]):
        if idx % sample_length == 0:
            sampled[idx] = data[idx]
        elif idx > 0 and np.isfinite(sampled[idx - 1]):
            sampled[idx] = sampled[idx - 1]
        else:
            sampled[idx] = 0.0

    def weights(length):
        idx = np.arange(1, length + 1, dtype=float)
        values = 1.0 - np.cos(2.0 * np.pi * idx / (length + 1.0))
        return values, values.sum()

    fast_weights, fast_norm = weights(fast_length)
    slow_weights, slow_norm = weights(slow_length)

    for idx in range(first_valid, data.shape[0]):
        fast_acc = 0.0
        for offset in range(fast_length):
            if idx >= offset:
                value = sampled[idx - offset]
                fast_acc += fast_weights[offset] * (value if np.isfinite(value) else 0.0)
        fast[idx] = fast_acc / fast_norm

        slow_acc = 0.0
        for offset in range(slow_length):
            if idx >= offset:
                value = sampled[idx - offset]
                slow_acc += slow_weights[offset] * (value if np.isfinite(value) else 0.0)
        slow[idx] = slow_acc / slow_norm

    return fast, slow


class TestEhlersUndersampledDoubleMovingAverage:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_basic_output_shape(self, test_data):
        data = hlcc4_series(test_data)
        fast, slow = ta_indicators.ehlers_undersampled_double_moving_average(data)

        assert len(fast) == len(data)
        assert len(slow) == len(data)
        assert np.isfinite(fast[0])
        assert np.isfinite(slow[0])
        assert np.any(np.isfinite(fast[16:]))
        assert np.any(np.isfinite(slow[16:]))

    def test_matches_naive_reference(self):
        data = np.array([np.nan, 10.0, 11.0, 12.0, 13.0, 12.5, 14.0, 15.0], dtype=float)
        fast, slow = ta_indicators.ehlers_undersampled_double_moving_average(
            data,
            fast_length=3,
            slow_length=4,
            sample_length=2,
        )
        expected_fast, expected_slow = naive_eudma(data, 3, 4, 2)
        np.testing.assert_allclose(fast, expected_fast, rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(slow, expected_slow, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_stream_matches_batch(self, test_data):
        data = hlcc4_series(test_data)
        batch_fast, batch_slow = ta_indicators.ehlers_undersampled_double_moving_average(
            data,
            fast_length=6,
            slow_length=12,
            sample_length=5,
        )
        stream = ta_indicators.EhlersUndersampledDoubleMovingAverageStream(
            fast_length=6,
            slow_length=12,
            sample_length=5,
        )

        stream_fast = np.full(len(data), np.nan)
        stream_slow = np.full(len(data), np.nan)
        for idx, value in enumerate(data):
            result = stream.update(float(value))
            if result is not None:
                stream_fast[idx], stream_slow[idx] = result

        np.testing.assert_allclose(batch_fast, stream_fast, rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(batch_slow, stream_slow, rtol=1e-12, atol=1e-12, equal_nan=True)
        stream.reset()
        assert stream.update(float("nan")) is None

    def test_batch_matches_single(self, test_data):
        data = hlcc4_series(test_data)
        single_fast, single_slow = ta_indicators.ehlers_undersampled_double_moving_average(
            data,
            fast_length=6,
            slow_length=12,
            sample_length=5,
        )
        batch = ta_indicators.ehlers_undersampled_double_moving_average_batch(
            data,
            fast_length_range=(6, 6, 0),
            slow_length_range=(12, 12, 0),
            sample_length_range=(5, 5, 0),
        )

        assert batch["fast_values"].shape == (1, len(data))
        assert batch["slow_values"].shape == (1, len(data))
        assert list(batch["fast_lengths"]) == [6]
        assert list(batch["slow_lengths"]) == [12]
        assert list(batch["sample_lengths"]) == [5]
        assert batch["rows"] == 1
        assert batch["cols"] == len(data)
        np.testing.assert_allclose(
            batch["fast_values"][0], single_fast, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            batch["slow_values"][0], single_slow, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_errors(self):
        with pytest.raises(ValueError, match="Invalid fast_length|invalid fast_length"):
            ta_indicators.ehlers_undersampled_double_moving_average(
                np.arange(20, dtype=float),
                fast_length=0,
                slow_length=12,
                sample_length=5,
            )

        with pytest.raises(ValueError, match="Invalid slow_length|invalid slow_length"):
            ta_indicators.ehlers_undersampled_double_moving_average(
                np.arange(20, dtype=float),
                fast_length=6,
                slow_length=0,
                sample_length=5,
            )

        with pytest.raises(ValueError, match="Invalid sample_length|invalid sample_length"):
            ta_indicators.ehlers_undersampled_double_moving_average(
                np.arange(20, dtype=float),
                fast_length=6,
                slow_length=12,
                sample_length=0,
            )

        with pytest.raises(ValueError, match="empty|Empty"):
            ta_indicators.ehlers_undersampled_double_moving_average(np.array([], dtype=float))

        with pytest.raises(ValueError, match="all values are NaN|All values are NaN"):
            ta_indicators.ehlers_undersampled_double_moving_average(np.full(32, np.nan))
