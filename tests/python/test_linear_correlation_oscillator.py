import numpy as np
import pytest

try:
    import vector_ta as ta_indicators
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python' first",
        allow_module_level=True,
    )


def reference_lco(data, period):
    data = np.asarray(data, dtype=np.float64)
    out = np.full(data.shape, np.nan, dtype=np.float64)
    first_candidates = np.flatnonzero(~np.isnan(data))
    if len(first_candidates) == 0:
        raise ValueError("All values are NaN")
    first = int(first_candidates[0])
    if period == 0 or period > len(data):
        raise ValueError("Invalid period")
    if len(data) - first <= period + 1:
        raise ValueError("Not enough valid data")

    x = np.arange(1.0, period + 1.0, dtype=np.float64)
    var_x = (period * period - 1.0) / 12.0
    for end in range(first + period + 1, len(data)):
        window = data[end + 1 - period : end + 1]
        if np.isnan(window).any() or var_x <= 0.0:
            continue
        centered = np.dot(x, window) - ((period + 1.0) * 0.5) * window.sum()
        var_y = np.mean(window * window) - np.mean(window) ** 2
        if var_y <= 0.0:
            continue
        out[end] = np.clip((centered / period) / np.sqrt(var_y * var_x), -1.0, 1.0)
    return out


class TestLinearCorrelationOscillator:
    @pytest.fixture(scope="class")
    def sample_data(self):
        return np.array(
            [
                12.0,
                14.0,
                13.0,
                15.0,
                16.5,
                18.0,
                17.0,
                19.5,
                21.0,
                20.5,
                22.5,
                24.0,
                23.0,
                25.5,
                26.0,
                28.0,
            ],
            dtype=np.float64,
        )

    def test_matches_reference(self, sample_data):
        period = 5
        result = ta_indicators.linear_correlation_oscillator(sample_data, period=period)
        expected = reference_lco(sample_data, period)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_increasing_series_is_one(self):
        data = np.arange(0.0, 24.0, dtype=np.float64)
        result = ta_indicators.linear_correlation_oscillator(data, period=6)
        assert np.all(np.isnan(result[:7]))
        np.testing.assert_allclose(result[7:], 1.0, rtol=1e-12, atol=1e-12)

    def test_decreasing_series_is_negative_one(self):
        data = -np.arange(0.0, 24.0, dtype=np.float64)
        result = ta_indicators.linear_correlation_oscillator(data, period=6)
        assert np.all(np.isnan(result[:7]))
        np.testing.assert_allclose(result[7:], -1.0, rtol=1e-12, atol=1e-12)

    def test_constant_series_returns_nan(self):
        data = np.full(20, 5.0, dtype=np.float64)
        result = ta_indicators.linear_correlation_oscillator(data, period=5)
        assert np.all(np.isnan(result))

    def test_nan_prefix_matches_pine_semantics(self):
        data = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        result = ta_indicators.linear_correlation_oscillator(data, period=3)
        expected = reference_lco(data, 3)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
        assert np.all(np.isnan(result[:6]))
        assert result[6] == pytest.approx(1.0)

    def test_stream_matches_batch(self, sample_data):
        period = 5
        batch = ta_indicators.linear_correlation_oscillator(sample_data, period=period)
        stream = ta_indicators.LinearCorrelationOscillatorStream(period=period)
        out = []
        for value in sample_data:
            updated = stream.update(float(value))
            out.append(np.nan if updated is None else updated)
        streamed = np.array(out, dtype=np.float64)
        np.testing.assert_allclose(streamed, batch, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_batch_matches_single(self, sample_data):
        batch = ta_indicators.linear_correlation_oscillator_batch(
            sample_data,
            period_range=(5, 7, 1),
        )
        assert batch["values"].shape == (3, len(sample_data))
        assert list(batch["periods"]) == [5, 6, 7]
        for row, period in enumerate(batch["periods"]):
            single = ta_indicators.linear_correlation_oscillator(sample_data, period=int(period))
            np.testing.assert_allclose(
                batch["values"][row],
                single,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )

    def test_kernel_parameter_accepts_repo_values(self, sample_data):
        baseline = ta_indicators.linear_correlation_oscillator(sample_data, period=5, kernel="scalar")
        for kernel in [None, "auto", "scalar"]:
            result = ta_indicators.linear_correlation_oscillator(sample_data, period=5, kernel=kernel)
            np.testing.assert_allclose(result, baseline, rtol=1e-12, atol=1e-12, equal_nan=True)

        for kernel in ["avx2", "avx512"]:
            try:
                result = ta_indicators.linear_correlation_oscillator(
                    sample_data, period=5, kernel=kernel
                )
            except ValueError as exc:
                assert "not compiled" in str(exc).lower()
            else:
                np.testing.assert_allclose(
                    result, baseline, rtol=1e-12, atol=1e-12, equal_nan=True
                )

    def test_invalid_period_zero(self, sample_data):
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.linear_correlation_oscillator(sample_data, period=0)

    def test_not_enough_valid_data(self):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        with pytest.raises(ValueError, match="Not enough valid data|Invalid period"):
            ta_indicators.linear_correlation_oscillator(data, period=4)

    def test_all_nan_input(self):
        data = np.full(12, np.nan, dtype=np.float64)
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.linear_correlation_oscillator(data, period=5)
