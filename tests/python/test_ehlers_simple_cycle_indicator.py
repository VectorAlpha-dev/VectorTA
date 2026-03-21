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


def manual_reference(data: np.ndarray, alpha: float = 0.07):
    cycle = np.full(data.shape[0], np.nan, dtype=np.float64)
    trigger = np.full(data.shape[0], np.nan, dtype=np.float64)
    smooth = np.full(data.shape[0], np.nan, dtype=np.float64)
    coeff_cycle = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha)
    coeff_prev1 = 2.0 * (1.0 - alpha)
    coeff_prev2 = (1.0 - alpha) * (1.0 - alpha)

    for i in range(data.shape[0]):
        if not np.isfinite(data[i]):
            continue
        if i >= 3 and np.isfinite(data[i - 1]) and np.isfinite(data[i - 2]) and np.isfinite(data[i - 3]):
            smooth[i] = (data[i] + 2.0 * data[i - 1] + 2.0 * data[i - 2] + data[i - 3]) / 6.0

        cycle_main = np.nan
        if (
            i >= 2
            and np.isfinite(smooth[i])
            and np.isfinite(smooth[i - 1])
            and np.isfinite(smooth[i - 2])
            and np.isfinite(cycle[i - 1])
            and np.isfinite(cycle[i - 2])
        ):
            cycle_main = (
                coeff_cycle * (smooth[i] - 2.0 * smooth[i - 1] + smooth[i - 2])
                + coeff_prev1 * cycle[i - 1]
                - coeff_prev2 * cycle[i - 2]
            )

        cycle_fallback = np.nan
        if i >= 2 and np.isfinite(data[i - 1]) and np.isfinite(data[i - 2]):
            cycle_fallback = (data[i] - 2.0 * data[i - 1] + data[i - 2]) / 4.0

        cycle[i] = cycle_fallback if i < 7 else cycle_main
        if i >= 1 and np.isfinite(cycle[i - 1]):
            trigger[i] = cycle[i - 1]

    return cycle, trigger


class TestEhlersSimpleCycleIndicator:
    def test_manual_reference_matches_binding(self):
        data = load_test_data()
        source = ((np.asarray(data["high"][:160]) + np.asarray(data["low"][:160])) * 0.5).astype(
            np.float64
        )
        expected_cycle, expected_trigger = manual_reference(source, 0.07)
        result = mp.ehlers_simple_cycle_indicator(source, alpha=0.07)
        np.testing.assert_allclose(result["cycle"], expected_cycle, rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            result["trigger"], expected_trigger, rtol=0.0, atol=1e-12, equal_nan=True
        )

    def test_stream_matches_batch(self):
        data = load_test_data()
        source = ((np.asarray(data["high"][:144]) + np.asarray(data["low"][:144])) * 0.5).astype(
            np.float64
        )
        batch = mp.ehlers_simple_cycle_indicator(source, alpha=0.07)
        stream = mp.EhlersSimpleCycleIndicatorStream(0.07)
        streamed = np.array([stream.update(float(x)) for x in source], dtype=np.float64)
        np.testing.assert_allclose(streamed[:, 0], batch["cycle"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            streamed[:, 1], batch["trigger"], rtol=0.0, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_and_first_row_match_single(self):
        data = load_test_data()
        source = ((np.asarray(data["high"][:128]) + np.asarray(data["low"][:128])) * 0.5).astype(
            np.float64
        )
        result = mp.ehlers_simple_cycle_indicator_batch(
            source,
            alpha_range=(0.07, 0.09, 0.02),
        )
        assert result["rows"] == 2
        assert result["cols"] == source.shape[0]
        np.testing.assert_allclose(result["alphas"], np.array([0.07, 0.09]), rtol=0.0, atol=1e-12)
        single = mp.ehlers_simple_cycle_indicator(source, alpha=0.07)
        np.testing.assert_allclose(
            result["cycle"][0], single["cycle"], rtol=0.0, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            result["trigger"][0], single["trigger"], rtol=0.0, atol=1e-12, equal_nan=True
        )

    def test_invalid_alpha(self):
        data = load_test_data()
        source = np.asarray(data["close"][:64], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid alpha"):
            mp.ehlers_simple_cycle_indicator(source, alpha=1.5)
