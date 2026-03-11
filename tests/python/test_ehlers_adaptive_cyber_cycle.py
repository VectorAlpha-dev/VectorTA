import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    try:
        import my_project as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


def median3(a: float, b: float, c: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
        return np.nan
    return (a + b + c) - min(a, b, c) - max(a, b, c)


def manual_reference(data: np.ndarray, alpha: float = 0.07):
    length = data.shape[0]
    smooth = np.full(length, np.nan, dtype=np.float64)
    cycle = np.full(length, np.nan, dtype=np.float64)
    q1 = np.full(length, np.nan, dtype=np.float64)
    i1 = np.zeros(length, dtype=np.float64)
    dp = np.full(length, np.nan, dtype=np.float64)
    ip = np.full(length, np.nan, dtype=np.float64)
    p = np.full(length, np.nan, dtype=np.float64)
    adaptive = np.full(length, np.nan, dtype=np.float64)
    trigger = np.full(length, np.nan, dtype=np.float64)

    cycle_coef = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha)
    cycle_prev1 = 2.0 * (1.0 - alpha)
    cycle_prev2 = (1.0 - alpha) * (1.0 - alpha)

    for idx in range(length):
        if not np.isfinite(data[idx]):
            continue

        if idx >= 3 and np.all(np.isfinite(data[idx - 3 : idx + 1])):
            smooth[idx] = (data[idx] + 2.0 * data[idx - 1] + 2.0 * data[idx - 2] + data[idx - 3]) / 6.0

        cycle_fallback = np.nan
        if idx >= 2 and np.isfinite(data[idx - 1]) and np.isfinite(data[idx - 2]):
            cycle_fallback = (data[idx] - 2.0 * data[idx - 1] + data[idx - 2]) / 4.0

        cycle_main = np.nan
        if (
            idx >= 2
            and np.isfinite(smooth[idx])
            and np.isfinite(smooth[idx - 1])
            and np.isfinite(smooth[idx - 2])
            and np.isfinite(cycle[idx - 1])
            and np.isfinite(cycle[idx - 2])
        ):
            cycle_main = (
                cycle_coef * (smooth[idx] - 2.0 * smooth[idx - 1] + smooth[idx - 2])
                + cycle_prev1 * cycle[idx - 1]
                - cycle_prev2 * cycle[idx - 2]
            )

        cycle[idx] = cycle_fallback if idx < 7 else cycle_main

        if np.isfinite(cycle[idx]):
            c2 = cycle[idx - 2] if idx >= 2 and np.isfinite(cycle[idx - 2]) else 0.0
            c4 = cycle[idx - 4] if idx >= 4 and np.isfinite(cycle[idx - 4]) else 0.0
            c6 = cycle[idx - 6] if idx >= 6 and np.isfinite(cycle[idx - 6]) else 0.0
            ip1 = ip[idx - 1] if idx >= 1 and np.isfinite(ip[idx - 1]) else 0.0
            q1[idx] = (0.0962 * cycle[idx] + 0.5769 * c2 - 0.5769 * c4 - 0.0962 * c6) * (
                0.5 + 0.08 * ip1
            )

        if idx >= 3 and np.isfinite(cycle[idx - 3]):
            i1[idx] = cycle[idx - 3]

        prev_q1 = q1[idx - 1] if idx >= 1 else np.nan
        prev_i1 = i1[idx - 1] if idx >= 1 else 0.0
        if np.isfinite(q1[idx]) and np.isfinite(prev_q1) and q1[idx] != 0.0 and prev_q1 != 0.0:
            numer = (i1[idx] / q1[idx]) - (prev_i1 / prev_q1)
            denom = 1.0 + i1[idx] * prev_i1 / (q1[idx] * prev_q1)
            dp_raw = numer / denom
        else:
            dp_raw = 0.0
        dp[idx] = min(1.1, max(0.1, dp_raw))

        md_inner = median3(dp[idx - 2], dp[idx - 3], dp[idx - 4]) if idx >= 4 else np.nan
        md = median3(dp[idx], dp[idx - 1], md_inner) if idx >= 1 else np.nan
        dc = 15.0 if md == 0.0 else (2.0 * np.pi / md) + 0.5
        ip[idx] = 0.33 * dc + 0.67 * (ip[idx - 1] if idx >= 1 and np.isfinite(ip[idx - 1]) else 0.0)
        p[idx] = 0.15 * ip[idx] + 0.85 * (p[idx - 1] if idx >= 1 and np.isfinite(p[idx - 1]) else 0.0)

        adaptive_main = np.nan
        if (
            idx >= 2
            and np.isfinite(smooth[idx])
            and np.isfinite(smooth[idx - 1])
            and np.isfinite(smooth[idx - 2])
            and np.isfinite(adaptive[idx - 1])
            and np.isfinite(adaptive[idx - 2])
        ):
            a1 = 2.0 / (p[idx] + 1.0)
            adapt_coef = (1.0 - 0.5 * a1) * (1.0 - 0.5 * a1)
            one_minus_a1 = 1.0 - a1
            adaptive_main = (
                adapt_coef * (smooth[idx] - 2.0 * smooth[idx - 1] + smooth[idx - 2])
                + 2.0 * one_minus_a1 * adaptive[idx - 1]
                - one_minus_a1 * one_minus_a1 * adaptive[idx - 2]
            )

        adaptive[idx] = cycle_fallback if idx < 7 else (adaptive_main if np.isfinite(adaptive_main) else cycle_fallback)
        if idx >= 1 and np.isfinite(adaptive[idx - 1]):
            trigger[idx] = adaptive[idx - 1]

    return adaptive, trigger


class TestEhlersAdaptiveCyberCycle:
    def test_manual_reference_matches_binding(self):
        data = load_test_data()
        source = ((np.asarray(data["high"][:192]) + np.asarray(data["low"][:192])) * 0.5).astype(
            np.float64
        )
        expected_cycle, expected_trigger = manual_reference(source, 0.07)
        result = mp.ehlers_adaptive_cyber_cycle(source, alpha=0.07)
        np.testing.assert_allclose(result["cycle"], expected_cycle, rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            result["trigger"], expected_trigger, rtol=0.0, atol=1e-12, equal_nan=True
        )

    def test_stream_matches_batch(self):
        data = load_test_data()
        source = ((np.asarray(data["high"][:160]) + np.asarray(data["low"][:160])) * 0.5).astype(
            np.float64
        )
        batch = mp.ehlers_adaptive_cyber_cycle(source, alpha=0.07)
        stream = mp.EhlersAdaptiveCyberCycleStream(0.07)
        streamed = np.array([stream.update(float(x)) for x in source], dtype=np.float64)
        np.testing.assert_allclose(streamed[:, 0], batch["cycle"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            streamed[:, 1], batch["trigger"], rtol=0.0, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_and_first_row_match_single(self):
        data = load_test_data()
        source = ((np.asarray(data["high"][:144]) + np.asarray(data["low"][:144])) * 0.5).astype(
            np.float64
        )
        result = mp.ehlers_adaptive_cyber_cycle_batch(
            source,
            alpha_range=(0.07, 0.09, 0.02),
        )
        assert result["rows"] == 2
        assert result["cols"] == source.shape[0]
        np.testing.assert_allclose(result["alphas"], np.array([0.07, 0.09]), rtol=0.0, atol=1e-12)
        single = mp.ehlers_adaptive_cyber_cycle(source, alpha=0.07)
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
            mp.ehlers_adaptive_cyber_cycle(source, alpha=1.5)
