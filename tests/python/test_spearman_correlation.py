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


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        current = values[order[start]]
        while end < values.shape[0] and values[order[end]] == current:
            end += 1
        avg_rank = (start + 1 + end) * 0.5
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def manual_reference(
    main: np.ndarray,
    compare: np.ndarray,
    lookback: int = 30,
    smoothing_length: int = 3,
):
    n = main.shape[0]
    raw = np.full(n, np.nan, dtype=np.float64)
    smoothed = np.full(n, np.nan, dtype=np.float64)

    main_returns = np.full(n, np.nan, dtype=np.float64)
    compare_returns = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if (
            np.isfinite(main[i - 1])
            and np.isfinite(main[i])
            and np.isfinite(compare[i - 1])
            and np.isfinite(compare[i])
        ):
            main_returns[i] = main[i] - main[i - 1]
            compare_returns[i] = compare[i] - compare[i - 1]

    for i in range(lookback, n):
        main_window = main_returns[i + 1 - lookback : i + 1]
        compare_window = compare_returns[i + 1 - lookback : i + 1]
        if not np.all(np.isfinite(main_window)) or not np.all(np.isfinite(compare_window)):
            continue
        ranks_main = average_ranks(main_window)
        ranks_compare = average_ranks(compare_window)
        raw[i] = np.corrcoef(ranks_main, ranks_compare)[0, 1]

    for i in range(smoothing_length - 1, n):
        window = raw[i + 1 - smoothing_length : i + 1]
        if np.all(np.isfinite(window)):
            smoothed[i] = np.mean(window)

    return {"raw": raw, "smoothed": smoothed}


class TestSpearmanCorrelation:
    def test_manual_reference_matches_binding(self):
        data = load_test_data()
        main = np.asarray(data["close"][:180], dtype=np.float64)
        compare = np.asarray(data["open"][:180], dtype=np.float64)
        expected = manual_reference(main, compare, 30, 3)
        result = mp.spearman_correlation(main, compare, lookback=30, smoothing_length=3)
        np.testing.assert_allclose(result["raw"], expected["raw"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            result["smoothed"], expected["smoothed"], rtol=0.0, atol=1e-12, equal_nan=True
        )

    def test_stream_matches_batch(self):
        data = load_test_data()
        main = np.asarray(data["close"][:180], dtype=np.float64)
        compare = np.asarray(data["open"][:180], dtype=np.float64)
        batch = mp.spearman_correlation(main, compare)
        stream = mp.SpearmanCorrelationStream(30, 3)
        last = None
        for main_value, compare_value in zip(main, compare):
            last = stream.update(float(main_value), float(compare_value))
        assert last is not None
        assert np.isclose(last["raw"], batch["raw"][-1], equal_nan=True)
        assert np.isclose(last["smoothed"], batch["smoothed"][-1], equal_nan=True)

    def test_batch_metadata_and_first_row_match_single(self):
        data = load_test_data()
        main = np.asarray(data["close"][:160], dtype=np.float64)
        compare = np.asarray(data["open"][:160], dtype=np.float64)
        result = mp.spearman_correlation_batch(
            main,
            compare,
            lookback_range=(30, 32, 2),
            smoothing_length_range=(3, 3, 0),
        )
        assert result["rows"] == 2
        assert result["cols"] == main.shape[0]
        np.testing.assert_array_equal(result["lookback"], np.array([30, 32], dtype=np.int64))
        single = mp.spearman_correlation(main, compare, lookback=30, smoothing_length=3)
        np.testing.assert_allclose(result["raw"][0], single["raw"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            result["smoothed"][0], single["smoothed"], rtol=0.0, atol=1e-12, equal_nan=True
        )

    def test_kernel_parity(self):
        data = load_test_data()
        main = np.asarray(data["close"][:160], dtype=np.float64)
        compare = np.asarray(data["open"][:160], dtype=np.float64)
        auto = mp.spearman_correlation(main, compare)
        scalar = mp.spearman_correlation(main, compare, kernel="scalar")
        np.testing.assert_allclose(auto["raw"], scalar["raw"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            auto["smoothed"], scalar["smoothed"], rtol=0.0, atol=1e-12, equal_nan=True
        )

    def test_invalid_lookback(self):
        data = load_test_data()
        main = np.asarray(data["close"][:64], dtype=np.float64)
        compare = np.asarray(data["open"][:64], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid lookback"):
            mp.spearman_correlation(main, compare, lookback=0)
