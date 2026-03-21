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


class TestCyberpunkValueTrendAnalyzer:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]

        outputs = mp.cyberpunk_value_trend_analyzer(open_, high, low, close, 30, 75)
        assert len(outputs) == 6
        for out in outputs:
            assert len(out) == len(close)
        assert np.isfinite(outputs[0]).any()

    def test_invalid_params(self, test_data):
        open_ = test_data["open"][:128]
        high = test_data["high"][:128]
        low = test_data["low"][:128]
        close = test_data["close"][:128]

        with pytest.raises(ValueError, match="Invalid entry_level"):
            mp.cyberpunk_value_trend_analyzer(open_, high, low, close, 0, 75)

        with pytest.raises(ValueError, match="Invalid exit_level"):
            mp.cyberpunk_value_trend_analyzer(open_, high, low, close, 30, 101)

    def test_batch_single_param_matches_single(self, test_data):
        open_ = test_data["open"][:220]
        high = test_data["high"][:220]
        low = test_data["low"][:220]
        close = test_data["close"][:220]

        batch = mp.cyberpunk_value_trend_analyzer_batch(
            open_,
            high,
            low,
            close,
            entry_level_range=(30, 30, 0),
            exit_level_range=(75, 75, 0),
        )
        single = mp.cyberpunk_value_trend_analyzer(open_, high, low, close, 30, 75)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(batch["entry_levels"], np.array([30], dtype=np.uint64))
        np.testing.assert_array_equal(batch["exit_levels"], np.array([75], dtype=np.uint64))

        for key, expected in zip(
            (
                "value_trend",
                "value_trend_lag",
                "deviation_index",
                "overbought_signal",
                "buy_signal",
                "sell_signal",
            ),
            single,
        ):
            np.testing.assert_allclose(
                batch[key][0], expected, rtol=1e-12, atol=1e-12, equal_nan=True
            )

    def test_stream_matches_single(self, test_data):
        open_ = test_data["open"][:200]
        high = test_data["high"][:200]
        low = test_data["low"][:200]
        close = test_data["close"][:200]

        single = mp.cyberpunk_value_trend_analyzer(open_, high, low, close, 30, 75)
        stream = mp.CyberpunkValueTrendAnalyzerStream(30, 75)

        streamed = [[] for _ in range(6)]
        for o, h, l, c in zip(open_, high, low, close):
            point = stream.update(float(o), float(h), float(l), float(c))
            assert point is not None
            for idx, value in enumerate(point):
                streamed[idx].append(value)

        for expected, actual in zip(single, streamed):
            np.testing.assert_allclose(
                np.asarray(actual), expected, rtol=1e-12, atol=1e-12, equal_nan=True
            )

    def test_batch_multi_param_contract(self, test_data):
        open_ = test_data["open"][:180]
        high = test_data["high"][:180]
        low = test_data["low"][:180]
        close = test_data["close"][:180]

        batch = mp.cyberpunk_value_trend_analyzer_batch(
            open_,
            high,
            low,
            close,
            entry_level_range=(25, 35, 10),
            exit_level_range=(70, 80, 10),
        )

        assert batch["rows"] == 4
        assert batch["cols"] == len(close)
        assert batch["value_trend"].shape == (4, len(close))
        assert batch["sell_signal"].shape == (4, len(close))
