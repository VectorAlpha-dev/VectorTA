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


class TestEmdTrend:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]

        direction, average, upper, lower = mp.emd_trend(
            open_, high, low, close, "close", "SMA", 28, 1.0
        )

        assert len(direction) == len(close)
        assert len(average) == len(close)
        assert len(upper) == len(close)
        assert len(lower) == len(close)
        assert np.isfinite(average).any()

    def test_invalid_params(self, test_data):
        open_ = test_data["open"][:96]
        high = test_data["high"][:96]
        low = test_data["low"][:96]
        close = test_data["close"][:96]

        with pytest.raises(ValueError, match="Invalid source"):
            mp.emd_trend(open_, high, low, close, "bad", "SMA", 28, 1.0)

        with pytest.raises(ValueError, match="Invalid avg_type"):
            mp.emd_trend(open_, high, low, close, "close", "bad", 28, 1.0)

        with pytest.raises(ValueError, match="Invalid length"):
            mp.emd_trend(open_, high, low, close, "close", "SMA", 0, 1.0)

        with pytest.raises(ValueError, match="Invalid mult"):
            mp.emd_trend(open_, high, low, close, "close", "SMA", 28, 0.0)

    def test_batch_single_param_matches_single(self, test_data):
        open_ = test_data["open"][:240]
        high = test_data["high"][:240]
        low = test_data["low"][:240]
        close = test_data["close"][:240]

        batch = mp.emd_trend_batch(
            open_,
            high,
            low,
            close,
            length_range=(28, 28, 0),
            mult_range=(1.0, 1.0, 0.0),
            source="close",
            avg_type="SMA",
        )
        single = mp.emd_trend(open_, high, low, close, "close", "SMA", 28, 1.0)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(batch["lengths"], np.array([28], dtype=np.uint64))
        np.testing.assert_allclose(batch["mults"], np.array([1.0]), rtol=0.0, atol=0.0)

        for key, expected in zip(("direction", "average", "upper", "lower"), single):
            np.testing.assert_allclose(
                batch[key][0], expected, rtol=1e-12, atol=1e-12, equal_nan=True
            )

    def test_stream_contract(self, test_data):
        open_ = test_data["open"][:220]
        high = test_data["high"][:220]
        low = test_data["low"][:220]
        close = test_data["close"][:220]

        stream = mp.EmdTrendStream("close", "SMA", 28, 1.0)

        points = []
        for o, h, l, c in zip(open_, high, low, close):
            points.append(stream.update(float(o), float(h), float(l), float(c)))

        assert len(points) == len(close)
        assert any(point is not None and np.isfinite(point[1]) for point in points)

    def test_batch_multi_param_contract(self, test_data):
        open_ = test_data["open"][:180]
        high = test_data["high"][:180]
        low = test_data["low"][:180]
        close = test_data["close"][:180]

        batch = mp.emd_trend_batch(
            open_,
            high,
            low,
            close,
            length_range=(28, 30, 2),
            mult_range=(1.0, 1.5, 0.5),
            source="hlc3",
            avg_type="EMA",
        )

        assert batch["rows"] == 4
        assert batch["cols"] == len(close)
        assert batch["direction"].shape == (4, len(close))
        assert batch["average"].shape == (4, len(close))
        assert batch["upper"].shape == (4, len(close))
        assert batch["lower"].shape == (4, len(close))
