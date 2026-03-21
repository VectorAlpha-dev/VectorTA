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


class TestDirectionalImbalanceIndex:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = test_data["high"][:256]
        low = test_data["low"][:256]

        up, down, bulls, bears, upper, lower = mp.directional_imbalance_index(high, low)

        assert len(up) == len(high)
        assert len(down) == len(high)
        assert len(bulls) == len(high)
        assert len(bears) == len(high)
        assert len(upper) == len(high)
        assert len(lower) == len(high)

        assert up[0] == 1.0
        assert down[0] == 1.0
        assert bulls[0] == 50.0
        assert bears[0] == 50.0
        assert upper[0] == high[0]
        assert lower[0] == low[0]
        assert np.isfinite(up).all()
        assert np.isfinite(down).all()

    def test_invalid_params(self, test_data):
        high = test_data["high"][:128]
        low = test_data["low"][:128]

        with pytest.raises(ValueError, match="Invalid length"):
            mp.directional_imbalance_index(high, low, length=0)

        with pytest.raises(ValueError, match="Invalid period"):
            mp.directional_imbalance_index(high, low, period=0)

    def test_stream_matches_batch(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        batch = mp.directional_imbalance_index(high, low, length=7, period=20)

        stream = mp.DirectionalImbalanceIndexStream(7, 20)
        up = []
        down = []
        bulls = []
        bears = []
        upper = []
        lower = []
        for h, l in zip(high, low):
            out = stream.update(float(h), float(l))
            if out is None:
                up.append(np.nan)
                down.append(np.nan)
                bulls.append(np.nan)
                bears.append(np.nan)
                upper.append(np.nan)
                lower.append(np.nan)
            else:
                u, d, bu, be, hi, lo = out
                up.append(u)
                down.append(d)
                bulls.append(bu)
                bears.append(be)
                upper.append(hi)
                lower.append(lo)

        np.testing.assert_allclose(up, batch[0], rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(down, batch[1], rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(bulls, batch[2], rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(bears, batch[3], rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(upper, batch[4], rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(lower, batch[5], rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_batch_single_param_matches_single(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        batch = mp.directional_imbalance_index_batch(
            high,
            low,
            length_range=(10, 10, 0),
            period_range=(70, 70, 0),
        )
        single = mp.directional_imbalance_index(high, low, length=10, period=70)

        assert batch["rows"] == 1
        assert batch["cols"] == len(high)
        np.testing.assert_allclose(batch["up"][0], single[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(batch["down"][0], single[1], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(batch["bulls"][0], single[2], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(batch["bears"][0], single[3], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(batch["upper"][0], single[4], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(batch["lower"][0], single[5], rtol=1e-12, atol=1e-12)

    def test_batch_metadata_multiple_ranges(self, test_data):
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        batch = mp.directional_imbalance_index_batch(
            high,
            low,
            length_range=(6, 10, 2),
            period_range=(14, 14, 0),
        )

        assert batch["rows"] == 3
        assert batch["cols"] == len(high)
        np.testing.assert_array_equal(
            batch["lengths"], np.array([6, 8, 10], dtype=np.uint64)
        )
        np.testing.assert_array_equal(
            batch["periods"], np.array([14, 14, 14], dtype=np.uint64)
        )
