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


class TestZigZagChannels:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]

        middle, upper, lower = mp.zig_zag_channels(open_, high, low, close, 12, True)

        assert len(middle) == len(close)
        assert len(upper) == len(close)
        assert len(lower) == len(close)

        finite = np.flatnonzero(np.isfinite(middle))
        assert finite.size > 0

        mask = np.isfinite(middle) & np.isfinite(upper) & np.isfinite(lower)
        assert np.all(upper[mask] >= middle[mask])
        assert np.all(middle[mask] >= lower[mask])

    def test_kernel_parity(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        auto = mp.zig_zag_channels(open_, high, low, close, 10, True)
        scalar = mp.zig_zag_channels(open_, high, low, close, 10, True, kernel="scalar")

        for lhs, rhs in zip(auto, scalar):
            np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_invalid_params(self, test_data):
        open_ = test_data["open"][:64]
        high = test_data["high"][:64]
        low = test_data["low"][:64]
        close = test_data["close"][:64]

        with pytest.raises(ValueError, match="Invalid length"):
            mp.zig_zag_channels(open_, high, low, close, 0, True)

        with pytest.raises(ValueError, match="Not enough valid data"):
            mp.zig_zag_channels(open_, high, low, close, len(close), True)

    def test_extend_behavior(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]

        mid_true, up_true, lo_true = mp.zig_zag_channels(open_, high, low, close, 12, True)
        mid_false, up_false, lo_false = mp.zig_zag_channels(open_, high, low, close, 12, False)

        assert np.isfinite(mid_true).sum() >= np.isfinite(mid_false).sum()
        assert np.isfinite(up_true).sum() >= np.isfinite(up_false).sum()
        assert np.isfinite(lo_true).sum() >= np.isfinite(lo_false).sum()

    def test_batch_single_param_matches_single(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]

        batch = mp.zig_zag_channels_batch(
            open_,
            high,
            low,
            close,
            length_range=(12, 12, 0),
            extend=True,
        )
        single = mp.zig_zag_channels(open_, high, low, close, 12, True)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(batch["lengths"], np.array([12], dtype=np.uint64))
        assert batch["extends"] == [True]

        for key, expected in zip(("middle", "upper", "lower"), single):
            np.testing.assert_allclose(
                batch[key][0], expected, rtol=1e-12, atol=1e-12, equal_nan=True
            )

    def test_batch_metadata_multiple_lengths(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]

        batch = mp.zig_zag_channels_batch(
            open_,
            high,
            low,
            close,
            length_range=(8, 12, 2),
            extend=False,
        )

        assert batch["rows"] == 3
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(
            batch["lengths"], np.array([8, 10, 12], dtype=np.uint64)
        )
        assert batch["extends"] == [False, False, False]
