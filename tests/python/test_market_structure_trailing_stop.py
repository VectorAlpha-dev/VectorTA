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


class TestMarketStructureTrailingStop:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]

        trailing_stop, state, structure = mp.market_structure_trailing_stop(
            open_, high, low, close, 5, 100.0, "All"
        )

        assert len(trailing_stop) == len(close)
        assert len(state) == len(close)
        assert len(structure) == len(close)
        assert np.isfinite(trailing_stop).any()
        assert np.all(np.isin(state[np.isfinite(state)], np.array([-1.0, 0.0, 1.0])))
        assert np.all(np.isin(structure[np.isfinite(structure)], np.array([-1.0, 0.0, 1.0])))

    def test_kernel_parity(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]

        auto = mp.market_structure_trailing_stop(open_, high, low, close, 5, 100.0, "All")
        scalar = mp.market_structure_trailing_stop(
            open_, high, low, close, 5, 100.0, "All", kernel="scalar"
        )

        for lhs, rhs in zip(auto, scalar):
            np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_invalid_params(self, test_data):
        open_ = test_data["open"][:64]
        high = test_data["high"][:64]
        low = test_data["low"][:64]
        close = test_data["close"][:64]

        with pytest.raises(ValueError, match="Invalid length"):
            mp.market_structure_trailing_stop(open_, high, low, close, 0, 100.0, "CHoCH")

        with pytest.raises(ValueError, match="Invalid increment_factor"):
            mp.market_structure_trailing_stop(open_, high, low, close, 5, -1.0, "CHoCH")

        with pytest.raises(ValueError, match="Invalid reset_on"):
            mp.market_structure_trailing_stop(open_, high, low, close, 5, 100.0, "bad")

    def test_reset_mode_behavior(self, test_data):
        open_ = test_data["open"][:320]
        high = test_data["high"][:320]
        low = test_data["low"][:320]
        close = test_data["close"][:320]

        trailing_all, state_all, structure_all = mp.market_structure_trailing_stop(
            open_, high, low, close, 5, 100.0, "All"
        )
        trailing_choch, state_choch, structure_choch = mp.market_structure_trailing_stop(
            open_, high, low, close, 5, 100.0, "CHoCH"
        )

        assert not np.array_equal(trailing_all, trailing_choch)
        assert np.count_nonzero(np.nan_to_num(np.abs(structure_all))) >= np.count_nonzero(
            np.nan_to_num(np.abs(structure_choch))
        )
        assert len(state_all) == len(state_choch)

    def test_batch_single_param_matches_single(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]

        batch = mp.market_structure_trailing_stop_batch(
            open_,
            high,
            low,
            close,
            length_range=(5, 5, 0),
            increment_factor_range=(100.0, 100.0, 0.0),
            reset_on="All",
        )
        single = mp.market_structure_trailing_stop(open_, high, low, close, 5, 100.0, "All")

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(batch["lengths"], np.array([5], dtype=np.uint64))
        np.testing.assert_allclose(
            batch["increment_factors"], np.array([100.0]), rtol=0.0, atol=0.0
        )
        assert batch["reset_ons"] == ["All"]

        for key, expected in zip(("trailing_stop", "state", "structure"), single):
            np.testing.assert_allclose(
                batch[key][0], expected, rtol=1e-12, atol=1e-12, equal_nan=True
            )
