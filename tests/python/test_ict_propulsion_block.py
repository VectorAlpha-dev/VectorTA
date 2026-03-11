import numpy as np
import pytest

from test_utils import load_test_data

try:
    import my_project as mp
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python' first",
        allow_module_level=True,
    )


class TestIctPropulsionBlock:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        open_ = np.asarray(test_data["open"][:320], dtype=np.float64)
        high = np.asarray(test_data["high"][:320], dtype=np.float64)
        low = np.asarray(test_data["low"][:320], dtype=np.float64)
        close = np.asarray(test_data["close"][:320], dtype=np.float64)

        result = mp.ict_propulsion_block(open_, high, low, close, 3, "close")
        assert len(result) == 12
        assert all(len(arr) == len(close) for arr in result)
        assert np.isfinite(result[2]).any() or np.isfinite(result[8]).any()

    def test_kernel_parity(self, test_data):
        open_ = np.asarray(test_data["open"][:256], dtype=np.float64)
        high = np.asarray(test_data["high"][:256], dtype=np.float64)
        low = np.asarray(test_data["low"][:256], dtype=np.float64)
        close = np.asarray(test_data["close"][:256], dtype=np.float64)

        auto = mp.ict_propulsion_block(open_, high, low, close, 3, "close")
        scalar = mp.ict_propulsion_block(
            open_, high, low, close, 3, "close", kernel="scalar"
        )

        for lhs, rhs in zip(auto, scalar):
            np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_invalid_params(self, test_data):
        open_ = np.asarray(test_data["open"][:128], dtype=np.float64)
        high = np.asarray(test_data["high"][:128], dtype=np.float64)
        low = np.asarray(test_data["low"][:128], dtype=np.float64)
        close = np.asarray(test_data["close"][:128], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid swing_length"):
            mp.ict_propulsion_block(open_, high, low, close, 0, "close")

        with pytest.raises(ValueError, match="Invalid mitigation_price"):
            mp.ict_propulsion_block(open_, high, low, close, 3, "bad")

    def test_stream_matches_batch(self, test_data):
        open_ = np.asarray(test_data["open"][:220], dtype=np.float64)
        high = np.asarray(test_data["high"][:220], dtype=np.float64)
        low = np.asarray(test_data["low"][:220], dtype=np.float64)
        close = np.asarray(test_data["close"][:220], dtype=np.float64)

        batch = mp.ict_propulsion_block(open_, high, low, close, 3, "close")
        stream = mp.IctPropulsionBlockStream(3, "close")

        streamed = [[] for _ in range(12)]
        for o, h, l, c in zip(open_, high, low, close):
            values = stream.update(float(o), float(h), float(l), float(c))
            assert values is not None
            for idx, value in enumerate(values):
                streamed[idx].append(value)

        for lhs, rhs in zip(streamed, batch):
            np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_batch_single_param_matches_single(self, test_data):
        open_ = np.asarray(test_data["open"][:200], dtype=np.float64)
        high = np.asarray(test_data["high"][:200], dtype=np.float64)
        low = np.asarray(test_data["low"][:200], dtype=np.float64)
        close = np.asarray(test_data["close"][:200], dtype=np.float64)

        result = mp.ict_propulsion_block_batch(
            open_,
            high,
            low,
            close,
            swing_length_range=(3, 3, 0),
            mitigation_price_toggle=(True, False),
        )
        single = mp.ict_propulsion_block(open_, high, low, close, 3, "close")

        assert result["rows"] == 1
        assert result["cols"] == len(close)
        assert result["bullish_high"].shape == (1, len(close))
        assert list(result["mitigation_prices"]) == ["close"]
        np.testing.assert_allclose(result["swing_lengths"], np.array([3], dtype=np.uint64))

        keys = [
            "bullish_high",
            "bullish_low",
            "bullish_kind",
            "bullish_active",
            "bullish_mitigated",
            "bullish_new",
            "bearish_high",
            "bearish_low",
            "bearish_kind",
            "bearish_active",
            "bearish_mitigated",
            "bearish_new",
        ]
        for idx, key in enumerate(keys):
            np.testing.assert_allclose(
                result[key][0], single[idx], rtol=1e-10, atol=1e-10, equal_nan=True
            )

    def test_batch_metadata(self, test_data):
        open_ = np.asarray(test_data["open"][:180], dtype=np.float64)
        high = np.asarray(test_data["high"][:180], dtype=np.float64)
        low = np.asarray(test_data["low"][:180], dtype=np.float64)
        close = np.asarray(test_data["close"][:180], dtype=np.float64)

        result = mp.ict_propulsion_block_batch(
            open_,
            high,
            low,
            close,
            swing_length_range=(3, 4, 1),
            mitigation_price_toggle=(True, True),
        )

        assert result["rows"] == 4
        assert result["cols"] == len(close)
        np.testing.assert_allclose(
            result["swing_lengths"], np.array([3, 3, 4, 4], dtype=np.uint64)
        )
        assert list(result["mitigation_prices"]) == ["close", "wick", "close", "wick"]
