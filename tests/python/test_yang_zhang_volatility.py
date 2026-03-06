"""
Python binding tests for Yang-Zhang volatility.
"""
import pytest
import numpy as np

from test_utils import load_test_data

try:
    import my_project as mp
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python' first",
        allow_module_level=True,
    )


class TestYangZhangVolatility:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        yz, rs = mp.yang_zhang_volatility(
            open_,
            high,
            low,
            close,
            lookback=14,
        )

        assert len(yz) == len(close)
        assert len(rs) == len(close)

        valid = np.flatnonzero(~np.isnan(yz))
        assert valid.size > 0
        first_valid = int(valid[0])
        assert first_valid >= 14

        tail_start = min(first_valid + 32, len(yz))
        assert not np.any(np.isnan(yz[tail_start:]))
        assert not np.any(np.isnan(rs[tail_start:]))

    def test_kernel_parity(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        yz_auto, rs_auto = mp.yang_zhang_volatility(
            open_,
            high,
            low,
            close,
            lookback=20,
            k_override=True,
            k=0.42,
        )
        yz_scalar, rs_scalar = mp.yang_zhang_volatility(
            open_,
            high,
            low,
            close,
            lookback=20,
            k_override=True,
            k=0.42,
            kernel="scalar",
        )

        np.testing.assert_allclose(
            yz_auto, yz_scalar, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            rs_auto, rs_scalar, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_length_mismatch(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        with pytest.raises(ValueError, match="length mismatch"):
            mp.yang_zhang_volatility(open_[:-1], high, low, close, lookback=14)

    def test_invalid_lookback(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        with pytest.raises(ValueError, match="Invalid lookback"):
            mp.yang_zhang_volatility(open_, high, low, close, lookback=0)

    def test_invalid_k(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        with pytest.raises(ValueError, match="Invalid k"):
            mp.yang_zhang_volatility(
                open_,
                high,
                low,
                close,
                lookback=14,
                k_override=True,
                k=1.5,
            )

    def test_streaming_matches_batch(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        batch_yz, batch_rs = mp.yang_zhang_volatility(
            open_,
            high,
            low,
            close,
            lookback=20,
            k_override=True,
            k=0.42,
        )

        stream = mp.YangZhangVolatilityStream(20, True, 0.42)
        stream_yz = []
        stream_rs = []

        for o, h, l, c in zip(open_, high, low, close):
            out = stream.update(o, h, l, c)
            if out is None:
                stream_yz.append(np.nan)
                stream_rs.append(np.nan)
            else:
                yz, rs = out
                stream_yz.append(yz)
                stream_rs.append(rs)

        np.testing.assert_allclose(
            stream_yz, batch_yz, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_rs, batch_rs, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        open_ = test_data["open"]
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        result = mp.yang_zhang_volatility_batch(
            open_,
            high,
            low,
            close,
            lookback_range=(20, 20, 0),
            k_override=True,
            k_range=(0.42, 0.42, 0.0),
        )

        assert "yz" in result
        assert "rs" in result
        assert "lookbacks" in result
        assert "k_overrides" in result
        assert "ks" in result
        assert result["rows"] == 1
        assert result["cols"] == len(close)
        assert result["yz"].shape == (1, len(close))
        assert result["rs"].shape == (1, len(close))

        single_yz, single_rs = mp.yang_zhang_volatility(
            open_,
            high,
            low,
            close,
            lookback=20,
            k_override=True,
            k=0.42,
        )

        np.testing.assert_allclose(
            result["yz"][0], single_yz, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["rs"][0], single_rs, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_array_equal(result["lookbacks"], np.array([20], dtype=np.uint64))
        np.testing.assert_array_equal(result["k_overrides"], np.array([True]))
        np.testing.assert_allclose(result["ks"], np.array([0.42]), rtol=0.0, atol=1e-12)

    def test_batch_multiple_lookbacks_metadata(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]

        result = mp.yang_zhang_volatility_batch(
            open_,
            high,
            low,
            close,
            lookback_range=(10, 14, 2),
            k_override=True,
            k_range=(0.42, 0.42, 0.0),
        )

        assert result["rows"] == 3
        assert result["cols"] == len(close)
        assert result["yz"].shape == (3, len(close))
        assert result["rs"].shape == (3, len(close))
        np.testing.assert_array_equal(result["lookbacks"], np.array([10, 12, 14], dtype=np.uint64))
        np.testing.assert_array_equal(result["k_overrides"], np.array([True, True, True]))
        np.testing.assert_allclose(
            result["ks"], np.array([0.42, 0.42, 0.42]), rtol=0.0, atol=1e-12
        )

        single_yz, single_rs = mp.yang_zhang_volatility(
            open_,
            high,
            low,
            close,
            lookback=10,
            k_override=True,
            k=0.42,
        )
        np.testing.assert_allclose(
            result["yz"][0], single_yz, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["rs"][0], single_rs, rtol=1e-10, atol=1e-10, equal_nan=True
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
