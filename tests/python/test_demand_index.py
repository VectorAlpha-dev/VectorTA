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


class TestDemandIndex:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]
        volume = test_data["volume"]
        demand_index, signal = mp.demand_index(
            high, low, close, volume, len_bs=19, len_bs_ma=19, len_di_ma=19, ma_type="ema"
        )

        assert len(demand_index) == len(close)
        assert len(signal) == len(close)
        assert np.isfinite(demand_index).any()
        assert np.isfinite(signal).any()
        valid = demand_index[np.isfinite(demand_index)]
        assert np.all((valid >= -100.0) & (valid <= 100.0))

    def test_kernel_parity(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]
        volume = test_data["volume"]

        di_auto, signal_auto = mp.demand_index(
            high, low, close, volume, len_bs=10, len_bs_ma=10, len_di_ma=8
        )
        di_scalar, signal_scalar = mp.demand_index(
            high,
            low,
            close,
            volume,
            len_bs=10,
            len_bs_ma=10,
            len_di_ma=8,
            kernel="scalar",
        )

        np.testing.assert_allclose(di_auto, di_scalar, rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(
            signal_auto, signal_scalar, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_invalid_params(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]
        volume = test_data["volume"]

        with pytest.raises(ValueError, match="Invalid len_bs"):
            mp.demand_index(high, low, close, volume, len_bs=0)

        with pytest.raises(ValueError, match="Invalid ma_type"):
            mp.demand_index(high, low, close, volume, ma_type="bad_ma")

    def test_stream_matches_batch(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]
        volume = test_data["volume"]
        di_batch, signal_batch = mp.demand_index(
            high, low, close, volume, len_bs=10, len_bs_ma=10, len_di_ma=8
        )

        stream = mp.DemandIndexStream(10, 10, 8, "ema")
        di_stream = []
        signal_stream = []
        for h, l, c, v in zip(high, low, close, volume):
            out = stream.update(h, l, c, v)
            if out is None:
                di_stream.append(np.nan)
                signal_stream.append(np.nan)
            else:
                di_stream.append(out[0])
                signal_stream.append(out[1])

        np.testing.assert_allclose(
            di_stream, di_batch, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            signal_stream, signal_batch, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]
        volume = test_data["volume"][:256]
        batch = mp.demand_index_batch(
            high,
            low,
            close,
            volume,
            len_bs_range=(19, 19, 0),
            len_bs_ma_range=(19, 19, 0),
            len_di_ma_range=(19, 19, 0),
            ma_type="ema",
        )
        di, signal = mp.demand_index(high, low, close, volume)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_allclose(
            batch["demand_index"][0], di, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            batch["signal"][0], signal, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_metadata(self, test_data):
        high = test_data["high"][:192]
        low = test_data["low"][:192]
        close = test_data["close"][:192]
        volume = test_data["volume"][:192]
        batch = mp.demand_index_batch(
            high,
            low,
            close,
            volume,
            len_bs_range=(10, 12, 2),
            len_bs_ma_range=(8, 10, 2),
            len_di_ma_range=(5, 6, 1),
            ma_type="ema",
        )

        assert batch["rows"] == 8
        assert batch["cols"] == len(close)
        assert batch["demand_index"].shape == (8, len(close))
        assert batch["signal"].shape == (8, len(close))
        np.testing.assert_array_equal(
            batch["len_bs"], np.array([10, 10, 10, 10, 12, 12, 12, 12], dtype=np.uint64)
        )
        np.testing.assert_array_equal(
            batch["len_bs_ma"], np.array([8, 8, 10, 10, 8, 8, 10, 10], dtype=np.uint64)
        )
        np.testing.assert_array_equal(
            batch["len_di_ma"], np.array([5, 6, 5, 6, 5, 6, 5, 6], dtype=np.uint64)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
