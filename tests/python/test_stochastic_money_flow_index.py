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


class TestStochasticMoneyFlowIndex:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = test_data["close"][:256]
        volume = test_data["volume"][:256]
        k, d = mp.stochastic_money_flow_index(close, volume, 14, 3, 3, 14)

        assert len(k) == len(close)
        assert len(d) == len(close)
        assert int(np.flatnonzero(~np.isnan(k))[0]) == 28
        assert int(np.flatnonzero(~np.isnan(d))[0]) == 30

    def test_kernel_parity(self, test_data):
        close = test_data["close"]
        volume = test_data["volume"]
        auto_k, auto_d = mp.stochastic_money_flow_index(close, volume, 14, 3, 3, 14)
        scalar_k, scalar_d = mp.stochastic_money_flow_index(
            close, volume, 14, 3, 3, 14, kernel="scalar"
        )
        np.testing.assert_allclose(auto_k, scalar_k, rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(auto_d, scalar_d, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_invalid_params(self, test_data):
        close = test_data["close"][:64]
        volume = test_data["volume"][:64]
        with pytest.raises(ValueError, match="Invalid stoch_k_length"):
            mp.stochastic_money_flow_index(close, volume, 0, 3, 3, 14)

    def test_stream_matches_batch(self, test_data):
        close = test_data["close"]
        volume = test_data["volume"]
        batch_k, batch_d = mp.stochastic_money_flow_index(close, volume, 14, 3, 3, 14)

        stream = mp.StochasticMoneyFlowIndexStream(14, 3, 3, 14)
        k_stream = []
        d_stream = []
        for src, vol in zip(close, volume):
            out = stream.update(float(src), float(vol))
            if out is None:
                k_stream.append(np.nan)
                d_stream.append(np.nan)
            else:
                k_val, d_val = out
                k_stream.append(k_val)
                d_stream.append(d_val)

        np.testing.assert_allclose(
            k_stream, batch_k, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            d_stream, batch_d, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = test_data["close"]
        volume = test_data["volume"]
        batch = mp.stochastic_money_flow_index_batch(
            close,
            volume,
            stoch_k_length_range=(14, 14, 0),
            stoch_k_smooth_range=(3, 3, 0),
            stoch_d_smooth_range=(3, 3, 0),
            mfi_length_range=(14, 14, 0),
        )
        single_k, single_d = mp.stochastic_money_flow_index(close, volume, 14, 3, 3, 14)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_allclose(
            batch["k"][0], single_k, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            batch["d"][0], single_d, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_multiple_ranges(self, test_data):
        close = test_data["close"][:220]
        volume = test_data["volume"][:220]
        batch = mp.stochastic_money_flow_index_batch(
            close,
            volume,
            stoch_k_length_range=(10, 14, 2),
            stoch_k_smooth_range=(3, 3, 0),
            stoch_d_smooth_range=(3, 3, 0),
            mfi_length_range=(14, 14, 0),
        )

        assert batch["rows"] == 3
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(
            batch["stoch_k_lengths"], np.array([10, 12, 14], dtype=np.uint64)
        )
