import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python' first",
        allow_module_level=True,
    )


class TestStochasticConnorsRsi:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        source = np.asarray(test_data["close"][:256], dtype=np.float64)

        k, d = mp.stochastic_connors_rsi(source, 3, 3, 3, 3, 2, 100)

        assert len(k) == len(source)
        assert len(d) == len(source)
        assert np.isnan(k[:104]).all()
        assert np.isnan(d[:106]).all()
        assert np.isfinite(k[104:]).any()
        assert np.isfinite(d[106:]).any()

    def test_kernel_parity(self, test_data):
        source = np.asarray(test_data["close"][:256], dtype=np.float64)

        auto_k, auto_d = mp.stochastic_connors_rsi(source, 5, 4, 3, 4, 3, 30)
        scalar_k, scalar_d = mp.stochastic_connors_rsi(
            source, 5, 4, 3, 4, 3, 30, kernel="scalar"
        )

        np.testing.assert_allclose(auto_k, scalar_k, rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(auto_d, scalar_d, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_invalid_roc_length(self, test_data):
        source = np.asarray(test_data["close"][:128], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid roc_length"):
            mp.stochastic_connors_rsi(source, 3, 3, 3, 3, 2, 0)

    def test_stream_matches_batch(self, test_data):
        source = np.asarray(test_data["close"][:220], dtype=np.float64)

        batch_k, batch_d = mp.stochastic_connors_rsi(source, 3, 3, 3, 3, 2, 20)
        stream = mp.StochasticConnorsRsiStream(3, 3, 3, 3, 2, 20)
        stream_k = []
        stream_d = []

        for value in source:
            out = stream.update(float(value))
            if out is None:
                stream_k.append(np.nan)
                stream_d.append(np.nan)
            else:
                kv, dv = out
                stream_k.append(kv)
                stream_d.append(dv)

        np.testing.assert_allclose(stream_k, batch_k, rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(stream_d, batch_d, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_batch_single_param_matches_single(self, test_data):
        source = np.asarray(test_data["close"][:220], dtype=np.float64)

        result = mp.stochastic_connors_rsi_batch(
            source,
            stoch_length_range=(3, 3, 0),
            smooth_k_range=(3, 3, 0),
            smooth_d_range=(3, 3, 0),
            rsi_length_range=(3, 3, 0),
            updown_length_range=(2, 2, 0),
            roc_length_range=(20, 20, 0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(source)
        assert result["k"].shape == (1, len(source))
        assert result["d"].shape == (1, len(source))
        np.testing.assert_array_equal(result["stoch_lengths"], np.array([3], dtype=np.uint64))
        np.testing.assert_array_equal(result["smooth_ks"], np.array([3], dtype=np.uint64))
        np.testing.assert_array_equal(result["smooth_ds"], np.array([3], dtype=np.uint64))
        np.testing.assert_array_equal(result["rsi_lengths"], np.array([3], dtype=np.uint64))
        np.testing.assert_array_equal(
            result["updown_lengths"], np.array([2], dtype=np.uint64)
        )
        np.testing.assert_array_equal(result["roc_lengths"], np.array([20], dtype=np.uint64))

        single_k, single_d = mp.stochastic_connors_rsi(source, 3, 3, 3, 3, 2, 20)
        np.testing.assert_allclose(
            result["k"][0], single_k, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["d"][0], single_d, rtol=1e-10, atol=1e-10, equal_nan=True
        )
