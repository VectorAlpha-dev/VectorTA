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


class TestVolumeWeightedStochasticRsi:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        source = np.asarray(test_data["close"][:256], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:256], dtype=np.float64)

        k, d = mp.volume_weighted_stochastic_rsi(source, volume, 14, 14, 3, 3, "WSMA")

        assert len(k) == len(source)
        assert len(d) == len(source)
        assert np.isnan(k[:29]).all()
        assert np.isnan(d[:31]).all()
        assert np.isfinite(k[29:]).any()
        assert np.isfinite(d[31:]).any()

    def test_kernel_parity(self, test_data):
        source = np.asarray(test_data["close"][:256], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:256], dtype=np.float64)

        auto_k, auto_d = mp.volume_weighted_stochastic_rsi(
            source, volume, 12, 10, 4, 3, "EMA"
        )
        scalar_k, scalar_d = mp.volume_weighted_stochastic_rsi(
            source, volume, 12, 10, 4, 3, "EMA", kernel="scalar"
        )

        np.testing.assert_allclose(auto_k, scalar_k, rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(auto_d, scalar_d, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_invalid_ma_type(self, test_data):
        source = np.asarray(test_data["close"][:128], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:128], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid MA type"):
            mp.volume_weighted_stochastic_rsi(source, volume, 14, 14, 3, 3, "BAD")

    def test_stream_matches_batch(self, test_data):
        source = np.asarray(test_data["close"][:192], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:192], dtype=np.float64)

        batch_k, batch_d = mp.volume_weighted_stochastic_rsi(
            source, volume, 14, 10, 5, 4, "WMA"
        )
        stream = mp.VolumeWeightedStochasticRsiStream(14, 10, 5, 4, "WMA")
        stream_k = []
        stream_d = []

        for src, vol in zip(source, volume):
            out = stream.update(float(src), float(vol))
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
        source = np.asarray(test_data["close"][:192], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:192], dtype=np.float64)

        result = mp.volume_weighted_stochastic_rsi_batch(
            source,
            volume,
            rsi_length_range=(14, 14, 0),
            stoch_length_range=(14, 14, 0),
            k_length_range=(3, 3, 0),
            d_length_range=(3, 3, 0),
            ma_type="VWMA",
        )

        assert result["rows"] == 1
        assert result["cols"] == len(source)
        assert result["k"].shape == (1, len(source))
        assert result["d"].shape == (1, len(source))
        np.testing.assert_array_equal(result["rsi_lengths"], np.array([14], dtype=np.uint64))
        np.testing.assert_array_equal(result["stoch_lengths"], np.array([14], dtype=np.uint64))
        np.testing.assert_array_equal(result["k_lengths"], np.array([3], dtype=np.uint64))
        np.testing.assert_array_equal(result["d_lengths"], np.array([3], dtype=np.uint64))

        single_k, single_d = mp.volume_weighted_stochastic_rsi(
            source, volume, 14, 14, 3, 3, "VWMA"
        )
        np.testing.assert_allclose(
            result["k"][0], single_k, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["d"][0], single_d, rtol=1e-10, atol=1e-10, equal_nan=True
        )
