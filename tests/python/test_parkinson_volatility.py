"""
Python binding tests for Parkinson volatility.
"""
import pytest
import numpy as np

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python' first",
        allow_module_level=True,
    )


class TestParkinsonVolatility:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = test_data["high"]
        low = test_data["low"]

        volatility, variance = mp.parkinson_volatility(high, low, 8)

        assert len(volatility) == len(high)
        assert len(variance) == len(high)

        valid = np.flatnonzero(~np.isnan(volatility))
        assert valid.size > 0
        first_valid = int(valid[0])
        assert first_valid >= 7

        tail_start = min(first_valid + 32, len(volatility))
        assert not np.any(np.isnan(volatility[tail_start:]))
        assert not np.any(np.isnan(variance[tail_start:]))
        np.testing.assert_allclose(
            variance[tail_start:],
            np.square(volatility[tail_start:]),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_kernel_parity(self, test_data):
        high = test_data["high"]
        low = test_data["low"]

        auto_vol, auto_var = mp.parkinson_volatility(high, low, 12)
        scalar_vol, scalar_var = mp.parkinson_volatility(high, low, 12, kernel="scalar")

        np.testing.assert_allclose(auto_vol, scalar_vol, rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(auto_var, scalar_var, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_invalid_period(self, test_data):
        high = test_data["high"]
        low = test_data["low"]

        with pytest.raises(ValueError, match="Invalid period"):
            mp.parkinson_volatility(high, low, 0)

    def test_streaming_matches_batch(self, test_data):
        high = test_data["high"]
        low = test_data["low"]

        batch_vol, batch_var = mp.parkinson_volatility(high, low, 10)

        stream = mp.ParkinsonVolatilityStream(10)
        stream_vol = []
        stream_var = []

        for h, l in zip(high, low):
            out = stream.update(h, l)
            if out is None:
                stream_vol.append(np.nan)
                stream_var.append(np.nan)
            else:
                vol, var = out
                stream_vol.append(vol)
                stream_var.append(var)

        np.testing.assert_allclose(stream_vol, batch_vol, rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(stream_var, batch_var, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_batch_single_param_matches_single(self, test_data):
        high = test_data["high"]
        low = test_data["low"]

        result = mp.parkinson_volatility_batch(high, low, period_range=(10, 10, 0))

        assert result["rows"] == 1
        assert result["cols"] == len(high)
        assert result["volatility"].shape == (1, len(high))
        assert result["variance"].shape == (1, len(high))
        np.testing.assert_array_equal(result["periods"], np.array([10], dtype=np.uint64))

        single_vol, single_var = mp.parkinson_volatility(high, low, 10)
        np.testing.assert_allclose(
            result["volatility"][0], single_vol, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["variance"][0], single_var, rtol=1e-10, atol=1e-10, equal_nan=True
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
