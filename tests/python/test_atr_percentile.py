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


class TestAtrPercentile:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]
        values = mp.atr_percentile(
            high, low, close, atr_length=10, percentile_length=50
        )

        assert len(values) == len(close)
        valid = np.flatnonzero(~np.isnan(values))
        assert valid.size > 0
        assert int(valid[0]) >= 59
        assert np.all((values[valid] >= 0.0) & (values[valid] <= 100.0))

    def test_kernel_parity(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        auto = mp.atr_percentile(high, low, close, atr_length=7, percentile_length=21)
        scalar = mp.atr_percentile(
            high, low, close, atr_length=7, percentile_length=21, kernel="scalar"
        )

        np.testing.assert_allclose(auto, scalar, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_invalid_lengths(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        with pytest.raises(ValueError, match="Invalid ATR length"):
            mp.atr_percentile(high, low, close, atr_length=0, percentile_length=10)

        with pytest.raises(ValueError, match="Invalid percentile length"):
            mp.atr_percentile(high, low, close, atr_length=10, percentile_length=0)

    def test_length_mismatch(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]

        with pytest.raises(ValueError, match="length mismatch"):
            mp.atr_percentile(high[:-1], low, close, atr_length=10, percentile_length=20)

    def test_streaming_matches_batch(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]
        batch = mp.atr_percentile(high, low, close, atr_length=6, percentile_length=12)

        stream = mp.AtrPercentileStream(6, 12)
        streamed = []
        for h, l, c in zip(high, low, close):
            out = stream.update(h, l, c)
            streamed.append(np.nan if out is None else out)

        np.testing.assert_allclose(
            streamed, batch, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        close = test_data["close"]
        result = mp.atr_percentile_batch(
            high,
            low,
            close,
            atr_length_range=(10, 10, 0),
            percentile_length_range=(20, 20, 0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(close)
        assert result["values"].shape == (1, len(close))
        np.testing.assert_array_equal(
            result["atr_lengths"], np.array([10], dtype=np.uint64)
        )
        np.testing.assert_array_equal(
            result["percentile_lengths"], np.array([20], dtype=np.uint64)
        )

        single = mp.atr_percentile(
            high, low, close, atr_length=10, percentile_length=20
        )
        np.testing.assert_allclose(
            result["values"][0], single, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_multiple_metadata(self, test_data):
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]
        result = mp.atr_percentile_batch(
            high,
            low,
            close,
            atr_length_range=(10, 14, 2),
            percentile_length_range=(20, 24, 2),
        )

        assert result["rows"] == 9
        assert result["cols"] == len(close)
        assert result["values"].shape == (9, len(close))
        np.testing.assert_array_equal(
            result["atr_lengths"], np.array([10, 10, 10, 12, 12, 12, 14, 14, 14], dtype=np.uint64)
        )
        np.testing.assert_array_equal(
            result["percentile_lengths"],
            np.array([20, 22, 24, 20, 22, 24, 20, 22, 24], dtype=np.uint64),
        )

    def test_invalid_window_recovers(self, test_data):
        high = test_data["high"][:96].copy()
        low = test_data["low"][:96].copy()
        close = test_data["close"][:96].copy()
        high[30] = np.nan
        low[30] = np.nan
        close[30] = np.nan

        values = mp.atr_percentile(
            high, low, close, atr_length=5, percentile_length=5
        )
        assert np.isnan(values[30])
        assert np.isnan(values[39])
        assert np.isfinite(values[40])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
