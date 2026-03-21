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


class TestInsyncIndex:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = np.asarray(test_data["high"][:320], dtype=np.float64)
        low = np.asarray(test_data["low"][:320], dtype=np.float64)
        close = np.asarray(test_data["close"][:320], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:320], dtype=np.float64)

        result = mp.insync_index(high, low, close, volume)

        assert len(result) == len(close)
        assert np.isfinite(result[0])
        assert np.isfinite(result[25:]).any()
        assert not np.any(np.isnan(result[: min(32, len(result))]))

    def test_kernel_parity(self, test_data):
        high = np.asarray(test_data["high"][:320], dtype=np.float64)
        low = np.asarray(test_data["low"][:320], dtype=np.float64)
        close = np.asarray(test_data["close"][:320], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:320], dtype=np.float64)

        auto = mp.insync_index(high, low, close, volume, kernel="auto")
        scalar = mp.insync_index(high, low, close, volume, kernel="scalar")

        np.testing.assert_allclose(auto, scalar, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_stream_matches_batch(self, test_data):
        high = np.asarray(test_data["high"][:240], dtype=np.float64)
        low = np.asarray(test_data["low"][:240], dtype=np.float64)
        close = np.asarray(test_data["close"][:240], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:240], dtype=np.float64)

        batch = mp.insync_index(high, low, close, volume)
        stream = mp.InsyncIndexStream()
        streamed = []
        for h, l, c, v in zip(high, low, close, volume):
            out = stream.update(float(h), float(l), float(c), float(v))
            streamed.append(np.nan if out is None else out)

        np.testing.assert_allclose(streamed, batch, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_batch_single_param_matches_single(self, test_data):
        high = np.asarray(test_data["high"][:240], dtype=np.float64)
        low = np.asarray(test_data["low"][:240], dtype=np.float64)
        close = np.asarray(test_data["close"][:240], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:240], dtype=np.float64)

        batch = mp.insync_index_batch(high, low, close, volume)
        single = mp.insync_index(high, low, close, volume)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        assert batch["values"].shape == (1, len(close))
        np.testing.assert_allclose(
            batch["values"][0], single, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_invalid_param(self, test_data):
        high = np.asarray(test_data["high"][:64], dtype=np.float64)
        low = np.asarray(test_data["low"][:64], dtype=np.float64)
        close = np.asarray(test_data["close"][:64], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:64], dtype=np.float64)

        with pytest.raises(ValueError, match="bb_multiplier"):
            mp.insync_index(high, low, close, volume, bb_multiplier=0.0)
