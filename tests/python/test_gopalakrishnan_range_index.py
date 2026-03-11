import numpy as np
import pytest

from test_utils import load_test_data

try:
    import my_project as mp
except ImportError:
    try:
        import vector_ta as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


class TestGopalakrishnanRangeIndex:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        values = mp.gopalakrishnan_range_index(high, low, length=5)

        assert len(values) == len(high)
        valid = np.flatnonzero(~np.isnan(values))
        assert valid.size > 0
        assert int(valid[0]) >= 4

        tail_start = min(int(valid[0]) + 32, len(values))
        assert not np.any(np.isnan(values[tail_start:]))

    def test_kernel_parity(self, test_data):
        high = test_data["high"]
        low = test_data["low"]

        auto = mp.gopalakrishnan_range_index(high, low, length=9)
        scalar = mp.gopalakrishnan_range_index(high, low, length=9, kernel="scalar")

        np.testing.assert_allclose(auto, scalar, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_invalid_length(self, test_data):
        high = test_data["high"]
        low = test_data["low"]

        with pytest.raises(ValueError, match="Invalid length"):
            mp.gopalakrishnan_range_index(high, low, length=1)

    def test_length_mismatch(self, test_data):
        high = test_data["high"]
        low = test_data["low"]

        with pytest.raises(ValueError, match="length mismatch"):
            mp.gopalakrishnan_range_index(high[:-1], low, length=5)

    def test_streaming_matches_batch(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        batch = mp.gopalakrishnan_range_index(high, low, length=8)

        stream = mp.GopalakrishnanRangeIndexStream(8)
        streamed = []
        for h, l in zip(high, low):
            out = stream.update(h, l)
            streamed.append(np.nan if out is None else out)

        np.testing.assert_allclose(
            streamed, batch, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        high = test_data["high"]
        low = test_data["low"]
        result = mp.gopalakrishnan_range_index_batch(
            high,
            low,
            length_range=(5, 5, 0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(high)
        assert result["values"].shape == (1, len(high))
        np.testing.assert_array_equal(result["lengths"], np.array([5], dtype=np.uint64))

        single = mp.gopalakrishnan_range_index(high, low, length=5)
        np.testing.assert_allclose(
            result["values"][0], single, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_multiple_metadata(self, test_data):
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        result = mp.gopalakrishnan_range_index_batch(
            high,
            low,
            length_range=(5, 9, 2),
        )

        assert result["rows"] == 3
        assert result["cols"] == len(high)
        assert result["values"].shape == (3, len(high))
        np.testing.assert_array_equal(
            result["lengths"], np.array([5, 7, 9], dtype=np.uint64)
        )

    def test_invalid_window_recovers(self, test_data):
        high = test_data["high"][:80].copy()
        low = test_data["low"][:80].copy()
        high[30] = np.nan
        low[30] = np.nan

        values = mp.gopalakrishnan_range_index(high, low, length=10)
        assert np.isnan(values[30])
        assert np.isnan(values[39])
        assert np.isfinite(values[40])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
