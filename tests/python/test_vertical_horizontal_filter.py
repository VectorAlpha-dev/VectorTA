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


class TestVerticalHorizontalFilter:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = test_data["close"]
        values = mp.vertical_horizontal_filter(close, length=28)

        assert len(values) == len(close)
        valid = np.flatnonzero(~np.isnan(values))
        assert valid.size > 0
        assert int(valid[0]) >= 28

        tail_start = min(int(valid[0]) + 32, len(values))
        assert not np.any(np.isnan(values[tail_start:]))

    def test_kernel_parity(self, test_data):
        close = test_data["close"]
        auto = mp.vertical_horizontal_filter(close, length=21)
        scalar = mp.vertical_horizontal_filter(close, length=21, kernel="scalar")

        np.testing.assert_allclose(auto, scalar, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_invalid_length(self, test_data):
        close = test_data["close"]

        with pytest.raises(ValueError, match="Invalid length"):
            mp.vertical_horizontal_filter(close, length=0)

    def test_streaming_matches_batch(self, test_data):
        close = test_data["close"]
        batch = mp.vertical_horizontal_filter(close, length=16)

        stream = mp.VerticalHorizontalFilterStream(16)
        streamed = []
        for value in close:
            out = stream.update(value)
            streamed.append(np.nan if out is None else out)

        np.testing.assert_allclose(
            streamed, batch, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = test_data["close"]
        result = mp.vertical_horizontal_filter_batch(
            close,
            length_range=(28, 28, 0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(close)
        assert result["values"].shape == (1, len(close))
        np.testing.assert_array_equal(result["lengths"], np.array([28], dtype=np.uint64))

        single = mp.vertical_horizontal_filter(close, length=28)
        np.testing.assert_allclose(
            result["values"][0], single, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_multiple_metadata(self, test_data):
        close = test_data["close"][:256]
        result = mp.vertical_horizontal_filter_batch(
            close,
            length_range=(28, 32, 2),
        )

        assert result["rows"] == 3
        assert result["cols"] == len(close)
        assert result["values"].shape == (3, len(close))
        np.testing.assert_array_equal(
            result["lengths"], np.array([28, 30, 32], dtype=np.uint64)
        )

    def test_invalid_window_recovers(self, test_data):
        close = test_data["close"][:96].copy()
        close[30] = np.nan

        values = mp.vertical_horizontal_filter(close, length=10)
        assert np.isnan(values[30])
        assert np.isnan(values[40])
        assert np.isfinite(values[41])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
