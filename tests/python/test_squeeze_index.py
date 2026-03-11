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


class TestSqueezeIndex:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = test_data["close"]
        values = mp.squeeze_index(close, conv=50.0, length=20)

        assert len(values) == len(close)
        valid = np.flatnonzero(~np.isnan(values))
        assert valid.size > 0
        assert int(valid[0]) >= 19

        tail_start = min(int(valid[0]) + 32, len(values))
        assert not np.any(np.isnan(values[tail_start:]))

    def test_kernel_parity(self, test_data):
        close = test_data["close"]
        auto = mp.squeeze_index(close, conv=50.0, length=20)
        scalar = mp.squeeze_index(close, conv=50.0, length=20, kernel="scalar")

        np.testing.assert_allclose(auto, scalar, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_invalid_params(self, test_data):
        close = test_data["close"][:64]

        with pytest.raises(ValueError, match="Invalid conv"):
            mp.squeeze_index(close, conv=1.0, length=20)

        with pytest.raises(ValueError, match="Invalid length"):
            mp.squeeze_index(close, conv=50.0, length=0)

    def test_streaming_matches_batch_and_resets_after_nan(self, test_data):
        close = np.array(test_data["close"][:160], dtype=float)
        close[40] = np.nan

        batch = mp.squeeze_index(close, conv=50.0, length=10)
        stream = mp.SqueezeIndexStream(conv=50.0, length=10)
        streamed = []

        for value in close:
            out = stream.update(value)
            streamed.append(np.nan if out is None else out)

        streamed = np.array(streamed)
        np.testing.assert_allclose(streamed, batch, rtol=1e-10, atol=1e-10, equal_nan=True)
        assert np.isnan(streamed[40])
        assert np.isnan(streamed[48])
        assert np.isfinite(streamed[50])

    def test_batch_single_param_matches_single(self, test_data):
        close = test_data["close"][:256]
        result = mp.squeeze_index_batch(
            close,
            conv_range=(50.0, 50.0, 0.0),
            length_range=(20, 20, 0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(close)
        assert result["values"].shape == (1, len(close))
        np.testing.assert_allclose(result["convs"], np.array([50.0]), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(result["lengths"], np.array([20], dtype=np.uint64))

        single = mp.squeeze_index(close, conv=50.0, length=20)
        np.testing.assert_allclose(
            result["values"][0], single, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_metadata(self, test_data):
        close = test_data["close"][:200]
        result = mp.squeeze_index_batch(
            close,
            conv_range=(50.0, 60.0, 10.0),
            length_range=(20, 24, 4),
        )

        assert result["rows"] == 4
        assert result["cols"] == len(close)
        assert result["values"].shape == (4, len(close))
        np.testing.assert_allclose(
            result["convs"], np.array([50.0, 50.0, 60.0, 60.0]), rtol=0, atol=1e-12
        )
        np.testing.assert_array_equal(
            result["lengths"], np.array([20, 24, 20, 24], dtype=np.uint64)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
