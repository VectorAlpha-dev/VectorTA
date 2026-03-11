import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    try:
        import my_project as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


class TestRollingSkewnessKurtosis:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = test_data["close"][:256]
        skewness, kurtosis = mp.rolling_skewness_kurtosis(close, 50, 3)

        assert len(skewness) == len(close)
        assert len(kurtosis) == len(close)
        skew_valid = np.flatnonzero(~np.isnan(skewness))
        kurt_valid = np.flatnonzero(~np.isnan(kurtosis))
        assert skew_valid.size > 0
        assert kurt_valid.size > 0
        assert int(skew_valid[0]) == 51
        assert int(kurt_valid[0]) == 51

    def test_kernel_parity(self, test_data):
        close = test_data["close"]
        auto_skew, auto_kurt = mp.rolling_skewness_kurtosis(close, 50, 3)
        scalar_skew, scalar_kurt = mp.rolling_skewness_kurtosis(
            close, 50, 3, kernel="scalar"
        )
        np.testing.assert_allclose(auto_skew, scalar_skew, rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(auto_kurt, scalar_kurt, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_invalid_params(self, test_data):
        close = test_data["close"][:64]

        with pytest.raises(ValueError, match="Invalid length"):
            mp.rolling_skewness_kurtosis(close, 0, 3)

        with pytest.raises(ValueError, match="Invalid smooth_length"):
            mp.rolling_skewness_kurtosis(close, 10, 0)

    def test_stream_matches_batch(self, test_data):
        close = test_data["close"]
        batch_skew, batch_kurt = mp.rolling_skewness_kurtosis(close, 50, 3)

        stream = mp.RollingSkewnessKurtosisStream(50, 3)
        skew_stream = []
        kurt_stream = []
        for value in close:
            out = stream.update(float(value))
            if out is None:
                skew_stream.append(np.nan)
                kurt_stream.append(np.nan)
            else:
                skew, kurt = out
                skew_stream.append(skew)
                kurt_stream.append(kurt)

        np.testing.assert_allclose(
            skew_stream, batch_skew, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            kurt_stream, batch_kurt, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = test_data["close"]
        batch = mp.rolling_skewness_kurtosis_batch(
            close,
            length_range=(50, 50, 0),
            smooth_length_range=(3, 3, 0),
        )
        single_skew, single_kurt = mp.rolling_skewness_kurtosis(close, 50, 3)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_allclose(
            batch["skewness"][0], single_skew, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            batch["kurtosis"][0], single_kurt, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_multiple_ranges(self, test_data):
        close = test_data["close"][:256]
        batch = mp.rolling_skewness_kurtosis_batch(
            close,
            length_range=(40, 50, 10),
            smooth_length_range=(2, 3, 1),
        )

        assert batch["rows"] == 4
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(
            batch["lengths"], np.array([40, 40, 50, 50], dtype=np.uint64)
        )
        np.testing.assert_array_equal(
            batch["smooth_lengths"], np.array([2, 3, 2, 3], dtype=np.uint64)
        )
