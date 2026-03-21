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


class TestAutocorrelationIndicator:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = test_data["close"][:256]
        out = mp.autocorrelation_indicator(close, 20, 12, False)

        assert len(out["filtered"]) == len(close)
        assert out["correlations"].shape == (12, len(close))
        assert out["lag_count"] == 12
        assert out["cols"] == len(close)
        lag1_valid = np.flatnonzero(~np.isnan(out["correlations"][0]))
        lag12_valid = np.flatnonzero(~np.isnan(out["correlations"][11]))
        assert lag1_valid.size > 0
        assert lag12_valid.size > 0
        assert int(lag1_valid[0]) == 20
        assert int(lag12_valid[0]) == 31

    def test_kernel_parity(self, test_data):
        close = test_data["close"]
        auto = mp.autocorrelation_indicator(close, 20, 12, False)
        scalar = mp.autocorrelation_indicator(close, 20, 12, False, kernel="scalar")
        np.testing.assert_allclose(
            auto["filtered"], scalar["filtered"], rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            auto["correlations"],
            scalar["correlations"],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )

    def test_invalid_params(self, test_data):
        close = test_data["close"][:64]

        with pytest.raises(ValueError, match="Invalid length"):
            mp.autocorrelation_indicator(close, 0, 8, False)

        with pytest.raises(ValueError, match="Invalid max_lag"):
            mp.autocorrelation_indicator(close, 20, 0, False)

    def test_stream_matches_batch(self, test_data):
        close = test_data["close"][:180]
        batch = mp.autocorrelation_indicator(close, 16, 8, False)

        stream = mp.AutocorrelationIndicatorStream(16, 8, False)
        filtered = []
        correlations = []
        for value in close:
            out = stream.update(float(value))
            assert out is not None
            filt, corr = out
            filtered.append(filt)
            correlations.append(corr)

        np.testing.assert_allclose(
            filtered, batch["filtered"], rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            np.asarray(correlations).T,
            batch["correlations"],
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = test_data["close"][:180]
        batch = mp.autocorrelation_indicator_batch(
            close,
            length_range=(20, 20, 0),
            max_lag=8,
            use_test_signal=False,
        )
        single = mp.autocorrelation_indicator(close, 20, 8, False)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        assert batch["lag_count"] == 8
        np.testing.assert_allclose(
            batch["filtered"][0], single["filtered"], rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            batch["correlations"][0],
            single["correlations"],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )

    def test_batch_metadata_multiple_ranges(self, test_data):
        close = test_data["close"][:220]
        batch = mp.autocorrelation_indicator_batch(
            close,
            length_range=(16, 20, 2),
            max_lag=6,
            use_test_signal=True,
        )

        assert batch["rows"] == 3
        assert batch["cols"] == len(close)
        assert batch["lag_count"] == 6
        np.testing.assert_array_equal(
            batch["lengths"], np.array([16, 18, 20], dtype=np.uint64)
        )
