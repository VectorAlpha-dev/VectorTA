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


class TestDisparityIndex:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = test_data["close"][:256]
        values = mp.disparity_index(close, 14, 14, 9, "ema")

        assert len(values) == len(close)
        valid = np.flatnonzero(~np.isnan(values))
        assert valid.size > 0
        assert int(valid[0]) == 34
        assert np.nanmin(values) >= 0.0
        assert np.nanmax(values) <= 100.0

    def test_kernel_parity(self, test_data):
        close = test_data["close"]
        auto = mp.disparity_index(close, 14, 14, 9, "ema")
        scalar = mp.disparity_index(close, 14, 14, 9, "ema", kernel="scalar")
        np.testing.assert_allclose(auto, scalar, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_invalid_params(self, test_data):
        close = test_data["close"][:64]

        with pytest.raises(ValueError, match="Invalid ema_period"):
            mp.disparity_index(close, 0, 14, 9, "ema")

        with pytest.raises(ValueError, match="Invalid smoothing_type"):
            mp.disparity_index(close, 14, 14, 9, "bad")

    def test_stream_matches_batch(self, test_data):
        close = test_data["close"]
        batch = mp.disparity_index(close, 14, 14, 9, "sma")

        stream = mp.DisparityIndexStream(14, 14, 9, "sma")
        stream_values = []
        for value in close:
            out = stream.update(float(value))
            stream_values.append(np.nan if out is None else out)

        np.testing.assert_allclose(
            stream_values, batch, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = test_data["close"]
        batch = mp.disparity_index_batch(
            close,
            ema_period_range=(14, 14, 0),
            lookback_period_range=(14, 14, 0),
            smoothing_period_range=(9, 9, 0),
            smoothing_types=["ema"],
        )
        single = mp.disparity_index(close, 14, 14, 9, "ema")

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        assert batch["values"].shape == (1, len(close))
        assert batch["smoothing_types"] == ["ema"]
        np.testing.assert_allclose(
            batch["values"][0], single, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_multiple_ranges(self, test_data):
        close = test_data["close"][:256]
        batch = mp.disparity_index_batch(
            close,
            ema_period_range=(10, 14, 2),
            lookback_period_range=(14, 14, 0),
            smoothing_period_range=(9, 9, 0),
            smoothing_types=["ema", "sma"],
        )

        assert batch["rows"] == 6
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(
            batch["ema_periods"], np.array([10, 10, 12, 12, 14, 14], dtype=np.uint64)
        )
        assert batch["smoothing_types"] == ["ema", "sma", "ema", "sma", "ema", "sma"]
