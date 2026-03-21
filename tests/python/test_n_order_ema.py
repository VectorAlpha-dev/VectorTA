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


class TestNOrderEma:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_default_output_contract(self, test_data):
        close = np.asarray(test_data["close"][:256], dtype=np.float64)
        result = mp.n_order_ema(close)

        assert len(result) == len(close)
        assert int(np.flatnonzero(~np.isnan(result))[0]) == 8
        assert np.all(np.isnan(result[:8]))
        assert np.all(np.isfinite(result[32:]))

    def test_invalid_params(self, test_data):
        close = np.asarray(test_data["close"][:128], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid period"):
            mp.n_order_ema(close, period=0.0)

        with pytest.raises(ValueError, match="Invalid order"):
            mp.n_order_ema(close, order=0)

        with pytest.raises(ValueError, match="Invalid ema_style"):
            mp.n_order_ema(close, ema_style="bad")

        with pytest.raises(ValueError, match="Invalid iir_style"):
            mp.n_order_ema(close, iir_style="bad")

    def test_stream_matches_batch(self, test_data):
        close = np.asarray(test_data["close"][:320], dtype=np.float64)
        batch = mp.n_order_ema(
            close,
            period=15.0,
            order=2,
            ema_style="hema",
            iir_style="bilinear",
        )

        stream = mp.NOrderEmaStream(
            period=15.0,
            order=2,
            ema_style="hema",
            iir_style="bilinear",
        )
        streamed = np.asarray([stream.update(float(v)) for v in close], dtype=np.float64)
        np.testing.assert_allclose(streamed, batch, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_batch_single_param_matches_single(self, test_data):
        close = np.asarray(test_data["close"][:220], dtype=np.float64)
        batch = mp.n_order_ema_batch(
            close,
            period_range=(12.5, 12.5, 0.0),
            order_range=(3, 3, 0),
            ema_style="tema",
            iir_style="matched_z",
        )
        single = mp.n_order_ema(
            close,
            period=12.5,
            order=3,
            ema_style="tema",
            iir_style="matched_z",
        )

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_allclose(
            batch["values"][0], single, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_multiple_ranges(self, test_data):
        close = np.asarray(test_data["close"][:180], dtype=np.float64)
        batch = mp.n_order_ema_batch(
            close,
            period_range=(9.0, 10.0, 1.0),
            order_range=(1, 2, 1),
            ema_style="dema",
            iir_style="all_pole",
        )

        assert batch["rows"] == 4
        assert batch["cols"] == len(close)
        np.testing.assert_allclose(batch["periods"], np.array([9.0, 9.0, 10.0, 10.0]))
        np.testing.assert_array_equal(batch["orders"], np.array([1, 2, 1, 2], dtype=np.uint64))
