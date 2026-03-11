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


class TestDualUlcerIndex:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = test_data["close"][:256]

        long_ulcer, short_ulcer, threshold = mp.dual_ulcer_index(
            close,
            period=5,
            auto_threshold=True,
            threshold=0.1,
        )

        assert len(long_ulcer) == len(close)
        assert len(short_ulcer) == len(close)
        assert len(threshold) == len(close)

        long_valid = np.flatnonzero(~np.isnan(long_ulcer))
        short_valid = np.flatnonzero(~np.isnan(short_ulcer))
        threshold_valid = np.flatnonzero(~np.isnan(threshold))
        assert long_valid.size > 0
        assert short_valid.size > 0
        assert threshold_valid.size > 0
        assert int(long_valid[0]) == 8
        assert int(short_valid[0]) == 8
        assert int(threshold_valid[0]) == 8
        assert np.nanmin(long_ulcer) >= 0.0
        assert np.nanmin(short_ulcer) >= 0.0
        assert np.nanmin(threshold) >= 0.0

    def test_kernel_parity(self, test_data):
        close = test_data["close"]

        auto_long, auto_short, auto_threshold = mp.dual_ulcer_index(
            close, 5, True, 0.1
        )
        scalar_long, scalar_short, scalar_threshold = mp.dual_ulcer_index(
            close, 5, True, 0.1, kernel="scalar"
        )

        np.testing.assert_allclose(
            auto_long, scalar_long, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            auto_short, scalar_short, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            auto_threshold,
            scalar_threshold,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )

    def test_invalid_params(self, test_data):
        close = test_data["close"]

        with pytest.raises(ValueError, match="Invalid period"):
            mp.dual_ulcer_index(close, period=0)

        with pytest.raises(ValueError, match="Invalid threshold"):
            mp.dual_ulcer_index(close, period=5, auto_threshold=False, threshold=-0.1)

    def test_stream_matches_batch(self, test_data):
        close = test_data["close"]
        batch_long, batch_short, batch_threshold = mp.dual_ulcer_index(close, 6, True, 0.1)

        stream = mp.DualUlcerIndexStream(6, True, 0.1)
        stream_long = []
        stream_short = []
        stream_threshold = []

        for value in close:
            out = stream.update(value)
            if out is None:
                stream_long.append(np.nan)
                stream_short.append(np.nan)
                stream_threshold.append(np.nan)
            else:
                long_ulcer, short_ulcer, threshold = out
                stream_long.append(long_ulcer)
                stream_short.append(short_ulcer)
                stream_threshold.append(threshold)

        np.testing.assert_allclose(
            stream_long, batch_long, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_short, batch_short, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_threshold,
            batch_threshold,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = test_data["close"]

        batch = mp.dual_ulcer_index_batch(
            close,
            period_range=(5, 5, 0),
            threshold_range=(0.1, 0.1, 0.0),
            auto_threshold=True,
        )
        single_long, single_short, single_threshold = mp.dual_ulcer_index(
            close, 5, True, 0.1
        )

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        assert batch["long_ulcer"].shape == (1, len(close))
        assert batch["short_ulcer"].shape == (1, len(close))
        assert batch["threshold"].shape == (1, len(close))

        np.testing.assert_allclose(
            batch["long_ulcer"][0],
            single_long,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            batch["short_ulcer"][0],
            single_short,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            batch["threshold"][0],
            single_threshold,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )

    def test_batch_metadata_multiple_ranges(self, test_data):
        close = test_data["close"][:256]

        batch = mp.dual_ulcer_index_batch(
            close,
            period_range=(5, 7, 2),
            threshold_range=(0.1, 0.3, 0.2),
            auto_threshold=False,
        )

        assert batch["rows"] == 4
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(
            batch["periods"], np.array([5, 5, 7, 7], dtype=np.uint64)
        )
        np.testing.assert_allclose(batch["threshold_values"], np.array([0.1, 0.3, 0.1, 0.3]))
        assert list(batch["auto_threshold"]) == [False, False, False, False]
