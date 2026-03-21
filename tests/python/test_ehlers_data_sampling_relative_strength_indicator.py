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


class TestEhlersDataSamplingRelativeStrengthIndicator:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        open_ = test_data["open"][:256]
        close = test_data["close"][:256]
        ds_rsi, original_rsi, signal = mp.ehlers_data_sampling_relative_strength_indicator(
            open_, close, 14
        )

        assert len(ds_rsi) == len(close)
        assert len(original_rsi) == len(close)
        assert len(signal) == len(close)
        assert int(np.flatnonzero(~np.isnan(ds_rsi))[0]) == 14
        assert int(np.flatnonzero(~np.isnan(original_rsi))[0]) == 14
        assert int(np.flatnonzero(~np.isnan(signal))[0]) == 14

    def test_invalid_params(self, test_data):
        open_ = test_data["open"][:64]
        close = test_data["close"][:64]
        with pytest.raises(ValueError, match="Invalid length"):
            mp.ehlers_data_sampling_relative_strength_indicator(open_, close, 0)

    def test_stream_matches_batch(self, test_data):
        open_ = test_data["open"][:320]
        close = test_data["close"][:320]
        batch_ds, batch_orig, batch_sig = mp.ehlers_data_sampling_relative_strength_indicator(
            open_, close, 14
        )

        stream = mp.EhlersDataSamplingRelativeStrengthIndicatorStream(14)
        stream_ds = []
        stream_orig = []
        stream_sig = []
        for o, c in zip(open_, close):
            out = stream.update(float(o), float(c))
            if out is None:
                stream_ds.append(np.nan)
                stream_orig.append(np.nan)
                stream_sig.append(np.nan)
            else:
                a, b, d = out
                stream_ds.append(a)
                stream_orig.append(b)
                stream_sig.append(d)

        np.testing.assert_allclose(
            stream_ds, batch_ds, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_orig, batch_orig, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_sig, batch_sig, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        open_ = test_data["open"][:300]
        close = test_data["close"][:300]
        batch = mp.ehlers_data_sampling_relative_strength_indicator_batch(
            open_, close, length_range=(14, 14, 0)
        )
        single = mp.ehlers_data_sampling_relative_strength_indicator(open_, close, 14)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_allclose(
            batch["ds_rsi"][0], single[0], rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            batch["original_rsi"][0],
            single[1],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            batch["signal"][0], single[2], rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_multiple_lengths(self, test_data):
        open_ = test_data["open"][:256]
        close = test_data["close"][:256]
        batch = mp.ehlers_data_sampling_relative_strength_indicator_batch(
            open_, close, length_range=(10, 14, 2)
        )

        assert batch["rows"] == 3
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(
            batch["lengths"], np.array([10, 12, 14], dtype=np.uint64)
        )
