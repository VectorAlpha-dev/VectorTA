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


class TestVwapZscoreWithSignals:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = test_data["close"]
        volume = test_data["volume"]
        zvwap, support_signal, resistance_signal = mp.vwap_zscore_with_signals(
            close, volume, length=20, upper_bottom=2.5, lower_bottom=-2.5
        )

        assert len(zvwap) == len(close)
        assert len(support_signal) == len(close)
        assert len(resistance_signal) == len(close)
        assert np.isfinite(zvwap).any()
        valid_support = support_signal[np.isfinite(support_signal)]
        valid_resistance = resistance_signal[np.isfinite(resistance_signal)]
        assert np.all((valid_support == 0.0) | (valid_support == 1.0))
        assert np.all((valid_resistance == 0.0) | (valid_resistance == 1.0))

    def test_kernel_parity(self, test_data):
        close = test_data["close"]
        volume = test_data["volume"]
        auto = mp.vwap_zscore_with_signals(
            close, volume, length=17, upper_bottom=2.25, lower_bottom=-2.25
        )
        scalar = mp.vwap_zscore_with_signals(
            close,
            volume,
            length=17,
            upper_bottom=2.25,
            lower_bottom=-2.25,
            kernel="scalar",
        )
        for a, b in zip(auto, scalar):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_invalid_params(self, test_data):
        close = test_data["close"]
        volume = test_data["volume"]

        with pytest.raises(ValueError, match="Invalid length"):
            mp.vwap_zscore_with_signals(close, volume, length=0)

        with pytest.raises(ValueError, match="slice length mismatch"):
            mp.vwap_zscore_with_signals(close[:-1], volume, length=20)

    def test_stream_matches_batch(self, test_data):
        close = test_data["close"]
        volume = test_data["volume"]
        batch = mp.vwap_zscore_with_signals(
            close, volume, length=20, upper_bottom=2.5, lower_bottom=-2.5
        )

        stream = mp.VwapZscoreWithSignalsStream(20, 2.5, -2.5)
        out_z = []
        out_s = []
        out_r = []
        for c, v in zip(close, volume):
            out = stream.update(c, v)
            if out is None:
                out_z.append(np.nan)
                out_s.append(np.nan)
                out_r.append(np.nan)
            else:
                out_z.append(out[0])
                out_s.append(out[1])
                out_r.append(out[2])

        np.testing.assert_allclose(out_z, batch[0], rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(out_s, batch[1], rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(out_r, batch[2], rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_batch_single_param_matches_single(self, test_data):
        close = test_data["close"][:256]
        volume = test_data["volume"][:256]
        batch = mp.vwap_zscore_with_signals_batch(
            close,
            volume,
            length_range=(20, 20, 0),
            upper_bottom_range=(2.5, 2.5, 0.0),
            lower_bottom_range=(-2.5, -2.5, 0.0),
        )
        single = mp.vwap_zscore_with_signals(
            close, volume, length=20, upper_bottom=2.5, lower_bottom=-2.5
        )

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_allclose(
            batch["zvwap"][0], single[0], rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            batch["support_signal"][0], single[1], rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            batch["resistance_signal"][0], single[2], rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_metadata(self, test_data):
        close = test_data["close"][:192]
        volume = test_data["volume"][:192]
        batch = mp.vwap_zscore_with_signals_batch(
            close,
            volume,
            length_range=(18, 20, 2),
            upper_bottom_range=(2.0, 2.5, 0.5),
            lower_bottom_range=(-2.5, -2.5, 0.0),
        )

        assert batch["rows"] == 4
        assert batch["cols"] == len(close)
        assert batch["zvwap"].shape == (4, len(close))
        assert batch["support_signal"].shape == (4, len(close))
        assert batch["resistance_signal"].shape == (4, len(close))
        np.testing.assert_array_equal(batch["lengths"], np.array([18, 18, 20, 20], dtype=np.uint64))
        np.testing.assert_allclose(batch["upper_bottoms"], np.array([2.0, 2.5, 2.0, 2.5]))
        np.testing.assert_allclose(batch["lower_bottoms"], np.array([-2.5, -2.5, -2.5, -2.5]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
