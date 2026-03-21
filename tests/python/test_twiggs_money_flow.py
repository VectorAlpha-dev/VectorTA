import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python' first",
        allow_module_level=True,
    )


class TestTwiggsMoneyFlow:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = np.asarray(test_data["high"][:256], dtype=np.float64)
        low = np.asarray(test_data["low"][:256], dtype=np.float64)
        close = np.asarray(test_data["close"][:256], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:256], dtype=np.float64)

        tmf, smoothed = mp.twiggs_money_flow(high, low, close, volume, 5, 4, "WMA")

        assert len(tmf) == len(close)
        assert len(smoothed) == len(close)
        assert np.isnan(tmf[:5]).all()
        assert np.isnan(smoothed[:9]).all()
        assert np.isfinite(tmf[5:]).any()
        assert np.isfinite(smoothed[9:]).any()

    def test_kernel_parity(self, test_data):
        high = np.asarray(test_data["high"][:256], dtype=np.float64)
        low = np.asarray(test_data["low"][:256], dtype=np.float64)
        close = np.asarray(test_data["close"][:256], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:256], dtype=np.float64)

        auto_tmf, auto_smoothed = mp.twiggs_money_flow(high, low, close, volume, 5, 4, "EMA")
        scalar_tmf, scalar_smoothed = mp.twiggs_money_flow(
            high, low, close, volume, 5, 4, "EMA", kernel="scalar"
        )

        np.testing.assert_allclose(
            auto_tmf, scalar_tmf, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            auto_smoothed, scalar_smoothed, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_invalid_ma_type(self, test_data):
        high = np.asarray(test_data["high"][:128], dtype=np.float64)
        low = np.asarray(test_data["low"][:128], dtype=np.float64)
        close = np.asarray(test_data["close"][:128], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:128], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid MA type"):
            mp.twiggs_money_flow(high, low, close, volume, 5, 4, "BAD")

    def test_stream_matches_batch(self, test_data):
        high = np.asarray(test_data["high"][:192], dtype=np.float64)
        low = np.asarray(test_data["low"][:192], dtype=np.float64)
        close = np.asarray(test_data["close"][:192], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:192], dtype=np.float64)

        batch_tmf, batch_smoothed = mp.twiggs_money_flow(high, low, close, volume, 5, 4, "VWMA")
        stream = mp.TwiggsMoneyFlowStream(5, 4, "VWMA")
        stream_tmf = []
        stream_smoothed = []

        for h, l, c, v in zip(high, low, close, volume):
            out = stream.update(float(h), float(l), float(c), float(v))
            if out is None:
                stream_tmf.append(np.nan)
                stream_smoothed.append(np.nan)
            else:
                v0, v1 = out
                stream_tmf.append(v0)
                stream_smoothed.append(v1)

        np.testing.assert_allclose(
            stream_tmf, batch_tmf, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_smoothed, batch_smoothed, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        high = np.asarray(test_data["high"][:192], dtype=np.float64)
        low = np.asarray(test_data["low"][:192], dtype=np.float64)
        close = np.asarray(test_data["close"][:192], dtype=np.float64)
        volume = np.asarray(test_data["volume"][:192], dtype=np.float64)

        result = mp.twiggs_money_flow_batch(
            high,
            low,
            close,
            volume,
            length_range=(5, 5, 0),
            smoothing_length_range=(4, 4, 0),
            ma_type="WMA",
        )

        assert result["rows"] == 1
        assert result["cols"] == len(close)
        assert result["tmf"].shape == (1, len(close))
        assert result["smoothed"].shape == (1, len(close))
        np.testing.assert_array_equal(result["lengths"], np.array([5], dtype=np.uint64))
        np.testing.assert_array_equal(
            result["smoothing_lengths"], np.array([4], dtype=np.uint64)
        )

        single_tmf, single_smoothed = mp.twiggs_money_flow(
            high, low, close, volume, 5, 4, "WMA"
        )
        np.testing.assert_allclose(
            result["tmf"][0], single_tmf, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["smoothed"][0],
            single_smoothed,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
