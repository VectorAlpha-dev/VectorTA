import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as ta
except ImportError:
    try:
        import my_project as ta
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


OUTPUT_KEYS = [
    "basis",
    "upper_band",
    "lower_band",
    "structure_direction",
    "bullish_arrow",
    "bearish_arrow",
    "bullish_change",
    "bearish_change",
    "hh",
    "lh",
    "hl",
    "ll",
    "bullish_bos",
    "bullish_choch",
    "bearish_bos",
    "bearish_choch",
]


class TestMarketStructureConfluence:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_binding_returns_expected_outputs(self, test_data):
        high = np.asarray(test_data["high"][:420], dtype=np.float64)
        low = np.asarray(test_data["low"][:420], dtype=np.float64)
        close = np.asarray(test_data["close"][:420], dtype=np.float64)
        result = ta.market_structure_confluence(high, low, close)
        assert list(result.keys()) == OUTPUT_KEYS
        for key in OUTPUT_KEYS:
            values = np.asarray(result[key], dtype=np.float64)
            assert values.shape == close.shape
        assert np.isfinite(np.asarray(result["basis"], dtype=np.float64)).any()

    def test_stream_matches_batch(self, test_data):
        high = np.asarray(test_data["high"][:420], dtype=np.float64)
        low = np.asarray(test_data["low"][:420], dtype=np.float64)
        close = np.asarray(test_data["close"][:420], dtype=np.float64)
        params = {
            "swing_size": 10,
            "bos_confirmation": "Candle Close",
            "basis_length": 100,
            "atr_length": 14,
            "atr_smooth": 21,
            "vol_mult": 2.0,
        }
        batch = ta.market_structure_confluence(high, low, close, **params)
        stream = ta.MarketStructureConfluenceStream(**params)
        collected = {key: [] for key in OUTPUT_KEYS}
        for h, l, c in zip(high, low, close):
            point = stream.update(float(h), float(l), float(c))
            if point is None:
                for key in OUTPUT_KEYS:
                    collected[key].append(np.nan)
            else:
                for idx, key in enumerate(OUTPUT_KEYS):
                    collected[key].append(point[idx])
        for key in OUTPUT_KEYS:
            np.testing.assert_allclose(
                np.asarray(collected[key], dtype=np.float64),
                np.asarray(batch[key], dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )

    def test_batch_first_row_matches_single(self, test_data):
        high = np.asarray(test_data["high"][:420], dtype=np.float64)
        low = np.asarray(test_data["low"][:420], dtype=np.float64)
        close = np.asarray(test_data["close"][:420], dtype=np.float64)
        single = ta.market_structure_confluence(high, low, close)
        batch = ta.market_structure_confluence_batch(
            high,
            low,
            close,
            swing_size_range=(10, 12, 2),
            bos_confirmation_options=["Candle Close"],
            basis_length_range=(100, 100, 0),
            atr_length_range=(14, 14, 0),
            atr_smooth_range=(21, 21, 0),
            vol_mult_range=(2.0, 2.0, 0.0),
        )
        assert batch["rows"] == 2
        assert batch["cols"] == close.shape[0]
        np.testing.assert_array_equal(batch["swing_sizes"], np.array([10, 12], dtype=np.uint64))
        for key in OUTPUT_KEYS:
            np.testing.assert_allclose(
                np.asarray(batch[key][0], dtype=np.float64),
                np.asarray(single[key], dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )

    def test_invalid_params_raise(self, test_data):
        high = np.asarray(test_data["high"][:256], dtype=np.float64)
        low = np.asarray(test_data["low"][:256], dtype=np.float64)
        close = np.asarray(test_data["close"][:256], dtype=np.float64)
        with pytest.raises(Exception, match="invalid swing_size"):
            ta.market_structure_confluence(high, low, close, swing_size=1)
        with pytest.raises(Exception, match="invalid bos_confirmation"):
            ta.market_structure_confluence(high, low, close, bos_confirmation="bad")
