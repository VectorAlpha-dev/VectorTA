import pytest
import numpy as np

try:
    import vector_ta as ta_indicators
except ImportError:
    try:
        import vector_ta as ta_indicators
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )

from test_utils import load_test_data, assert_close


def hl2_series(test_data):
    return (test_data["high"] + test_data["low"]) * 0.5


class TestEhlersAdaptiveCg:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_basic_output_shape(self, test_data):
        data = hl2_series(test_data)
        cg, trigger = ta_indicators.ehlers_adaptive_cg(data)

        assert len(cg) == len(data)
        assert len(trigger) == len(data)
        assert np.any(np.isfinite(cg[64:]))
        assert np.any(np.isfinite(trigger[64:]))

    def test_stream_matches_batch(self, test_data):
        data = hl2_series(test_data)
        batch_cg, batch_trigger = ta_indicators.ehlers_adaptive_cg(data, alpha=0.07)

        stream = ta_indicators.EhlersAdaptiveCgStream(alpha=0.07)
        cg = np.full(len(data), np.nan)
        trigger = np.full(len(data), np.nan)

        for idx, value in enumerate(data):
            result = stream.update(float(value))
            if result is not None:
                cg[idx], trigger[idx] = result

        np.testing.assert_allclose(batch_cg, cg, rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(
            batch_trigger, trigger, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_matches_single(self, test_data):
        data = hl2_series(test_data)
        single_cg, single_trigger = ta_indicators.ehlers_adaptive_cg(data, alpha=0.07)
        batch = ta_indicators.ehlers_adaptive_cg_batch(
            data,
            alpha_range=(0.07, 0.07, 0.0),
        )

        assert batch["cg"].shape == (1, len(data))
        assert batch["trigger"].shape == (1, len(data))
        assert list(batch["alphas"]) == [0.07]
        assert_close(batch["cg"][0], single_cg, rtol=1e-10, atol=1e-10)
        assert_close(batch["trigger"][0], single_trigger, rtol=1e-10, atol=1e-10)

    def test_errors(self):
        with pytest.raises(ValueError, match="invalid alpha|Invalid alpha"):
            ta_indicators.ehlers_adaptive_cg(np.ones(32), alpha=0.0)

        with pytest.raises(ValueError, match="empty|Empty"):
            ta_indicators.ehlers_adaptive_cg(np.array([]))

        with pytest.raises(ValueError, match="all values are NaN|All values are NaN"):
            ta_indicators.ehlers_adaptive_cg(np.full(32, np.nan))

        with pytest.raises(ValueError, match="not enough valid data|Not enough valid data"):
            ta_indicators.ehlers_adaptive_cg(np.arange(10, dtype=float))
