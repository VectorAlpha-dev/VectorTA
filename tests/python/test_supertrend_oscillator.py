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


class TestSuperTrendOscillator:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = np.asarray(test_data["high"][:320], dtype=np.float64)
        low = np.asarray(test_data["low"][:320], dtype=np.float64)
        source = np.asarray(test_data["close"][:320], dtype=np.float64)

        oscillator, signal, histogram = mp.supertrend_oscillator(
            high, low, source, 10, 2.0, 72
        )

        assert len(oscillator) == len(source)
        assert len(signal) == len(source)
        assert len(histogram) == len(source)
        assert np.isnan(oscillator[:9]).all()
        assert np.isnan(signal[:9]).all()
        assert np.isnan(histogram[:9]).all()
        assert np.isfinite(oscillator[9:]).any()
        assert np.isfinite(signal[9:]).any()
        assert np.isfinite(histogram[9:]).any()

    def test_kernel_parity(self, test_data):
        high = np.asarray(test_data["high"][:256], dtype=np.float64)
        low = np.asarray(test_data["low"][:256], dtype=np.float64)
        source = np.asarray(test_data["close"][:256], dtype=np.float64)

        auto = mp.supertrend_oscillator(high, low, source, 12, 2.5, 20)
        scalar = mp.supertrend_oscillator(
            high, low, source, 12, 2.5, 20, kernel="scalar"
        )

        for lhs, rhs in zip(auto, scalar):
            np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_invalid_params(self, test_data):
        high = np.asarray(test_data["high"][:128], dtype=np.float64)
        low = np.asarray(test_data["low"][:128], dtype=np.float64)
        source = np.asarray(test_data["close"][:128], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid length"):
            mp.supertrend_oscillator(high, low, source, 0, 2.0, 20)

        with pytest.raises(ValueError, match="Invalid multiplier"):
            mp.supertrend_oscillator(high, low, source, 10, 0.0, 20)

        with pytest.raises(ValueError, match="Invalid smooth"):
            mp.supertrend_oscillator(high, low, source, 10, 2.0, 0)

    def test_stream_matches_batch(self, test_data):
        high = np.asarray(test_data["high"][:240], dtype=np.float64)
        low = np.asarray(test_data["low"][:240], dtype=np.float64)
        source = np.asarray(test_data["close"][:240], dtype=np.float64)

        batch = mp.supertrend_oscillator(high, low, source, 12, 1.75, 18)
        stream = mp.SuperTrendOscillatorStream(12, 1.75, 18)

        oscillator = []
        signal = []
        histogram = []
        for h, l, s in zip(high, low, source):
            out = stream.update(float(h), float(l), float(s))
            if out is None:
                oscillator.append(np.nan)
                signal.append(np.nan)
                histogram.append(np.nan)
            else:
                osc, sig, hist = out
                oscillator.append(osc)
                signal.append(sig)
                histogram.append(hist)

        np.testing.assert_allclose(
            oscillator, batch[0], rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            signal, batch[1], rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            histogram, batch[2], rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        high = np.asarray(test_data["high"][:240], dtype=np.float64)
        low = np.asarray(test_data["low"][:240], dtype=np.float64)
        source = np.asarray(test_data["close"][:240], dtype=np.float64)

        result = mp.supertrend_oscillator_batch(
            high,
            low,
            source,
            length_range=(12, 12, 0),
            mult_range=(2.5, 2.5, 0.0),
            smooth_range=(18, 18, 0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(source)
        assert result["oscillator"].shape == (1, len(source))
        assert result["signal"].shape == (1, len(source))
        assert result["histogram"].shape == (1, len(source))
        np.testing.assert_array_equal(result["lengths"], np.array([12], dtype=np.uint64))
        np.testing.assert_allclose(result["mults"], np.array([2.5]))
        np.testing.assert_array_equal(result["smooths"], np.array([18], dtype=np.uint64))

        single = mp.supertrend_oscillator(high, low, source, 12, 2.5, 18)
        np.testing.assert_allclose(
            result["oscillator"][0], single[0], rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["signal"][0], single[1], rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["histogram"][0], single[2], rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_metadata(self, test_data):
        high = np.asarray(test_data["high"][:180], dtype=np.float64)
        low = np.asarray(test_data["low"][:180], dtype=np.float64)
        source = np.asarray(test_data["close"][:180], dtype=np.float64)

        result = mp.supertrend_oscillator_batch(
            high,
            low,
            source,
            length_range=(10, 12, 2),
            mult_range=(1.5, 2.5, 0.5),
            smooth_range=(18, 18, 0),
        )

        assert result["rows"] == 6
        assert result["cols"] == len(source)
        np.testing.assert_array_equal(
            result["lengths"], np.array([10, 10, 10, 12, 12, 12], dtype=np.uint64)
        )
        np.testing.assert_allclose(result["mults"], np.array([1.5, 2.0, 2.5, 1.5, 2.0, 2.5]))
        np.testing.assert_array_equal(
            result["smooths"], np.array([18, 18, 18, 18, 18, 18], dtype=np.uint64)
        )
