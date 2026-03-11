import numpy as np
import pytest

from test_utils import load_test_data

try:
    import my_project as mp
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python' first",
        allow_module_level=True,
    )


class TestPrettyGoodOscillator:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        high = np.asarray(test_data["high"][:256], dtype=np.float64)
        low = np.asarray(test_data["low"][:256], dtype=np.float64)
        close = np.asarray(test_data["close"][:256], dtype=np.float64)
        source = (high + low) / 2.0

        values = mp.pretty_good_oscillator(high, low, close, source, 5)

        assert len(values) == len(close)
        assert np.isnan(values[:4]).all()
        assert np.isfinite(values[4:]).any()

    def test_kernel_parity(self, test_data):
        high = np.asarray(test_data["high"][:256], dtype=np.float64)
        low = np.asarray(test_data["low"][:256], dtype=np.float64)
        close = np.asarray(test_data["close"][:256], dtype=np.float64)
        source = np.asarray(test_data["close"][:256], dtype=np.float64)

        auto_values = mp.pretty_good_oscillator(high, low, close, source, 14)
        scalar_values = mp.pretty_good_oscillator(
            high, low, close, source, 14, kernel="scalar"
        )

        np.testing.assert_allclose(
            auto_values, scalar_values, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_invalid_length(self, test_data):
        high = np.asarray(test_data["high"][:64], dtype=np.float64)
        low = np.asarray(test_data["low"][:64], dtype=np.float64)
        close = np.asarray(test_data["close"][:64], dtype=np.float64)
        source = np.asarray(test_data["close"][:64], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid length"):
            mp.pretty_good_oscillator(high, low, close, source, 0)

    def test_stream_matches_batch(self, test_data):
        high = np.asarray(test_data["high"][:192], dtype=np.float64)
        low = np.asarray(test_data["low"][:192], dtype=np.float64)
        close = np.asarray(test_data["close"][:192], dtype=np.float64)
        source = (high + low) / 2.0

        batch = mp.pretty_good_oscillator(high, low, close, source, 14)
        stream = mp.PrettyGoodOscillatorStream(14)
        stream_values = []

        for h, l, c, s in zip(high, low, close, source):
            out = stream.update(float(h), float(l), float(c), float(s))
            stream_values.append(np.nan if out is None else out)

        np.testing.assert_allclose(
            stream_values, batch, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        high = np.asarray(test_data["high"][:192], dtype=np.float64)
        low = np.asarray(test_data["low"][:192], dtype=np.float64)
        close = np.asarray(test_data["close"][:192], dtype=np.float64)
        source = np.asarray(test_data["close"][:192], dtype=np.float64)

        result = mp.pretty_good_oscillator_batch(
            high,
            low,
            close,
            source,
            length_range=(14, 14, 0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(close)
        assert result["values"].shape == (1, len(close))
        np.testing.assert_array_equal(result["lengths"], np.array([14], dtype=np.uint64))

        single = mp.pretty_good_oscillator(high, low, close, source, 14)
        np.testing.assert_allclose(
            result["values"][0], single, rtol=1e-10, atol=1e-10, equal_nan=True
        )
