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


class TestMomentumRatioOscillator:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        source = np.asarray(test_data["close"][:320], dtype=np.float64)

        line, signal = mp.momentum_ratio_oscillator(source, 50)

        assert len(line) == len(source)
        assert len(signal) == len(source)
        assert np.isnan(line[:1]).all()
        assert np.isnan(signal[:2]).all()
        assert np.isfinite(line[1:]).any()
        assert np.isfinite(signal[2:]).any()

    def test_kernel_parity(self, test_data):
        source = np.asarray(test_data["close"][:320], dtype=np.float64)

        auto_line, auto_signal = mp.momentum_ratio_oscillator(source, 30)
        scalar_line, scalar_signal = mp.momentum_ratio_oscillator(
            source, 30, kernel="scalar"
        )

        np.testing.assert_allclose(
            auto_line, scalar_line, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            auto_signal, scalar_signal, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_invalid_period(self, test_data):
        source = np.asarray(test_data["close"][:128], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid period"):
            mp.momentum_ratio_oscillator(source, 0)

    def test_stream_matches_batch(self, test_data):
        source = np.asarray(test_data["close"][:240], dtype=np.float64)

        batch_line, batch_signal = mp.momentum_ratio_oscillator(source, 35)
        stream = mp.MomentumRatioOscillatorStream(35)
        stream_line = []
        stream_signal = []

        for value in source:
            out = stream.update(float(value))
            if out is None:
                stream_line.append(np.nan)
                stream_signal.append(np.nan)
            else:
                l, s = out
                stream_line.append(l)
                stream_signal.append(s)

        np.testing.assert_allclose(
            stream_line, batch_line, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_signal, batch_signal, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        source = np.asarray(test_data["close"][:240], dtype=np.float64)

        result = mp.momentum_ratio_oscillator_batch(
            source,
            period_range=(35, 35, 0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(source)
        assert result["line"].shape == (1, len(source))
        assert result["signal"].shape == (1, len(source))
        np.testing.assert_array_equal(result["periods"], np.array([35], dtype=np.uint64))

        single_line, single_signal = mp.momentum_ratio_oscillator(source, 35)
        np.testing.assert_allclose(
            result["line"][0], single_line, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["signal"][0], single_signal, rtol=1e-10, atol=1e-10, equal_nan=True
        )
