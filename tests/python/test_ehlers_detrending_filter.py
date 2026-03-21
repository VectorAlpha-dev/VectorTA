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


class TestEhlersDetrendingFilter:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        data = np.asarray(test_data["close"][:320], dtype=np.float64)
        edf, signal = mp.ehlers_detrending_filter(data, 10)

        assert len(edf) == len(data)
        assert len(signal) == len(data)
        assert np.isnan(edf[:9]).all()
        assert np.isnan(signal[:9]).all()
        assert np.isfinite(edf[9:]).any()
        assert np.isfinite(signal[9:]).any()

    def test_kernel_parity(self, test_data):
        data = np.asarray(test_data["close"][:320], dtype=np.float64)
        auto_edf, auto_signal = mp.ehlers_detrending_filter(data, 12)
        scalar_edf, scalar_signal = mp.ehlers_detrending_filter(data, 12, kernel="scalar")

        np.testing.assert_allclose(auto_edf, scalar_edf, rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(
            auto_signal, scalar_signal, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_invalid_length(self, test_data):
        data = np.asarray(test_data["close"][:64], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid length"):
            mp.ehlers_detrending_filter(data, 0)

    def test_stream_matches_batch(self, test_data):
        data = np.asarray(test_data["close"][:320], dtype=np.float64)
        batch_edf, batch_signal = mp.ehlers_detrending_filter(data, 14)
        stream = mp.EhlersDetrendingFilterStream(14)
        stream_edf = []
        stream_signal = []

        for value in data:
            out = stream.update(float(value))
            if out is None:
                stream_edf.append(np.nan)
                stream_signal.append(np.nan)
            else:
                a, b = out
                stream_edf.append(a)
                stream_signal.append(b)

        np.testing.assert_allclose(
            stream_edf, batch_edf, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_signal, batch_signal, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        data = np.asarray(test_data["close"][:320], dtype=np.float64)
        result = mp.ehlers_detrending_filter_batch(data, length_range=(11, 11, 0))

        assert result["rows"] == 1
        assert result["cols"] == len(data)
        assert result["edf"].shape == (1, len(data))
        assert result["signal"].shape == (1, len(data))
        np.testing.assert_array_equal(result["lengths"], np.array([11], dtype=np.uint64))

        single_edf, single_signal = mp.ehlers_detrending_filter(data, 11)
        np.testing.assert_allclose(
            result["edf"][0], single_edf, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            result["signal"][0], single_signal, rtol=1e-10, atol=1e-10, equal_nan=True
        )
