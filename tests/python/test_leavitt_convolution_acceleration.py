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


class TestLeavittConvolutionAcceleration:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        data = np.asarray(test_data["close"][:720], dtype=np.float64)
        conv_acceleration, signal = mp.leavitt_convolution_acceleration(data, 21, 34, True)

        assert len(conv_acceleration) == len(data)
        assert len(signal) == len(data)
        assert np.isnan(conv_acceleration[:53]).all()
        assert np.isnan(signal[:53]).all()
        assert np.isfinite(conv_acceleration[53:]).any()
        assert np.isfinite(signal[53:]).any()

    def test_kernel_parity(self, test_data):
        data = np.asarray(test_data["close"][:512], dtype=np.float64)
        auto_conv, auto_signal = mp.leavitt_convolution_acceleration(data, 21, 34, False)
        scalar_conv, scalar_signal = mp.leavitt_convolution_acceleration(
            data, 21, 34, False, kernel="scalar"
        )

        np.testing.assert_allclose(
            auto_conv, scalar_conv, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            auto_signal, scalar_signal, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_invalid_norm_length(self, test_data):
        data = np.asarray(test_data["close"][:128], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid norm_length"):
            mp.leavitt_convolution_acceleration(data, 21, 0, True)

    def test_stream_matches_batch(self, test_data):
        data = np.asarray(test_data["close"][:320], dtype=np.float64)
        batch_conv, batch_signal = mp.leavitt_convolution_acceleration(data, 20, 25, True)
        stream = mp.LeavittConvolutionAccelerationStream(20, 25, True)
        stream_conv = []
        stream_signal = []

        for value in data:
            out = stream.update(float(value))
            if out is None:
                stream_conv.append(np.nan)
                stream_signal.append(np.nan)
            else:
                a, b = out
                stream_conv.append(a)
                stream_signal.append(b)

        np.testing.assert_allclose(
            stream_conv, batch_conv, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            stream_signal, batch_signal, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        data = np.asarray(test_data["close"][:320], dtype=np.float64)
        result = mp.leavitt_convolution_acceleration_batch(
            data,
            length_range=(21, 21, 0),
            norm_length_range=(34, 34, 0),
            use_norm_hyperbolic=False,
        )

        assert result["rows"] == 1
        assert result["cols"] == len(data)
        assert result["conv_acceleration"].shape == (1, len(data))
        assert result["signal"].shape == (1, len(data))
        np.testing.assert_array_equal(result["lengths"], np.array([21], dtype=np.uint64))
        np.testing.assert_array_equal(result["norm_lengths"], np.array([34], dtype=np.uint64))
        assert result["use_norm_hyperbolic"] is False

        single_conv, single_signal = mp.leavitt_convolution_acceleration(data, 21, 34, False)
        np.testing.assert_allclose(
            result["conv_acceleration"][0],
            single_conv,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            result["signal"][0],
            single_signal,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
