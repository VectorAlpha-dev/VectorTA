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


class TestLinearRegressionIntensity:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        source = np.asarray(test_data["close"][:256], dtype=np.float64)
        values = mp.linear_regression_intensity(source, 12, 90.0, 90)

        assert len(values) == len(source)
        assert np.isfinite(values).any()

    def test_kernel_parity(self, test_data):
        source = np.asarray(test_data["close"][:256], dtype=np.float64)

        auto_values = mp.linear_regression_intensity(source, 12, 90.0, 90)
        scalar_values = mp.linear_regression_intensity(
            source, 12, 90.0, 90, kernel="scalar"
        )

        np.testing.assert_allclose(
            auto_values, scalar_values, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_stream_matches_batch(self, test_data):
        source = np.asarray(test_data["close"][:192], dtype=np.float64)

        batch = mp.linear_regression_intensity(source, 12, 90.0, 24)
        stream = mp.LinearRegressionIntensityStream(12, 90.0, 24)
        stream_values = []

        for value in source:
            out = stream.update(float(value))
            stream_values.append(np.nan if out is None else out)

        np.testing.assert_allclose(
            stream_values, batch, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        source = np.asarray(test_data["close"][:192], dtype=np.float64)

        result = mp.linear_regression_intensity_batch(
            source,
            lookback_period_range=(12, 12, 0),
            range_tolerance_range=(90.0, 90.0, 0.0),
            linreg_length_range=(24, 24, 0),
        )

        assert result["rows"] == 1
        assert result["cols"] == len(source)
        assert result["values"].shape == (1, len(source))

        single = mp.linear_regression_intensity(source, 12, 90.0, 24)
        np.testing.assert_allclose(
            result["values"][0], single, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_invalid_tolerance(self, test_data):
        source = np.asarray(test_data["close"][:64], dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid range tolerance"):
            mp.linear_regression_intensity(source, 12, 120.0, 24)
