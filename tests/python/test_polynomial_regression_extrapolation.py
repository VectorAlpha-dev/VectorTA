import numpy as np
import pytest

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

from test_utils import assert_close


def quadratic_data(size):
    return np.array([float(i * i) for i in range(size)], dtype=np.float64)


def cubic_data(size):
    values = []
    for i in range(size):
        x = float(i)
        values.append(x * x * x - 2.0 * x * x + 3.0 * x + 1.0)
    return np.array(values, dtype=np.float64)


def test_polynomial_regression_extrapolation_quadratic_exactness():
    data = quadratic_data(24)
    result = ta_indicators.polynomial_regression_extrapolation(
        data, length=5, extrapolate=2, degree=2
    )

    expected = np.full(data.shape, np.nan, dtype=np.float64)
    for idx in range(4, data.size):
        x = float(idx + 2)
        expected[idx] = x * x

    assert_close(result, expected, atol=1e-9, msg="quadratic extrapolation mismatch")


def test_polynomial_regression_extrapolation_nan_gap_semantics():
    data = np.array(
        [np.nan, 1.0, 4.0, 9.0, 16.0, 25.0, np.nan, 64.0, 81.0, 100.0, 121.0, 144.0],
        dtype=np.float64,
    )
    result = ta_indicators.polynomial_regression_extrapolation(
        data, length=3, extrapolate=1, degree=2
    )

    assert np.all(np.isnan(result[:3]))
    assert_close(result[3], 16.0, atol=1e-9)
    assert_close(result[4], 25.0, atol=1e-9)
    assert_close(result[5], 36.0, atol=1e-9)
    assert np.all(np.isnan(result[6:9]))
    assert_close(result[9], 121.0, atol=1e-9)
    assert_close(result[10], 144.0, atol=1e-9)
    assert_close(result[11], 169.0, atol=1e-9)


def test_polynomial_regression_extrapolation_invalid_params():
    data = quadratic_data(16)

    with pytest.raises(ValueError, match="Invalid degree"):
        ta_indicators.polynomial_regression_extrapolation(
            data, length=4, extrapolate=1, degree=9
        )

    with pytest.raises(ValueError, match="Degree exceeds length"):
        ta_indicators.polynomial_regression_extrapolation(
            data, length=3, extrapolate=1, degree=3
        )


def test_polynomial_regression_extrapolation_stream_matches_batch():
    data = cubic_data(24)
    batch = ta_indicators.polynomial_regression_extrapolation(
        data, length=6, extrapolate=2, degree=3
    )

    stream = ta_indicators.PolynomialRegressionExtrapolationStream(
        length=6, extrapolate=2, degree=3
    )
    streamed = []
    for value in data:
        current = stream.update(float(value))
        streamed.append(np.nan if current is None else current)
    streamed = np.array(streamed, dtype=np.float64)

    assert_close(streamed, batch, atol=1e-12, msg="stream output mismatch")


def test_polynomial_regression_extrapolation_batch_matches_single():
    data = quadratic_data(22)
    batch = ta_indicators.polynomial_regression_extrapolation_batch(
        data,
        length_range=(4, 5, 1),
        extrapolate_range=(1, 2, 1),
        degree_range=(1, 2, 1),
    )

    assert batch["values"].shape == (8, data.size)
    assert batch["rows"] == 8
    assert batch["cols"] == data.size

    row = 0
    for length in [4, 5]:
        for extrapolate in [1, 2]:
            for degree in [1, 2]:
                single = ta_indicators.polynomial_regression_extrapolation(
                    data, length=length, extrapolate=extrapolate, degree=degree
                )
                assert batch["lengths"][row] == length
                assert batch["extrapolates"][row] == extrapolate
                assert batch["degrees"][row] == degree
                assert_close(
                    batch["values"][row],
                    single,
                    atol=1e-12,
                    msg=f"batch row mismatch at row {row}",
                )
                row += 1
