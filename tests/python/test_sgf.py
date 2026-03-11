import numpy as np
import pytest

from my_project import sgf, sgf_batch, SgfStream
from test_utils import assert_close


def poly_series(length, coeffs, warm_prefix=0):
    out = np.full(length, np.nan, dtype=np.float64)
    for i in range(warm_prefix, length):
        x = float(i)
        power = 1.0
        value = 0.0
        for coef in coeffs:
            value += coef * power
            power *= x
        out[i] = value
    return out


class TestSgf:
    def test_sgf_quadratic_exactness(self):
        data = poly_series(64, [2.0, -0.5, 0.25], warm_prefix=3)
        result = sgf(data, 9, 2)
        assert np.isnan(result[:11]).all()
        assert_close(result[11:], data[11:], rtol=0.0, atol=1e-10)

    def test_sgf_even_period_uses_odd_effective_window(self):
        data = poly_series(64, [1.0, 0.2, -0.01], warm_prefix=0)
        direct = sgf(data, 9, 2)
        even = sgf(data, 10, 2)
        assert_close(even, direct, rtol=0.0, atol=1e-10)

    def test_sgf_invalid_poly_order(self):
        data = poly_series(32, [1.0, 2.0], warm_prefix=0)
        with pytest.raises(ValueError, match="invalid polynomial order"):
            sgf(data, 5, 5)

    def test_sgf_stream_matches_batch(self):
        data = poly_series(48, [3.0, 0.2, -0.03], warm_prefix=1)
        batch = sgf(data, 9, 2)
        stream = SgfStream(9, 2)
        streamed = np.full_like(batch, np.nan)
        for idx, value in enumerate(data):
            if np.isnan(value):
                continue
            result = stream.update(float(value))
            if result is not None:
                streamed[idx] = result
        assert_close(streamed, batch, rtol=0.0, atol=1e-10)

    def test_sgf_batch(self):
        data = poly_series(80, [1.0, 0.3, -0.02], warm_prefix=2)
        batch = sgf_batch(data, (7, 11, 2), (2, 2, 0))
        assert batch["values"].shape == (3, len(data))
        assert np.array_equal(batch["periods"], np.array([7, 9, 11]))
        assert np.array_equal(batch["poly_orders"], np.array([2, 2, 2]))
        for row, period in enumerate((7, 9, 11)):
            assert_close(batch["values"][row], sgf(data, period, 2), rtol=0.0, atol=1e-10)
