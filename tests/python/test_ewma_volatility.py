import math

import numpy as np
import pytest

try:
    import vector_ta as mp
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python' first",
        allow_module_level=True,
    )


def geometric_series(length: int, start: float, ratio: float) -> np.ndarray:
    out = np.empty(length, dtype=np.float64)
    value = start
    for i in range(length):
        out[i] = value
        value *= ratio
    return out


def pine_period(lambda_: float) -> int:
    return max(1, int(round(2.0 / (1.0 - lambda_) - 1.0)))


def manual_ewma_volatility(data: np.ndarray, lambda_: float) -> np.ndarray:
    period = pine_period(lambda_)
    alpha = 2.0 / (period + 1.0)
    out = np.full(data.shape[0], np.nan, dtype=np.float64)
    sq_returns = []
    seed_idx = None

    for i in range(1, data.shape[0]):
        prev = data[i - 1]
        curr = data[i]
        if np.isfinite(prev) and np.isfinite(curr) and prev > 0.0 and curr > 0.0:
            ret = math.log(curr / prev)
            sq = ret * ret
            sq_returns.append((i, sq))
            if len(sq_returns) == period and seed_idx is None:
                seed_idx = i
                ema = sum(v for _, v in sq_returns[:period]) / period
                out[i] = math.sqrt(max(ema, 0.0)) * 100.0
            elif seed_idx is not None:
                ema = (1.0 - alpha) * ema + alpha * sq
                out[i] = math.sqrt(max(ema, 0.0)) * 100.0
        elif seed_idx is not None:
            out[i] = out[i - 1]

    if seed_idx is not None:
        for i in range(seed_idx + 1, data.shape[0]):
            if np.isnan(out[i]):
                out[i] = out[i - 1]
    return out


class TestEwmaVolatility:
    def test_constant_return_series_matches_expected_level(self):
        data = geometric_series(128, 100.0, 1.01)
        result = mp.ewma_volatility(data, lambda_=0.94)
        expected = abs(math.log(1.01)) * 100.0
        period = pine_period(0.94)

        assert np.all(np.isnan(result[:period]))
        np.testing.assert_allclose(
            result[period:],
            np.full(data.shape[0] - period, expected),
            rtol=0.0,
            atol=1e-12,
        )

    def test_manual_reference_matches_binding(self):
        data = geometric_series(96, 25.0, 1.003)
        data[12] = np.nan
        data[13] = 0.0
        expected = manual_ewma_volatility(data, 0.90)
        result = mp.ewma_volatility(data, lambda_=0.90)
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_stream_matches_batch(self):
        data = geometric_series(96, 50.0, 1.005)
        batch = mp.ewma_volatility(data, lambda_=0.90)
        streamed = []
        stream = mp.EwmaVolatilityStream(0.90)
        for value in data:
            out = stream.update(float(value))
            streamed.append(np.nan if out is None else out)
        np.testing.assert_allclose(
            np.array(streamed),
            batch,
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )

    def test_batch_metadata_and_first_row_match_single(self):
        data = geometric_series(128, 100.0, 1.002)
        result = mp.ewma_volatility_batch(data, lambda_range=(0.90, 0.94, 0.02))
        assert result["rows"] == 3
        assert result["cols"] == data.shape[0]
        np.testing.assert_allclose(result["lambdas"], np.array([0.90, 0.92, 0.94]))

        single = mp.ewma_volatility(data, lambda_=0.90)
        np.testing.assert_allclose(
            result["values"][0],
            single,
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )

    def test_invalid_lambda(self):
        data = geometric_series(32, 10.0, 1.01)
        with pytest.raises(ValueError, match="Invalid lambda"):
            mp.ewma_volatility(data, lambda_=1.0)

    def test_kernel_parity(self):
        data = geometric_series(96, 30.0, 1.004)
        auto = mp.ewma_volatility(data, lambda_=0.94)
        scalar = mp.ewma_volatility(data, lambda_=0.94, kernel="scalar")
        np.testing.assert_allclose(auto, scalar, rtol=0.0, atol=1e-12, equal_nan=True)
