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


def manual_vqi(open_, high, low, close, fast_length, slow_length):
    n = len(close)
    vqi_sum = np.zeros(n, dtype=np.float64)
    fast_sma = np.full(n, np.nan, dtype=np.float64)
    slow_sma = np.full(n, np.nan, dtype=np.float64)

    prev_close = np.nan
    prev_vqi_t = 0.0
    cumulative = 0.0

    for i in range(n):
        rng = high[i] - low[i]
        if np.isfinite(high[i]) and np.isfinite(low[i]):
            if np.isfinite(prev_close):
                tr = max(abs(rng), abs(high[i] - prev_close), abs(low[i] - prev_close))
            else:
                tr = abs(rng)
        else:
            tr = np.nan

        if (
            np.isfinite(prev_close)
            and np.isfinite(open_[i])
            and np.isfinite(high[i])
            and np.isfinite(low[i])
            and np.isfinite(close[i])
            and np.isfinite(tr)
            and tr != 0.0
            and np.isfinite(rng)
            and rng != 0.0
        ):
            vqi_t = 0.5 * (((close[i] - prev_close) / tr) + ((close[i] - open_[i]) / rng))
        else:
            vqi_t = prev_vqi_t

        if np.isfinite(prev_close) and np.isfinite(open_[i]) and np.isfinite(close[i]):
            raw = abs(vqi_t) * 0.5 * ((close[i] - prev_close) + (close[i] - open_[i]))
        else:
            raw = 0.0

        cumulative += raw
        vqi_sum[i] = cumulative
        prev_vqi_t = vqi_t
        if np.isfinite(close[i]):
            prev_close = close[i]

    for i in range(fast_length - 1, n):
        fast_sma[i] = np.mean(vqi_sum[i - fast_length + 1 : i + 1])
    for i in range(slow_length - 1, n):
        slow_sma[i] = np.mean(vqi_sum[i - slow_length + 1 : i + 1])

    return vqi_sum, fast_sma, slow_sma


class TestVolatilityQualityIndex:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract_and_manual_reference(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]

        vqi_sum, fast_sma, slow_sma = mp.volatility_quality_index(
            open_,
            high,
            low,
            close,
            fast_length=9,
            slow_length=21,
        )
        expected = manual_vqi(open_, high, low, close, 9, 21)

        np.testing.assert_allclose(vqi_sum, expected[0], rtol=0.0, atol=5e-12, equal_nan=True)
        np.testing.assert_allclose(fast_sma, expected[1], rtol=0.0, atol=5e-12, equal_nan=True)
        np.testing.assert_allclose(slow_sma, expected[2], rtol=0.0, atol=5e-12, equal_nan=True)

    def test_stream_matches_batch(self, test_data):
        open_ = test_data["open"][:128]
        high = test_data["high"][:128]
        low = test_data["low"][:128]
        close = test_data["close"][:128]

        batch = mp.volatility_quality_index(open_, high, low, close, fast_length=9, slow_length=21)
        stream = mp.VolatilityQualityIndexStream(9, 21)
        streamed = [stream.update(o, h, l, c) for o, h, l, c in zip(open_, high, low, close)]

        vqi = np.array([x[0] for x in streamed], dtype=np.float64)
        fast = np.array([x[1] for x in streamed], dtype=np.float64)
        slow = np.array([x[2] for x in streamed], dtype=np.float64)

        np.testing.assert_allclose(vqi, batch[0], rtol=0.0, atol=5e-12, equal_nan=True)
        np.testing.assert_allclose(fast, batch[1], rtol=0.0, atol=5e-12, equal_nan=True)
        np.testing.assert_allclose(slow, batch[2], rtol=0.0, atol=5e-12, equal_nan=True)

    def test_batch_metadata_and_first_row(self, test_data):
        open_ = test_data["open"][:128]
        high = test_data["high"][:128]
        low = test_data["low"][:128]
        close = test_data["close"][:128]

        result = mp.volatility_quality_index_batch(
            open_,
            high,
            low,
            close,
            fast_length_range=(9, 11, 2),
            slow_length_range=(20, 20, 0),
        )

        assert result["rows"] == 2
        assert result["cols"] == len(close)
        np.testing.assert_array_equal(result["fast_lengths"], np.array([9, 11], dtype=np.uint64))
        np.testing.assert_array_equal(result["slow_lengths"], np.array([20, 20], dtype=np.uint64))

        single = mp.volatility_quality_index(open_, high, low, close, fast_length=9, slow_length=20)
        np.testing.assert_allclose(result["vqi_sum"][0], single[0], rtol=0.0, atol=5e-12, equal_nan=True)
        np.testing.assert_allclose(result["fast_sma"][0], single[1], rtol=0.0, atol=5e-12, equal_nan=True)
        np.testing.assert_allclose(result["slow_sma"][0], single[2], rtol=0.0, atol=5e-12, equal_nan=True)

    def test_kernel_parity(self, test_data):
        open_ = test_data["open"][:128]
        high = test_data["high"][:128]
        low = test_data["low"][:128]
        close = test_data["close"][:128]

        auto = mp.volatility_quality_index(open_, high, low, close, fast_length=9, slow_length=21)
        scalar = mp.volatility_quality_index(
            open_,
            high,
            low,
            close,
            fast_length=9,
            slow_length=21,
            kernel="scalar",
        )

        for a, b in zip(auto, scalar):
            np.testing.assert_allclose(a, b, rtol=0.0, atol=5e-12, equal_nan=True)

    def test_invalid_length(self, test_data):
        open_ = test_data["open"][:64]
        high = test_data["high"][:64]
        low = test_data["low"][:64]
        close = test_data["close"][:64]

        with pytest.raises(ValueError, match="Invalid fast_length"):
            mp.volatility_quality_index(open_, high, low, close, fast_length=0, slow_length=21)

    def test_length_mismatch(self, test_data):
        open_ = test_data["open"][:64]
        high = test_data["high"][:64]
        low = test_data["low"][:64]
        close = test_data["close"][:63]

        with pytest.raises(ValueError, match="length mismatch"):
            mp.volatility_quality_index(open_, high, low, close, fast_length=9, slow_length=21)
