import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    try:
        import vector_ta as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


def manual_vwma(price: np.ndarray, volume: np.ndarray, length: int) -> np.ndarray:
    out = np.full(price.shape[0], np.nan, dtype=np.float64)
    for i in range(length - 1, price.shape[0]):
        p = price[i + 1 - length : i + 1]
        v = volume[i + 1 - length : i + 1]
        if np.all(np.isfinite(p)) and np.all(np.isfinite(v)):
            denom = np.sum(v)
            if denom != 0.0:
                out[i] = np.sum(p * v) / denom
    return out


def manual_sma(values: np.ndarray, length: int) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for i in range(length - 1, values.shape[0]):
        window = values[i + 1 - length : i + 1]
        if np.all(np.isfinite(window)):
            out[i] = np.mean(window)
    return out


def manual_pmarp(
    price: np.ndarray,
    volume: np.ndarray,
    ma_length: int = 20,
    pmarp_lookback: int = 350,
    signal_ma_length: int = 20,
):
    ma = manual_vwma(price, volume, ma_length)
    pmar = np.full(price.shape[0], np.nan, dtype=np.float64)
    pmarp = np.full(price.shape[0], np.nan, dtype=np.float64)
    pmar_high = np.full(price.shape[0], np.nan, dtype=np.float64)
    pmar_low = np.full(price.shape[0], np.nan, dtype=np.float64)
    scaled_pmar = np.full(price.shape[0], np.nan, dtype=np.float64)

    hi = 1.0
    lo = 1.0
    seen = False
    for i in range(price.shape[0]):
        if np.isfinite(price[i]) and np.isfinite(ma[i]) and ma[i] != 0.0:
            pmar[i] = price[i] / ma[i]
            hi = max(hi, pmar[i])
            lo = min(lo, pmar[i])
            seen = True
        if seen:
            pmar_high[i] = hi
            pmar_low[i] = lo
            if np.isfinite(pmar[i]):
                if pmar[i] >= 1.0:
                    scaled_pmar[i] = 50.0 if hi == 1.0 else (((pmar[i] - 1.0) * (100.0 / (hi - 1.0))) / 2.0) + 50.0
                else:
                    scaled_pmar[i] = 50.0 if lo == 1.0 else ((pmar[i] - lo) * (100.0 / (1.0 - lo))) / 2.0

    for i in range(ma_length, price.shape[0]):
        if not np.isfinite(pmar[i]):
            continue
        current = abs(pmar[i])
        lookback = min(i, pmarp_lookback)
        if lookback == 0:
            continue
        count = 0
        for offset in range(1, lookback + 1):
            prev = abs(pmar[i - offset])
            if not (np.isfinite(prev) and prev > current):
                count += 1
        pmarp[i] = (count / lookback) * 100.0

    plotline = pmarp.copy()
    signal = manual_sma(plotline, signal_ma_length)
    return {
        "pmar": pmar,
        "pmarp": pmarp,
        "plotline": plotline,
        "signal": signal,
        "pmar_high": pmar_high,
        "pmar_low": pmar_low,
        "scaled_pmar": scaled_pmar,
    }


class TestPriceMovingAverageRatioPercentile:
    def test_manual_reference_matches_binding(self):
        data = load_test_data()
        price = np.asarray(data["close"][:160], dtype=np.float64)
        volume = np.asarray(data["volume"][:160], dtype=np.float64)
        expected = manual_pmarp(price, volume, 20, 60, 10)
        result = mp.price_moving_average_ratio_percentile(
            price,
            volume,
            ma_length=20,
            ma_type="vwma",
            pmarp_lookback=60,
            signal_ma_length=10,
            signal_ma_type="sma",
            line_mode="pmarp",
        )
        for key in expected:
            np.testing.assert_allclose(
                np.asarray(result[key], dtype=np.float64),
                expected[key],
                rtol=0.0,
                atol=5e-12,
                equal_nan=True,
            )

    def test_stream_matches_batch(self):
        data = load_test_data()
        price = np.asarray(data["close"][:144], dtype=np.float64)
        volume = np.asarray(data["volume"][:144], dtype=np.float64)
        batch = mp.price_moving_average_ratio_percentile(
            price,
            volume,
            ma_length=20,
            ma_type="vwma",
            pmarp_lookback=50,
            signal_ma_length=8,
            signal_ma_type="sma",
            line_mode="pmarp",
        )
        stream = mp.PriceMovingAverageRatioPercentileStream(20, "vwma", 50, 8, "sma", "pmarp")
        last = None
        for p, v in zip(price, volume):
            last = stream.update(float(p), float(v))

        assert last is not None
        for key in ["pmar", "pmarp", "plotline", "signal", "pmar_high", "pmar_low", "scaled_pmar"]:
            assert np.isclose(last[key], batch[key][-1], equal_nan=True)

    def test_batch_metadata_and_first_row_match_single(self):
        data = load_test_data()
        price = np.asarray(data["close"][:144], dtype=np.float64)
        volume = np.asarray(data["volume"][:144], dtype=np.float64)
        result = mp.price_moving_average_ratio_percentile_batch(
            price,
            volume,
            ma_length_range=(20, 22, 2),
            ma_type="vwma",
            pmarp_lookback_range=(60, 60, 0),
            signal_ma_length_range=(10, 10, 0),
            signal_ma_type="sma",
            line_mode="pmarp",
        )
        assert result["rows"] == 2
        assert result["cols"] == price.shape[0]

        single = mp.price_moving_average_ratio_percentile(
            price,
            volume,
            ma_length=20,
            ma_type="vwma",
            pmarp_lookback=60,
            signal_ma_length=10,
            signal_ma_type="sma",
            line_mode="pmarp",
        )
        for key in ["pmar", "pmarp", "plotline", "signal", "pmar_high", "pmar_low", "scaled_pmar"]:
            np.testing.assert_allclose(
                np.asarray(result[key][0], dtype=np.float64),
                np.asarray(single[key], dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )

    def test_kernel_parity(self):
        data = load_test_data()
        price = np.asarray(data["close"][:160], dtype=np.float64)
        volume = np.asarray(data["volume"][:160], dtype=np.float64)
        auto = mp.price_moving_average_ratio_percentile(
            price,
            volume,
            ma_length=20,
            ma_type="vwma",
            pmarp_lookback=60,
            signal_ma_length=10,
            signal_ma_type="sma",
            line_mode="pmarp",
        )
        scalar = mp.price_moving_average_ratio_percentile(
            price,
            volume,
            ma_length=20,
            ma_type="vwma",
            pmarp_lookback=60,
            signal_ma_length=10,
            signal_ma_type="sma",
            line_mode="pmarp",
            kernel="scalar",
        )
        for key in auto:
            np.testing.assert_allclose(auto[key], scalar[key], rtol=0.0, atol=1e-12, equal_nan=True)

    def test_invalid_length(self):
        data = load_test_data()
        price = np.asarray(data["close"][:32], dtype=np.float64)
        volume = np.asarray(data["volume"][:32], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid period"):
            mp.price_moving_average_ratio_percentile(price, volume, ma_length=0)

    def test_length_mismatch(self):
        data = load_test_data()
        price = np.asarray(data["close"][:32], dtype=np.float64)
        volume = np.asarray(data["volume"][:31], dtype=np.float64)
        with pytest.raises(ValueError, match="Inconsistent slice lengths"):
            mp.price_moving_average_ratio_percentile(price, volume)
