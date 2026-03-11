import numpy as np
import pytest

try:
    import vector_ta as ta_indicators
except ImportError:
    try:
        import my_project as ta_indicators
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )

from test_utils import assert_close


def linear_data(size):
    return np.array([float(i) for i in range(size)], dtype=np.float64)


def constant_data(size, value):
    return np.full(size, float(value), dtype=np.float64)


def test_adaptive_macd_constant_series_flattens_to_zero():
    data = constant_data(24, 100.0)
    macd, signal, hist = ta_indicators.adaptive_macd(
        data, length=6, fast_period=5, slow_period=10, signal_period=4
    )

    assert np.all(np.isnan(macd[:5]))
    assert np.all(np.isnan(signal[:5]))
    assert np.all(np.isnan(hist[:5]))
    assert_close(macd[5:], np.zeros(19), atol=1e-12)
    assert_close(signal[5:], np.zeros(19), atol=1e-12)
    assert_close(hist[5:], np.zeros(19), atol=1e-12)


def test_adaptive_macd_invalid_period():
    data = linear_data(10)
    with pytest.raises(ValueError, match="(?i)invalid period"):
        ta_indicators.adaptive_macd(
            data, length=1, fast_period=3, slow_period=6, signal_period=3
        )


def test_adaptive_macd_nan_gap_semantics():
    data = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
        dtype=np.float64,
    )
    macd, signal, hist = ta_indicators.adaptive_macd(
        data, length=4, fast_period=3, slow_period=6, signal_period=3
    )

    assert np.all(np.isnan(macd[:3]))
    assert np.isfinite(macd[3])
    assert np.all(np.isnan(macd[6:10]))
    assert np.isfinite(macd[10])
    assert np.isnan(hist[6])


def test_adaptive_macd_stream_matches_batch():
    data = linear_data(28)
    batch_macd, batch_signal, batch_hist = ta_indicators.adaptive_macd(
        data, length=5, fast_period=4, slow_period=8, signal_period=3
    )

    stream = ta_indicators.AdaptiveMacdStream(
        length=5, fast_period=4, slow_period=8, signal_period=3
    )
    macd = []
    signal = []
    hist = []
    for value in data:
        current = stream.update(float(value))
        if current is None:
            macd.append(np.nan)
            signal.append(np.nan)
            hist.append(np.nan)
        else:
            m, s, h = current
            macd.append(m)
            signal.append(s)
            hist.append(h)

    assert_close(np.array(macd), batch_macd, atol=1e-12)
    assert_close(np.array(signal), batch_signal, atol=1e-12)
    assert_close(np.array(hist), batch_hist, atol=1e-12)


def test_adaptive_macd_batch_matches_single():
    data = linear_data(26)
    batch = ta_indicators.adaptive_macd_batch(
        data,
        length_range=(4, 5, 1),
        fast_period_range=(3, 4, 1),
        slow_period_range=(6, 7, 1),
        signal_period_range=(3, 3, 0),
    )

    assert batch["macd"].shape == (8, data.size)
    assert batch["signal"].shape == (8, data.size)
    assert batch["hist"].shape == (8, data.size)
    assert batch["rows"] == 8
    assert batch["cols"] == data.size

    row = 0
    for length in [4, 5]:
        for fast_period in [3, 4]:
            for slow_period in [6, 7]:
                macd, signal, hist = ta_indicators.adaptive_macd(
                    data,
                    length=length,
                    fast_period=fast_period,
                    slow_period=slow_period,
                    signal_period=3,
                )
                assert batch["lengths"][row] == length
                assert batch["fast_periods"][row] == fast_period
                assert batch["slow_periods"][row] == slow_period
                assert batch["signal_periods"][row] == 3
                assert_close(batch["macd"][row], macd, atol=1e-12)
                assert_close(batch["signal"][row], signal, atol=1e-12)
                assert_close(batch["hist"][row], hist, atol=1e-12)
                row += 1
