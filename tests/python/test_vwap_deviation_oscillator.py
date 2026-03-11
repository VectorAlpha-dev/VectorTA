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


DAY_MS = 86_400_000


def price_ref(high, low, close, use_close):
    if use_close:
        return close
    if np.isfinite(high) and np.isfinite(low) and np.isfinite(close):
        return (high + low + close) / 3.0
    return np.nan


def period_id(ts_ms, session_mode):
    sec = ts_ms // 1000
    if session_mode == "4_hours":
        return (sec // 3600) // 4
    if session_mode == "daily":
        return sec // 86_400
    if session_mode == "weekly":
        return ((sec // 86_400) + 3) // 7
    return 0


def rolling_std_ignore_nan(values, period):
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    buf = np.zeros(period, dtype=np.float64)
    head = 0
    count = 0
    total = 0.0
    total_sq = 0.0
    for i, value in enumerate(values):
        if np.isfinite(value):
            if count == period:
                old = buf[head]
                total -= old
                total_sq -= old * old
            else:
                count += 1
            buf[head] = value
            total += value
            total_sq += value * value
            head = (head + 1) % period
        if count == period:
            mean = total / period
            var = max(total_sq / period - mean * mean, 0.0)
            out[i] = np.sqrt(var)
    return out


def manual_vdo(
    timestamps,
    high,
    low,
    close,
    volume,
    session_mode,
    rolling_period,
    rolling_days,
    use_close,
    deviation_mode,
    z_window,
    pct_vol_lookback,
    pct_min_sigma,
    abs_vol_lookback,
):
    n = len(close)
    resid_abs = np.full(n, np.nan, dtype=np.float64)
    resid_pct = np.full(n, np.nan, dtype=np.float64)

    if session_mode == "rolling_bars":
        sum_pv = 0.0
        sum_vol = 0.0
        queue = [(0.0, 0.0)] * rolling_period
        head = 0
        count = 0
        for i in range(n):
            pr = price_ref(high[i], low[i], close[i], use_close)
            pv = pr * volume[i] if np.isfinite(pr) and np.isfinite(volume[i]) else 0.0
            vv = volume[i] if np.isfinite(pr) and np.isfinite(volume[i]) else 0.0
            if count == rolling_period:
                old_pv, old_v = queue[head]
                sum_pv -= old_pv
                sum_vol -= old_v
            else:
                count += 1
            queue[head] = (pv, vv)
            sum_pv += pv
            sum_vol += vv
            head = (head + 1) % rolling_period
            vwap = sum_pv / sum_vol if sum_vol != 0.0 else np.nan
            if np.isfinite(pr) and np.isfinite(vwap):
                resid_abs[i] = pr - vwap
                if vwap != 0.0:
                    resid_pct[i] = 100.0 * (pr / vwap - 1.0)
    elif session_mode == "rolling_days":
        sum_pv = 0.0
        sum_vol = 0.0
        queue = []
        days_ms = rolling_days * DAY_MS
        for i in range(n):
            pr = price_ref(high[i], low[i], close[i], use_close)
            pv = pr * volume[i] if np.isfinite(pr) and np.isfinite(volume[i]) else 0.0
            vv = volume[i] if np.isfinite(pr) and np.isfinite(volume[i]) else 0.0
            queue.append((timestamps[i], pv, vv))
            sum_pv += pv
            sum_vol += vv
            cutoff = timestamps[i] - days_ms
            while queue and queue[0][0] < cutoff:
                _, old_pv, old_v = queue.pop(0)
                sum_pv -= old_pv
                sum_vol -= old_v
            vwap = sum_pv / sum_vol if sum_vol != 0.0 else np.nan
            if np.isfinite(pr) and np.isfinite(vwap):
                resid_abs[i] = pr - vwap
                if vwap != 0.0:
                    resid_pct[i] = 100.0 * (pr / vwap - 1.0)
    else:
        sum_pv = 0.0
        sum_vol = 0.0
        last_id = None
        for i in range(n):
            current_id = period_id(int(timestamps[i]), session_mode)
            if last_id != current_id:
                last_id = current_id
                sum_pv = 0.0
                sum_vol = 0.0
            pr = price_ref(high[i], low[i], close[i], use_close)
            if np.isfinite(pr) and np.isfinite(volume[i]):
                sum_pv += pr * volume[i]
                sum_vol += volume[i]
            vwap = sum_pv / sum_vol if sum_vol != 0.0 else np.nan
            if np.isfinite(pr) and np.isfinite(vwap):
                resid_abs[i] = pr - vwap
                if vwap != 0.0:
                    resid_pct[i] = 100.0 * (pr / vwap - 1.0)

    if deviation_mode == "absolute":
        osc = resid_abs
        std1 = rolling_std_ignore_nan(resid_abs, abs_vol_lookback)
        std1 = np.where(np.isfinite(std1), np.maximum(std1, 1.0), np.nan)
    elif deviation_mode == "percent":
        osc = resid_pct
        std1 = rolling_std_ignore_nan(resid_pct, pct_vol_lookback)
        std1 = np.where(np.isfinite(std1), np.maximum(std1, pct_min_sigma), np.nan)
    else:
        osc = np.full(n, np.nan, dtype=np.float64)
        std1 = np.ones(n, dtype=np.float64)
        stats = rolling_std_ignore_nan(resid_abs, z_window)
        buf = np.zeros(z_window, dtype=np.float64)
        head = 0
        count = 0
        total = 0.0
        total_sq = 0.0
        for i, value in enumerate(resid_abs):
            if np.isfinite(value):
                if count == z_window:
                    old = buf[head]
                    total -= old
                    total_sq -= old * old
                else:
                    count += 1
                buf[head] = value
                total += value
                total_sq += value * value
                head = (head + 1) % z_window
            if count == z_window and np.isfinite(value):
                mean = total / z_window
                std = np.sqrt(max(total_sq / z_window - mean * mean, 0.0))
                if std != 0.0:
                    osc[i] = (value - mean) / std
        return osc, std1, std1 * 2.0, std1 * 3.0

    return osc, std1, std1 * 2.0, std1 * 3.0


class TestVwapDeviationOscillator:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_manual_reference_matches_binding(self, test_data):
        n = 180
        timestamps = test_data["timestamp"][:n]
        high = test_data["high"][:n]
        low = test_data["low"][:n]
        close = test_data["close"][:n]
        volume = test_data["volume"][:n]

        result = mp.vwap_deviation_oscillator(
            timestamps,
            high,
            low,
            close,
            volume,
            session_mode="rolling_bars",
            rolling_period=20,
            use_close=True,
            deviation_mode="percent",
            pct_vol_lookback=25,
            pct_min_sigma=0.1,
        )
        expected = manual_vdo(
            timestamps,
            high,
            low,
            close,
            volume,
            "rolling_bars",
            20,
            30,
            True,
            "percent",
            50,
            25,
            0.1,
            100,
        )

        for actual, want in zip(result, expected):
            np.testing.assert_allclose(actual, want, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_stream_matches_batch(self, test_data):
        n = 128
        timestamps = test_data["timestamp"][:n]
        high = test_data["high"][:n]
        low = test_data["low"][:n]
        close = test_data["close"][:n]
        volume = test_data["volume"][:n]

        batch = mp.vwap_deviation_oscillator(
            timestamps,
            high,
            low,
            close,
            volume,
            session_mode="rolling_days",
            rolling_days=5,
            use_close=False,
            deviation_mode="absolute",
            abs_vol_lookback=20,
        )
        stream = mp.VwapDeviationOscillatorStream(
            session_mode="rolling_days",
            rolling_days=5,
            use_close=False,
            deviation_mode="absolute",
            abs_vol_lookback=20,
        )
        rows = [stream.update(int(t), float(h), float(l), float(c), float(v)) for t, h, l, c, v in zip(timestamps, high, low, close, volume)]
        got = [np.array([row[i] for row in rows], dtype=np.float64) for i in range(4)]

        for actual, want in zip(got, batch):
            np.testing.assert_allclose(actual, want, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_batch_metadata_and_first_row(self, test_data):
        n = 96
        timestamps = test_data["timestamp"][:n]
        high = test_data["high"][:n]
        low = test_data["low"][:n]
        close = test_data["close"][:n]
        volume = test_data["volume"][:n]

        result = mp.vwap_deviation_oscillator_batch(
            timestamps,
            high,
            low,
            close,
            volume,
            session_mode="rolling_bars",
            deviation_mode="zscore",
            rolling_period_range=(20, 22, 2),
            z_window_range=(25, 25, 0),
        )

        assert result["rows"] == 2
        assert result["cols"] == n
        np.testing.assert_array_equal(result["rolling_periods"], np.array([20, 22], dtype=np.uint64))

        single = mp.vwap_deviation_oscillator(
            timestamps,
            high,
            low,
            close,
            volume,
            session_mode="rolling_bars",
            rolling_period=20,
            deviation_mode="zscore",
            z_window=25,
        )
        np.testing.assert_allclose(result["osc"][0], single[0], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(result["std1"][0], single[1], rtol=0.0, atol=1e-12, equal_nan=True)

    def test_kernel_parity(self, test_data):
        n = 128
        timestamps = test_data["timestamp"][:n]
        high = test_data["high"][:n]
        low = test_data["low"][:n]
        close = test_data["close"][:n]
        volume = test_data["volume"][:n]

        auto = mp.vwap_deviation_oscillator(
            timestamps, high, low, close, volume, session_mode="daily", deviation_mode="absolute"
        )
        scalar = mp.vwap_deviation_oscillator(
            timestamps,
            high,
            low,
            close,
            volume,
            session_mode="daily",
            deviation_mode="absolute",
            kernel="scalar",
        )
        for a, b in zip(auto, scalar):
            np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_invalid_params(self, test_data):
        n = 64
        timestamps = test_data["timestamp"][:n]
        high = test_data["high"][:n]
        low = test_data["low"][:n]
        close = test_data["close"][:n]
        volume = test_data["volume"][:n]

        with pytest.raises(ValueError, match="Invalid z_window"):
            mp.vwap_deviation_oscillator(
                timestamps, high, low, close, volume, deviation_mode="zscore", z_window=4
            )

        with pytest.raises(ValueError, match="length mismatch"):
            mp.vwap_deviation_oscillator(
                timestamps[:-1], high, low, close, volume, session_mode="rolling_bars"
            )
