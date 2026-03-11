import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    try:
        import my_project as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


def manual_reference(source: np.ndarray, high: np.ndarray, low: np.ndarray, smooth_period: int = 10) -> np.ndarray:
    n = source.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    smooth = np.zeros(n, dtype=np.float64)
    detrender = np.zeros(n, dtype=np.float64)
    q1 = np.zeros(n, dtype=np.float64)
    i1 = np.zeros(n, dtype=np.float64)
    range_1 = np.zeros(n, dtype=np.float64)

    period_mult = 0.075 * float(smooth_period) + 0.54
    i2_prev = 0.0
    q2_prev = 0.0
    re_prev = 0.0
    im_prev = 0.0
    period_prev = 0.0
    snr_prev = 0.0
    valid_count = 0

    def nz(arr: np.ndarray, idx: int) -> float:
        if idx < 0:
            return 0.0
        value = arr[idx]
        return value if np.isfinite(value) else 0.0

    for i in range(n):
        if not (np.isfinite(source[i]) and np.isfinite(high[i]) and np.isfinite(low[i])):
            continue

        range_1[i] = 0.1 * (high[i] - low[i]) + 0.9 * (range_1[i - 1] if i > 0 else 0.0)

        if valid_count > 5:
            smooth[i] = (4.0 * source[i] + 3.0 * nz(source, i - 1) + 2.0 * nz(source, i - 2) + nz(source, i - 3)) / 10.0
            detrender[i] = (
                0.0962 * smooth[i]
                + 0.5769 * nz(smooth, i - 2)
                - 0.5769 * nz(smooth, i - 4)
                - 0.0962 * nz(smooth, i - 6)
            ) * period_mult
            i1[i] = nz(detrender, i - 3)
            q1[i] = (
                0.0962 * detrender[i]
                + 0.5769 * nz(detrender, i - 2)
                - 0.5769 * nz(detrender, i - 4)
                - 0.0962 * nz(detrender, i - 6)
            ) * period_mult

            ji = (
                0.0962 * i1[i]
                + 0.5769 * nz(i1, i - 2)
                - 0.5769 * nz(i1, i - 4)
                - 0.0962 * nz(i1, i - 6)
            ) * period_mult
            jq = (
                0.0962 * q1[i]
                + 0.5769 * nz(q1, i - 2)
                - 0.5769 * nz(q1, i - 4)
                - 0.0962 * nz(q1, i - 6)
            ) * period_mult

            i2 = 0.2 * (i1[i] - jq) + 0.8 * i2_prev
            q2 = 0.2 * (q1[i] + ji) + 0.8 * q2_prev

            re = 0.2 * (i2 * i2_prev + q2 * q2_prev) + 0.8 * re_prev
            im = 0.2 * (i2 * q2_prev - q2 * i2_prev) + 0.8 * im_prev

            period = period_prev
            if re != 0.0 and im != 0.0:
                angle = np.arctan2(im, re)
                if angle != 0.0:
                    period = 2.0 * np.pi / abs(angle)
            if period_prev != 0.0:
                period = min(period, 1.5 * period_prev)
                period = max(period, 0.67 * period_prev)
            period = min(max(period, 6.0), 50.0)
            period = 0.2 * period + 0.8 * period_prev

            power = i1[i] * i1[i] + q1[i] * q1[i]
            noise = range_1[i] * range_1[i]
            snr = snr_prev
            if power > 0.0 and noise > 0.0:
                snr_raw = 10.0 * np.log(power / noise) / np.log(10.0) + 6.0
                snr = 0.25 * snr_raw + 0.75 * snr_prev

            i2_prev = i2
            q2_prev = q2
            re_prev = re
            im_prev = im
            period_prev = period
            snr_prev = snr

        valid_count += 1
        if valid_count > 6:
            out[i] = snr_prev

    return out


class TestL2EhlersSignalToNoise:
    def test_manual_reference_matches_binding(self):
        data = load_test_data()
        high = np.asarray(data["high"][:180], dtype=np.float64)
        low = np.asarray(data["low"][:180], dtype=np.float64)
        source = (high + low) * 0.5
        expected = manual_reference(source, high, low, 10)
        result = mp.l2_ehlers_signal_to_noise(source, high, low, smooth_period=10)
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_stream_matches_batch(self):
        data = load_test_data()
        high = np.asarray(data["high"][:160], dtype=np.float64)
        low = np.asarray(data["low"][:160], dtype=np.float64)
        source = (high + low) * 0.5
        batch = mp.l2_ehlers_signal_to_noise(source, high, low, smooth_period=10)
        stream = mp.L2EhlersSignalToNoiseStream(10)
        last = np.nan
        for i in range(source.shape[0]):
            last = stream.update(float(source[i]), float(high[i]), float(low[i]))
        assert np.isclose(last, batch[-1], equal_nan=True)

    def test_batch_metadata_and_first_row_match_single(self):
        data = load_test_data()
        high = np.asarray(data["high"][:160], dtype=np.float64)
        low = np.asarray(data["low"][:160], dtype=np.float64)
        source = np.asarray(data["close"][:160], dtype=np.float64)
        result = mp.l2_ehlers_signal_to_noise_batch(
            source,
            high,
            low,
            smooth_period_range=(10, 12, 2),
        )
        assert result["rows"] == 2
        assert result["cols"] == source.shape[0]
        np.testing.assert_array_equal(result["smooth_periods"], np.array([10, 12], dtype=np.uint64))
        single = mp.l2_ehlers_signal_to_noise(source, high, low, smooth_period=10)
        np.testing.assert_allclose(result["values"][0], single, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_invalid_period(self):
        data = load_test_data()
        high = np.asarray(data["high"][:64], dtype=np.float64)
        low = np.asarray(data["low"][:64], dtype=np.float64)
        source = np.asarray(data["close"][:64], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid smooth_period"):
            mp.l2_ehlers_signal_to_noise(source, high, low, smooth_period=0)
