import math

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


MAX_LOOKBACK = 75


def nz(value: float) -> float:
    return value if np.isfinite(value) else 0.0


def ring_get(buf, center: int, off: int) -> float:
    n = len(buf)
    idx = center + n - (off % n)
    if idx >= n:
        idx -= n
    return buf[idx]


def median3(a: float, b: float, c: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
        return np.nan
    return (a + b + c) - min(a, b, c) - max(a, b, c)


def manual_reference(data: np.ndarray, alpha: float = 0.07, cutoff: float = 8.0) -> np.ndarray:
    out = np.full(data.shape[0], np.nan, dtype=np.float64)
    src_ring = np.full(MAX_LOOKBACK + 1, np.nan, dtype=np.float64)
    smooth_ring = np.full(3, np.nan, dtype=np.float64)
    c_ring = np.full(7, np.nan, dtype=np.float64)
    dp_ring = np.full(5, np.nan, dtype=np.float64)
    src_idx = 0
    smooth_idx = 0
    c_idx = 0
    dp_idx = 0
    prev_ip = np.nan
    prev_p = np.nan
    prev_q1 = np.nan
    prev_i1 = np.nan
    f3_hist = [np.nan, np.nan, np.nan]
    valid_count = 0

    coef_c = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha)
    one_minus_alpha = 1.0 - alpha
    a1 = math.exp(-math.pi / cutoff)
    b1 = 2.0 * a1 * math.cos(1.738 * math.pi / cutoff)
    c1 = a1 * a1
    coef2 = b1 + c1
    coef3 = -(c1 + b1 * c1)
    coef4 = c1 * c1
    coef1 = 1.0 - coef2 - coef3 - coef4

    for i, source in enumerate(data):
        if not np.isfinite(source):
            continue

        src_ring[src_idx] = source
        src0 = ring_get(src_ring, src_idx, 0)
        src1 = ring_get(src_ring, src_idx, 1)
        src2 = ring_get(src_ring, src_idx, 2)
        src3 = ring_get(src_ring, src_idx, 3)

        smooth = (src0 + 2.0 * src1 + 2.0 * src2 + src3) / 6.0 if np.isfinite(src0) and np.isfinite(src1) and np.isfinite(src2) and np.isfinite(src3) else np.nan
        smooth_ring[smooth_idx] = smooth

        smooth1 = nz(ring_get(smooth_ring, smooth_idx, 1))
        smooth2 = nz(ring_get(smooth_ring, smooth_idx, 2))
        c_prev1 = nz(ring_get(c_ring, c_idx, 1))
        c_prev2 = nz(ring_get(c_ring, c_idx, 2))
        c_main = coef_c * (smooth - 2.0 * smooth1 + smooth2) + 2.0 * one_minus_alpha * c_prev1 - one_minus_alpha * one_minus_alpha * c_prev2 if np.isfinite(smooth) else np.nan
        c_fallback = (src0 - 2.0 * src1 + src2) / 4.0 if np.isfinite(src0) and np.isfinite(src1) and np.isfinite(src2) else np.nan
        c_value = c_main if np.isfinite(c_main) else c_fallback
        c_ring[c_idx] = c_value

        if np.isfinite(c_value):
            factor = 0.5 + 0.08 * nz(prev_ip)
            q1 = (
                0.0962 * c_value
                + 0.5769 * nz(ring_get(c_ring, c_idx, 2))
                - 0.5769 * nz(ring_get(c_ring, c_idx, 4))
                - 0.0962 * nz(ring_get(c_ring, c_idx, 6))
            ) * factor
        else:
            q1 = np.nan
        i1 = nz(ring_get(c_ring, c_idx, 3))

        if np.isfinite(q1) and np.isfinite(prev_q1) and q1 != 0.0 and prev_q1 != 0.0:
            prev_i1_nz = nz(prev_i1)
            prev_q1_nz = nz(prev_q1)
            numer = (i1 / q1) - (prev_i1_nz / prev_q1_nz)
            denom = 1.0 + i1 * prev_i1_nz / (q1 * prev_q1_nz)
            dp_raw = numer / denom
        else:
            dp_raw = 0.0
        dp = 0.1 if dp_raw < 0.1 else 1.1 if dp_raw > 1.1 else dp_raw
        dp_ring[dp_idx] = dp

        md_inner = median3(
            ring_get(dp_ring, dp_idx, 2),
            ring_get(dp_ring, dp_idx, 3),
            ring_get(dp_ring, dp_idx, 4),
        )
        md = median3(dp, ring_get(dp_ring, dp_idx, 1), md_inner)
        dc = 15.0 if md == 0.0 else (2.0 * math.pi / md) + 0.5
        ip = 0.33 * dc + 0.67 * nz(prev_ip)
        p = 0.15 * ip + 0.85 * nz(prev_p)

        pr = round(abs(p - 1.0)) if np.isfinite(p) else np.nan
        if np.isfinite(pr):
            lookback = int(pr)
            if 1 <= lookback <= MAX_LOOKBACK:
                past = ring_get(src_ring, src_idx, lookback)
                v1 = source - past if np.isfinite(past) else np.nan
            else:
                v1 = 0.0
        else:
            v1 = 0.0

        raw_f3 = coef1 * v1 + coef2 * nz(f3_hist[0]) + coef3 * nz(f3_hist[1]) + coef4 * nz(f3_hist[2]) if np.isfinite(v1) else np.nan
        f3 = raw_f3 if np.isfinite(raw_f3) else v1

        prev_q1 = q1
        prev_i1 = i1
        prev_ip = ip
        prev_p = p
        f3_hist[2] = f3_hist[1]
        f3_hist[1] = f3_hist[0]
        f3_hist[0] = f3

        valid_count += 1
        src_idx = (src_idx + 1) % src_ring.shape[0]
        smooth_idx = (smooth_idx + 1) % smooth_ring.shape[0]
        c_idx = (c_idx + 1) % c_ring.shape[0]
        dp_idx = (dp_idx + 1) % dp_ring.shape[0]

        if valid_count > MAX_LOOKBACK:
            out[i] = f3

    return out


class TestEhlersSmoothedAdaptiveMomentum:
    def test_manual_reference_matches_binding(self):
        data = load_test_data()
        source = ((np.asarray(data["high"][:220]) + np.asarray(data["low"][:220])) * 0.5).astype(np.float64)
        expected = manual_reference(source, 0.07, 8.0)
        result = mp.ehlers_smoothed_adaptive_momentum(source, alpha=0.07, cutoff=8.0)
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_stream_matches_batch(self):
        data = load_test_data()
        source = ((np.asarray(data["high"][:220]) + np.asarray(data["low"][:220])) * 0.5).astype(np.float64)
        batch = mp.ehlers_smoothed_adaptive_momentum(source, alpha=0.07, cutoff=8.0)
        stream = mp.EhlersSmoothedAdaptiveMomentumStream(0.07, 8.0)
        streamed = np.array([stream.update(float(x)) for x in source], dtype=np.float64)
        np.testing.assert_allclose(streamed, batch, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_batch_metadata_and_first_row_match_single(self):
        data = load_test_data()
        source = ((np.asarray(data["high"][:180]) + np.asarray(data["low"][:180])) * 0.5).astype(np.float64)
        result = mp.ehlers_smoothed_adaptive_momentum_batch(
            source,
            alpha_range=(0.07, 0.09, 0.02),
            cutoff_range=(8.0, 8.0, 0.0),
        )
        assert result["rows"] == 2
        assert result["cols"] == source.shape[0]
        np.testing.assert_allclose(result["alphas"], np.array([0.07, 0.09]), rtol=0.0, atol=1e-12)
        single = mp.ehlers_smoothed_adaptive_momentum(source, alpha=0.07, cutoff=8.0)
        np.testing.assert_allclose(result["values"][0], single, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_invalid_cutoff(self):
        data = load_test_data()
        source = np.asarray(data["close"][:128], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid cutoff"):
            mp.ehlers_smoothed_adaptive_momentum(source, alpha=0.07, cutoff=0.0)
