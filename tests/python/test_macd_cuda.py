"""
Python binding tests for MACD CUDA kernels (EMA path).
Skips gracefully when CUDA is unavailable or CUDA feature not built.
"""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import my_project as ti
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python,cuda' first",
        allow_module_level=True,
    )

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "macd_cuda_batch_dev"):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        d = ti.macd_cuda_batch_dev(x, (3, 3, 0), (4, 4, 0), (2, 2, 0), "ema")
        _ = cp.asarray(d["macd"])
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "device" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestMacdCuda:
    @pytest.fixture(scope="class")
    def close(self):
        return load_test_data()["close"].astype(np.float64)

    def test_macd_cuda_batch_matches_cpu(self, close):
        f, s, g = 12, 26, 9
        macd_cpu, sig_cpu, hist_cpu = ti.macd(close, f, s, g, "ema")
        d = ti.macd_cuda_batch_dev(
            close.astype(np.float32),
            (f, f, 0),
            (s, s, 0),
            (g, g, 0),
            "ema",
        )
        macd_gpu = cp.asnumpy(cp.asarray(d["macd"]))[0]
        sig_gpu = cp.asnumpy(cp.asarray(d["signal"]))[0]
        hist_gpu = cp.asnumpy(cp.asarray(d["hist"]))[0]

        assert_close(macd_gpu, macd_cpu, rtol=1e-4, atol=1e-5, msg="MACD mismatch")
        assert_close(sig_gpu, sig_cpu, rtol=1e-4, atol=1e-5, msg="Signal mismatch")
        assert_close(hist_gpu, hist_cpu, rtol=1e-4, atol=1e-5, msg="Hist mismatch")

    def test_macd_cuda_batch_multi_combo_matches_cpu(self, close):
        fast_range = (10, 14, 2)
        slow_range = (24, 28, 2)
        sig_range = (8, 10, 1)
        cpu = ti.macd_batch(close, fast_range, slow_range, sig_range, "ema")
        d = ti.macd_cuda_batch_dev(
            close.astype(np.float32), fast_range, slow_range, sig_range, "ema"
        )
        macd_gpu = cp.asnumpy(cp.asarray(d["macd"]))
        sig_gpu = cp.asnumpy(cp.asarray(d["signal"]))
        hist_gpu = cp.asnumpy(cp.asarray(d["hist"]))
        assert macd_gpu.shape == cpu["macd"].shape
        assert sig_gpu.shape == cpu["signal"].shape
        assert hist_gpu.shape == cpu["hist"].shape
        assert_close(macd_gpu, cpu["macd"], rtol=1e-4, atol=1e-5)
        assert_close(sig_gpu, cpu["signal"], rtol=1e-4, atol=1e-5)
        assert_close(hist_gpu, cpu["hist"], rtol=1e-4, atol=1e-5)

    def test_macd_cuda_many_series_one_param_matches_cpu(self, close):
        T = 2048
        N = 4
        series = np.vstack([close[:T] * (1.0 + 0.01 * j) for j in range(N)]).T
        f, s, g = 12, 26, 9
        macd_cpu = np.zeros_like(series)
        sig_cpu = np.zeros_like(series)
        hist_cpu = np.zeros_like(series)
        for j in range(N):
            m, si, h = ti.macd(series[:, j], f, s, g, "ema")
            macd_cpu[:, j] = m
            sig_cpu[:, j] = si
            hist_cpu[:, j] = h
        d = ti.macd_cuda_many_series_one_param_dev(
            series.astype(np.float32), f, s, g, "ema"
        )
        macd_gpu = cp.asnumpy(cp.asarray(d["macd"]))
        sig_gpu = cp.asnumpy(cp.asarray(d["signal"]))
        hist_gpu = cp.asnumpy(cp.asarray(d["hist"]))
        assert macd_gpu.shape == series.shape
        assert_close(macd_gpu, macd_cpu, rtol=1e-4, atol=1e-5)
        assert_close(sig_gpu, sig_cpu, rtol=1e-4, atol=1e-5)
        assert_close(hist_gpu, hist_cpu, rtol=1e-4, atol=1e-5)

