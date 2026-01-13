"""
Python binding tests for KVO CUDA kernels.
Skips gracefully when CUDA is unavailable or Python CUDA module not built.
"""
import pytest
import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

try:
    import my_project as ti
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python,cuda' first",
        allow_module_level=True,
    )

from test_utils import assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, 'kvo_cuda_batch_dev'):
        return False

    try:
        n = 8
        h = np.full(n, np.nan, dtype=np.float32)
        l = np.full(n, np.nan, dtype=np.float32)
        c = np.full(n, np.nan, dtype=np.float32)
        v = np.full(n, np.nan, dtype=np.float32)
        h[3:] = 10.0
        l[3:] = 9.0
        c[3:] = 9.5
        v[3:] = 1000.0
        handle, meta = ti.kvo_cuda_batch_dev(
            h, l, c, v, (2, 2, 0), (5, 5, 0)
        )
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        s = str(e).lower()
        return not ("cuda not available" in s or "ptx" in s or "nvcc" in s)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestKvoCuda:
    def test_kvo_cuda_batch_matches_cpu(self):
        n = 4096
        x = np.arange(n, dtype=np.float64)
        base = np.sin(x * 0.00123) + 0.00017 * x
        spread = 0.1 * np.abs(np.cos(x * 0.00077))
        high = base + spread
        low = base - spread
        close = base + 0.05 * np.sin(x * 0.00111)
        volume = (np.abs(np.cos(x * 0.0031)) + 1.0) * 500.0
        high[:6] = low[:6] = close[:6] = volume[:6] = np.nan

        short, long = 6, 20
        cpu = ti.kvo(high, low, close, volume, short, long)
        handle, meta = ti.kvo_cuda_batch_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
            volume.astype(np.float32),
            (short, short, 0),
            (long, long, 0),
        )
        gpu = cp.asnumpy(cp.asarray(handle))[0]
        assert_close(gpu, cpu, rtol=5e-3, atol=5e-4, msg="KVO CUDA batch vs CPU mismatch")

    def test_kvo_cuda_many_series_one_param_matches_cpu(self):
        T, N = 2048, 4
        x = np.arange(T, dtype=np.float64)
        base = np.cos(x * 0.0021) + 0.0004 * x
        spread = 0.08 * np.abs(np.sin(x * 0.001))
        series = base
        h = np.tile(series + spread, (N, 1)).T
        l = np.tile(series - spread, (N, 1)).T
        c = np.tile(series + 0.03 * np.sin(x * 0.0017), (N, 1)).T
        v = np.tile((np.abs(np.sin(x * 0.0042)) + 0.9) * 300.0, (N, 1)).T
        h[:6, :] = l[:6, :] = c[:6, :] = v[:6, :] = np.nan

        short, long = 6, 20
        cpu_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            cpu_tm[:, j] = ti.kvo(h[:, j], l[:, j], c[:, j], v[:, j], short, long)

        handle = ti.kvo_cuda_many_series_one_param_dev(
            h.astype(np.float32).ravel(),
            l.astype(np.float32).ravel(),
            c.astype(np.float32).ravel(),
            v.astype(np.float32).ravel(),
            N,
            T,
            short,
            long,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle)).reshape(T, N)
        assert_close(gpu_tm, cpu_tm, rtol=5e-3, atol=5e-4, msg="KVO CUDA many-series vs CPU mismatch")

