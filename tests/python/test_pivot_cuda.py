"""
Python binding tests for Pivot CUDA kernels.
Skips gracefully when CUDA or bindings are unavailable.
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
    pytest.skip("Python module not built. Run 'maturin develop --features python,cuda' first", allow_module_level=True)


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, 'pivot_cuda_batch_dev'):
        return False

    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32)

        h = x + 0.6
        l = x - 0.4
        c = x
        o = x + 0.1
        handle, meta = ti.pivot_cuda_batch_dev(h, l, c, o, (3, 3, 1))
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'no cuda device' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or CUDA bindings missing")
class TestPivotCuda:
    def test_pivot_cuda_batch_matches_cpu(self):
        n = 2048

        x = np.arange(n, dtype=np.float64)
        base = np.sin(x * 0.01) + 0.001 * x
        rng = 0.2 + 0.05 * np.abs(np.cos(x * 0.02))
        h = np.full(n, np.nan, dtype=np.float64)
        l = np.full(n, np.nan, dtype=np.float64)
        c = np.full(n, np.nan, dtype=np.float64)
        o = np.full(n, np.nan, dtype=np.float64)
        h[5:] = base[5:] + rng[5:]
        l[5:] = base[5:] - rng[5:]
        c[5:] = base[5:]
        o[5:] = base[5:] + 0.01 * np.sin(x[5:] * 0.03)


        r4, r3, r2, r1, pp, s1, s2, s3, s4 = ti.pivot(h, l, c, o, mode=3)


        handle, meta = ti.pivot_cuda_batch_dev(
            h.astype(np.float32),
            l.astype(np.float32),
            c.astype(np.float32),
            o.astype(np.float32),
            (3, 3, 1),
        )
        gpu = cp.asnumpy(cp.asarray(handle))
        assert gpu.shape == (9, n)


        np.testing.assert_allclose(gpu[4], pp, rtol=5e-4, atol=5e-6)
        np.testing.assert_allclose(gpu[0], r4, rtol=5e-4, atol=5e-6)

    def test_pivot_cuda_many_series_one_param_matches_cpu(self):
        rows = 1024
        cols = 4
        h_tm = np.full((rows, cols), np.nan, dtype=np.float64)
        l_tm = np.full((rows, cols), np.nan, dtype=np.float64)
        c_tm = np.full((rows, cols), np.nan, dtype=np.float64)
        o_tm = np.full((rows, cols), np.nan, dtype=np.float64)
        x = np.arange(rows, dtype=np.float64)
        for s in range(cols):
            base = np.sin(x * (0.01 + 0.002 * s)) + 0.001 * x
            rng = 0.2 + 0.04 * np.abs(np.cos(x * 0.02 + 0.1 * s))
            h_tm[3:, s] = base[3:] + rng[3:]
            l_tm[3:, s] = base[3:] - rng[3:]
            c_tm[3:, s] = base[3:]
            o_tm[3:, s] = base[3:] + 0.01 * np.sin(x[3:] * 0.03)


        cpu_pp = np.zeros_like(c_tm)
        for s in range(cols):
            r4, r3, r2, r1, pp, s1, s2, s3, s4 = ti.pivot(
                h_tm[:, s], l_tm[:, s], c_tm[:, s], o_tm[:, s], mode=3
            )
            cpu_pp[:, s] = pp

        handle = ti.pivot_cuda_many_series_one_param_dev(
            h_tm.astype(np.float32).ravel(),
            l_tm.astype(np.float32).ravel(),
            c_tm.astype(np.float32).ravel(),
            o_tm.astype(np.float32).ravel(),
            cols,
            rows,
            3,
        )
        gpu = cp.asnumpy(cp.asarray(handle))
        assert gpu.shape == (9 * rows, cols)
        gpu_pp = gpu[4 * rows : 5 * rows, :]
        np.testing.assert_allclose(gpu_pp, cpu_pp, rtol=5e-4, atol=5e-6)

