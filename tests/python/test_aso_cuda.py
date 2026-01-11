"""
Python binding tests for ASO CUDA kernels.
Skips gracefully when CUDA is unavailable or CUDA bindings not present.
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

from test_utils import assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, 'aso_cuda_batch_dev'):
        return False
    try:
        n = 16
        o = np.linspace(1, 2, n, dtype=np.float32)
        h = o + 0.1
        l = o - 0.1
        c = o + 0.02
        hb, he = ti.aso_cuda_batch_dev(o, h, l, c, (10, 10, 0), (0, 0, 0))
        _ = cp.asarray(hb); _ = cp.asarray(he)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestAsoCuda:
    def test_aso_cuda_batch_matches_cpu(self):
        n = 2048
        o = np.zeros(n, dtype=np.float64)
        h = np.zeros(n, dtype=np.float64)
        l = np.zeros(n, dtype=np.float64)
        c = np.zeros(n, dtype=np.float64)
        for i in range(4, n):
            x = float(i)
            base = np.sin(x * 0.0021) + 0.0002 * x
            o[i] = base - 0.05
            h[i] = base + 0.12
            l[i] = base - 0.11
            c[i] = base + 0.02

        
        bulls_cpu, bears_cpu = ti.aso(o, h, l, c, 10, 0)

        hb, he = ti.aso_cuda_batch_dev(
            o.astype(np.float32),
            h.astype(np.float32),
            l.astype(np.float32),
            c.astype(np.float32),
            (10, 10, 0),
            (0, 0, 0),
        )
        gb = cp.asnumpy(cp.asarray(hb))[0]
        ge = cp.asnumpy(cp.asarray(he))[0]

        assert_close(gb, bulls_cpu, rtol=2e-3, atol=2e-3, msg="ASO CUDA batch bulls mismatch")
        assert_close(ge, bears_cpu, rtol=2e-3, atol=2e-3, msg="ASO CUDA batch bears mismatch")

    def test_aso_cuda_many_series_one_param_matches_cpu(self):
        cols, rows = 4, 1024
        period, mode = 10, 0
        o_tm = np.zeros((rows, cols), dtype=np.float64)
        h_tm = np.zeros_like(o_tm)
        l_tm = np.zeros_like(o_tm)
        c_tm = np.zeros_like(o_tm)
        for s in range(cols):
            for t in range(s, rows):
                x = float(t) + 0.3 * float(s)
                base = np.sin(x * 0.0025) + 0.00015 * x
                o_tm[t, s] = base - 0.03
                h_tm[t, s] = base + 0.09
                l_tm[t, s] = base - 0.08
                c_tm[t, s] = base + 0.015

        cpu_b = np.zeros_like(c_tm)
        cpu_e = np.zeros_like(c_tm)
        for s in range(cols):
            b, e = ti.aso(o_tm[:, s], h_tm[:, s], l_tm[:, s], c_tm[:, s], period, mode)
            cpu_b[:, s] = b
            cpu_e[:, s] = e

        hb, he = ti.aso_cuda_many_series_one_param_dev(
            o_tm.astype(np.float32),
            h_tm.astype(np.float32),
            l_tm.astype(np.float32),
            c_tm.astype(np.float32),
            cols,
            rows,
            period,
            mode,
        )
        gb = cp.asnumpy(cp.asarray(hb))
        ge = cp.asnumpy(cp.asarray(he))
        assert gb.shape == (rows, cols)
        assert ge.shape == (rows, cols)
        assert_close(gb, cpu_b, rtol=2e-3, atol=2e-3, msg="ASO CUDA many-series bulls mismatch")
        assert_close(ge, cpu_e, rtol=2e-3, atol=2e-3, msg="ASO CUDA many-series bears mismatch")

