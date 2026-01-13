"""
Python binding tests for PRB CUDA kernels.
Skips gracefully when CUDA is unavailable or module not built with python,cuda.
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
    pytest.skip("Python module not built. Run 'maturin develop --features python,cuda'", allow_module_level=True)

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, 'prb_cuda_batch_dev'):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        main, up, lo = ti.prb_cuda_batch_dev(
            x.astype(np.float32),
            smooth_data=False,
            smooth_period_range=(10, 10, 0),
            regression_period_range=(3, 3, 0),
            polynomial_order_range=(2, 2, 0),
            regression_offset_range=(0, 0, 0),
        )
        _ = cp.asarray(main)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestPrbCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_prb_cuda_batch_matches_cpu(self, test_data):
        close = test_data['close']
        n, k, ro = 50, 2, 0

        cpu_main, cpu_up, cpu_lo = ti.prb(close, False, 10, n, k, ro, 2.0)


        main, up, lo = ti.prb_cuda_batch_dev(
            close.astype(np.float32),
            smooth_data=False,
            smooth_period_range=(10, 10, 0),
            regression_period_range=(n, n, 0),
            polynomial_order_range=(k, k, 0),
            regression_offset_range=(ro, ro, 0),
        )
        g_main = cp.asnumpy(cp.asarray(main))[0]
        g_up   = cp.asnumpy(cp.asarray(up))[0]
        g_lo   = cp.asnumpy(cp.asarray(lo))[0]


        assert_close(g_main, cpu_main, rtol=1e-2, atol=1e-3)
        assert_close(g_up,   cpu_up,   rtol=1e-2, atol=1e-3)
        assert_close(g_lo,   cpu_lo,   rtol=1e-2, atol=1e-3)

    def test_prb_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024; N = 4; n=64; k=2; ro=0
        series = test_data['close'][:T].astype(np.float64)
        tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            tm[:, j] = series * (1.0 + 0.01 * j)
        cpu_m = np.zeros_like(tm); cpu_u=np.zeros_like(tm); cpu_l=np.zeros_like(tm)
        for j in range(N):
            m, u, l = ti.prb(tm[:, j], False, 10, n, k, ro, 2.0)
            cpu_m[:, j] = m; cpu_u[:, j] = u; cpu_l[:, j] = l

        m_h, u_h, l_h = ti.prb_cuda_many_series_one_param_dev(
            tm.astype(np.float32), cols=N, rows=T,
            smooth_data=False, smooth_period=10, regression_period=n, polynomial_order=k, regression_offset=ro, ndev=2.0,
        )
        g_m = cp.asnumpy(cp.asarray(m_h))
        g_u = cp.asnumpy(cp.asarray(u_h))
        g_l = cp.asnumpy(cp.asarray(l_h))

        assert g_m.shape == tm.shape
        assert_close(g_m, cpu_m, rtol=1e-2, atol=1e-3)
        assert_close(g_u, cpu_u, rtol=1e-2, atol=1e-3)
        assert_close(g_l, cpu_l, rtol=1e-2, atol=1e-3)

