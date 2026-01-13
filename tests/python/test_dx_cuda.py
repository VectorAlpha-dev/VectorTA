"""
Python binding tests for DX CUDA kernels.
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
    if not hasattr(ti, 'dx_cuda_batch_dev'):
        return False
    try:

        close = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32)
        high = close + 0.1
        low = close - 0.1
        handle, meta = ti.dx_cuda_batch_dev(high, low, close, period_range=(3, 3, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'ptx' in msg or 'nvcc' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestDxCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        d = load_test_data()

        close = d['close'].astype(np.float64)
        high = close.copy()
        low = close.copy()
        for i in range(close.size):
            v = close[i]
            if np.isnan(v):
                continue
            x = float(i) * 0.0025
            off = abs(0.002 * np.sin(x)) + 0.15
            high[i] = v + off
            low[i] = v - off
        return {'high': high, 'low': low, 'close': close}

    def test_dx_cuda_batch_matches_cpu(self, test_data):
        h = test_data['high']
        l = test_data['low']
        c = test_data['close']
        period = 14


        cpu = ti.dx(h, l, c, period)


        handle, meta = ti.dx_cuda_batch_dev(
            h.astype(np.float32), l.astype(np.float32), c.astype(np.float32),
            period_range=(period, period, 0)
        )
        gpu_row = cp.asnumpy(cp.asarray(handle))[0]


        assert_close(gpu_row, cpu, rtol=1e-3, atol=1e-3, msg="DX CUDA batch vs CPU mismatch")

    def test_dx_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 2048
        N = 8
        h = test_data['high'][:T]
        l = test_data['low'][:T]
        c = test_data['close'][:T]
        H = np.zeros((T, N), dtype=np.float64)
        L = np.zeros((T, N), dtype=np.float64)
        C = np.zeros((T, N), dtype=np.float64)
        for j in range(N):

            H[:, j] = h * (1.0 + 0.002 * j)
            L[:, j] = l * (1.0 + 0.002 * j)
            C[:, j] = c * (1.0 + 0.002 * j)

        period = 14


        cpu_tm = np.zeros_like(C)
        for j in range(N):
            cpu_tm[:, j] = ti.dx(H[:, j], L[:, j], C[:, j], period)

        handle = ti.dx_cuda_many_series_one_param_dev(
            H.astype(np.float32), L.astype(np.float32), C.astype(np.float32),
            cols=N, rows=T, period=period
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))
        assert gpu_tm.shape == cpu_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-3, atol=1e-3, msg="DX CUDA many-series vs CPU mismatch")

