"""
Python CUDA binding tests for Chandelier Exit (CE).
Skips gracefully when CUDA/CuPy are unavailable or CUDA feature not built.
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

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, 'chandelier_exit_cuda_batch_dev'):
        return False
    try:

        x = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32)
        handle, _meta = ti.chandelier_exit_cuda_batch_dev(x, x, x, (3, 3, 0), (2.0, 2.0, 0.0), True)
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'ptx' in msg or 'nvcc' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestChandelierExitCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_ce_cuda_batch_matches_cpu(self, test_data):
        high = test_data['high'][:2048].astype(np.float64)
        low = test_data['low'][:2048].astype(np.float64)
        close = test_data['close'][:2048].astype(np.float64)


        period, mult, use_close = 22, 3.0, True
        cpu_long, cpu_short = ti.chandelier_exit(high, low, close, period, mult, use_close)


        handle, meta = ti.chandelier_exit_cuda_batch_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
            period_range=(period, period, 0),
            mult_range=(mult, mult, 0.0),
            use_close=use_close,
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert_close(gpu[0], cpu_long, rtol=5e-4, atol=1e-5, msg="CE CUDA batch long vs CPU")
        assert_close(gpu[1], cpu_short, rtol=5e-4, atol=1e-5, msg="CE CUDA batch short vs CPU")

    def test_ce_cuda_many_series_one_param_matches_cpu(self, test_data):

        T, N = 1024, 4
        close_tm = np.zeros((T, N), dtype=np.float64)
        base = test_data['close'][:T].astype(np.float64)
        for j in range(N):
            close_tm[:, j] = base * (1.0 + 0.01 * j)

        high_tm = close_tm + 0.15
        low_tm = close_tm - 0.15

        period, mult, use_close = 22, 3.0, True


        cpu_long = np.full_like(close_tm, np.nan)
        cpu_short = np.full_like(close_tm, np.nan)
        for j in range(N):
            l, s = ti.chandelier_exit(high_tm[:, j], low_tm[:, j], close_tm[:, j], period, mult, use_close)
            cpu_long[:, j] = l
            cpu_short[:, j] = s


        handle = ti.chandelier_exit_cuda_many_series_one_param_dev(
            high_tm.astype(np.float32).ravel(),
            low_tm.astype(np.float32).ravel(),
            close_tm.astype(np.float32).ravel(),
            N, T, period, mult, use_close,
        )
        gpu = cp.asnumpy(cp.asarray(handle))
        long_tm = gpu[:T, :]
        short_tm = gpu[T:, :]

        assert_close(long_tm, cpu_long, rtol=5e-4, atol=1e-5, msg="CE CUDA many-series long vs CPU")
        assert_close(short_tm, cpu_short, rtol=5e-4, atol=1e-5, msg="CE CUDA many-series short vs CPU")

