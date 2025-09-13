"""
Python binding tests for ALMA CUDA kernels.
These mirror the CPU tests and use the same reference values where applicable.
Skips gracefully when CUDA is unavailable or CUDA feature not built.
"""
import pytest
import numpy as np

try:
    import my_project as ti
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python,cuda' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    # Try a minimal call to detect CUDA availability and bindings presence
    if not hasattr(ti, 'alma_cuda_batch'):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float64)
        _ = ti.alma_cuda_batch(x, (3, 3, 0), (0.85, 0.85, 0.0), (6.0, 6.0, 0.0))
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        # Other errors mean CUDA path exists; consider available
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestAlmaCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_alma_cuda_batch_matches_cpu(self, test_data):
        close = test_data['close']
        period, offset, sigma = 9, 0.85, 6.0

        # CPU baseline
        cpu = ti.alma(close, period, offset, sigma)

        # CUDA single-combo batch
        res = ti.alma_cuda_batch(
            close,
            period_range=(period, period, 0),
            offset_range=(offset, offset, 0.0),
            sigma_range=(sigma, sigma, 0.0),
        )

        assert 'values' in res
        gpu = res['values'][0]

        # Compare entire row with modest tolerance (fp32 vs fp64)
        assert_close(gpu, cpu, rtol=1e-5, atol=1e-6, msg="CUDA batch vs CPU mismatch")

    # multi-stream variant removed

    def test_alma_cuda_many_series_one_param_matches_cpu(self, test_data):
        # Build small time-major matrix (T,N) with varied columns
        T = 1024
        N = 4
        series = test_data['close'][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = series * (1.0 + 0.01 * j)

        period = 14
        offset = 0.85
        sigma = 6.0

        # CPU baseline per series
        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.alma(data_tm[:, j], period, offset, sigma)

        # CUDA
        gpu_tm = ti.alma_cuda_many_series_one_param(data_tm, period, offset, sigma)

        assert gpu_tm.shape == data_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-5, atol=1e-6, msg="CUDA many-series vs CPU mismatch")
