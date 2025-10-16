"""
Python binding tests for RVI CUDA kernels.
Skips gracefully when CUDA is unavailable or CUDA feature is not built.
"""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # optional dependency for CUDA path
    cp = None

try:
    import my_project as ti
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python,cuda' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, 'rvi_cuda_batch_dev'):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float64)
        handle, _meta = ti.rvi_cuda_batch_dev(
            x.astype(np.float32),
            period_range=(3, 3, 0),
            ma_len_range=(3, 3, 0),
            matype_range=(1, 1, 0),
            devtype_range=(0, 0, 0),
        )
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestRviCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_rvi_cuda_batch_matches_cpu(self, test_data):
        close = test_data['close']
        period, ma_len, matype, devtype = 10, 14, 1, 0

        # CPU baseline
        cpu = ti.rvi(close, period, ma_len, matype, devtype)

        # CUDA single-combo batch
        handle, meta = ti.rvi_cuda_batch_dev(
            close.astype(np.float32),
            period_range=(period, period, 0),
            ma_len_range=(ma_len, ma_len, 0),
            matype_range=(matype, matype, 0),
            devtype_range=(devtype, devtype, 0),
        )

        gpu = cp.asarray(handle)
        gpu_first = cp.asnumpy(gpu)[0]

        # Compare entire row with modest tolerance (fp32 vs fp64)
        assert_close(gpu_first, cpu, rtol=1e-4, atol=2e-5, msg="CUDA batch vs CPU mismatch")

    def test_rvi_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 2048
        N = 4
        series = test_data['close'][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = series * (1.0 + 0.01 * j)

        period, ma_len, matype, devtype = 10, 14, 1, 0

        # CPU baseline per series
        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.rvi(data_tm[:, j], period, ma_len, matype, devtype)

        # CUDA
        handle = ti.rvi_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32).ravel(), data_tm.shape[1], data_tm.shape[0], period, ma_len, matype, devtype
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-4, atol=2e-5, msg="CUDA many-series vs CPU mismatch")

