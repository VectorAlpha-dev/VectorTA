"""
Python binding tests for Bollinger Bands Width (BBW) CUDA kernels.
Skips gracefully when CUDA is unavailable or the bindings are not built.
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
    if not hasattr(ti, 'bollinger_bands_width_cuda_batch_dev'):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        handle, _meta = ti.bollinger_bands_width_cuda_batch_dev(
            x.astype(np.float32),
            period_range=(3, 3, 0),
            devup_range=(2.0, 2.0, 0.0),
            devdn_range=(2.0, 2.0, 0.0),
        )
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestBbwCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_bbw_cuda_batch_matches_cpu(self, test_data):
        close = test_data['close']
        period, devup, devdn = 20, 2.0, 2.0


        cpu = ti.bollinger_bands_width(close, period, devup, devdn, matype='sma', devtype=0)


        handle, meta = ti.bollinger_bands_width_cuda_batch_dev(
            close.astype(np.float32),
            period_range=(period, period, 0),
            devup_range=(devup, devup, 0.0),
            devdn_range=(devdn, devdn, 0.0),
        )
        gpu = cp.asnumpy(cp.asarray(handle))[0]


        assert_close(gpu, cpu, rtol=1e-4, atol=2e-5, msg="BBW CUDA batch vs CPU mismatch")

    def test_bbw_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 2048
        N = 4
        series = test_data['close'][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = series * (1.0 + 0.01 * j)

        period, devup, devdn = 20, 2.0, 2.0


        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.bollinger_bands_width(
                data_tm[:, j], period, devup, devdn, matype='sma', devtype=0
            )

        handle = ti.bollinger_bands_width_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32), N, T, period, devup, devdn
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-4, atol=2e-5, msg="BBW CUDA many-series vs CPU mismatch")

