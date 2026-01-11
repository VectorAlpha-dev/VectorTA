"""Python binding tests for SINWMA CUDA kernels."""
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
    if not hasattr(ti, 'sinwma_cuda_batch_dev'):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle = ti.sinwma_cuda_batch_dev(
            x,
            period_range=(4, 4, 0),
        )
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "no cuda" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or sinwma CUDA bindings missing")
class TestSinwmaCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_sinwma_cuda_batch_matches_cpu(self, test_data):
        close = test_data['close']
        period = 18

        cpu = ti.sinwma(close.astype(np.float64), period)

        handle = ti.sinwma_cuda_batch_dev(
            close.astype(np.float32),
            period_range=(period, period, 0),
        )
        gpu = cp.asarray(handle)
        gpu_first = cp.asnumpy(gpu)[0]

        assert_close(
            gpu_first,
            cpu,
            rtol=1e-5,
            atol=1e-6,
            msg="SINWMA CUDA batch vs CPU mismatch",
        )

    def test_sinwma_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        period = 16
        base = test_data['close'][:T].astype(np.float64)

        data_tm = np.empty((T, N), dtype=np.float64)
        for j in range(N):
            scale = 1.0 + 0.02 * j
            data_tm[:, j] = base * scale

        cpu_tm = np.empty_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.sinwma(data_tm[:, j], period)

        handle = ti.sinwma_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32),
            period,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=1e-5,
            atol=1e-6,
            msg="SINWMA CUDA many-series vs CPU mismatch",
        )
