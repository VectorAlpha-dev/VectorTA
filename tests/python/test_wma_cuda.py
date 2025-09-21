"""Python binding tests for WMA CUDA kernels."""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency for CUDA path
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
    if not hasattr(ti, 'wma_cuda_batch_dev'):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        handle = ti.wma_cuda_batch_dev(
            x.astype(np.float32),
            period_range=(4, 4, 0),
        )
        _ = cp.asarray(handle)
        return True
    except Exception as e:  # pragma: no cover - defensive skip
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestWmaCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_wma_cuda_batch_matches_cpu(self, test_data):
        close = test_data['close']
        period = 20

        cpu = ti.wma(close, period)

        handle = ti.wma_cuda_batch_dev(
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
            msg="WMA CUDA batch vs CPU mismatch",
        )

    def test_wma_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        period = 24
        series = test_data['close'][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = series * (1.0 + 0.03 * j)

        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.wma(data_tm[:, j], period)

        handle = ti.wma_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32), period
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=1e-5,
            atol=1e-6,
            msg="WMA CUDA many-series vs CPU mismatch",
        )
