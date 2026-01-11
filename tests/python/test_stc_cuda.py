"""Python binding tests for STC CUDA kernels."""
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
    if not hasattr(ti, 'stc_cuda_batch_dev'):
        return False
    try:
        x = np.array([np.nan] * 40 + [1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        handle, meta = ti.stc_cuda_batch_dev(
            x.astype(np.float32),
            fast_period_range=(23, 23, 0),
            slow_period_range=(50, 50, 0),
            k_period_range=(10, 10, 0),
            d_period_range=(3, 3, 0),
        )
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'ptx' in msg or 'nvcc' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestStcCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_stc_cuda_batch_matches_cpu(self, test_data):
        close = test_data['close'].astype(np.float64)
        params = dict(fast_period=23, slow_period=50, k_period=10, d_period=3)

        cpu = ti.stc(close, **params)

        handle, meta = ti.stc_cuda_batch_dev(
            close.astype(np.float32),
            fast_period_range=(params['fast_period'], params['fast_period'], 0),
            slow_period_range=(params['slow_period'], params['slow_period'], 0),
            k_period_range=(params['k_period'], params['k_period'], 0),
            d_period_range=(params['d_period'], params['d_period'], 0),
        )

        gpu = cp.asarray(handle)
        gpu_first = cp.asnumpy(gpu)[0]

        assert_close(
            gpu_first,
            cpu,
            rtol=2e-3,
            atol=2e-3,
            msg="STC CUDA batch vs CPU mismatch",
        )

    def test_stc_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1536
        N = 6
        params = dict(fast_period=23, slow_period=50, k_period=10, d_period=3)
        series = test_data['close'][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = series * (1.0 + 0.01 * j)

        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.stc(data_tm[:, j], **params)

        handle = ti.stc_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32),
            cols=data_tm.shape[1],
            rows=data_tm.shape[0],
            **params,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=2e-3,
            atol=2e-3,
            msg="STC CUDA many-series vs CPU mismatch",
        )

