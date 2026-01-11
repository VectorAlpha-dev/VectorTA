"""
Python binding tests for PMA CUDA kernels (non-lag PMA variant).
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
    pytest.skip("Python module not built. Run 'maturin develop --features python,cuda' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, 'pma_cuda_batch_dev'):
        return False
    try:
        x = np.array([np.nan] * 6 + [1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        pred, trig = ti.pma_cuda_batch_dev(x.astype(np.float32))
        _ = (cp.asarray(pred), cp.asarray(trig))
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestPmaCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_pma_cuda_batch_matches_cpu(self, test_data):
        close = test_data['close']
        cpu_pred, cpu_trig = ti.pma(close)

        pred_h, trig_h = ti.pma_cuda_batch_dev(close.astype(np.float32))
        gpu_pred = cp.asnumpy(cp.asarray(pred_h))[0]
        gpu_trig = cp.asnumpy(cp.asarray(trig_h))[0]

        assert_close(gpu_pred, cpu_pred, rtol=2e-4, atol=2e-5, msg="PMA CUDA predict mismatch")
        assert_close(gpu_trig, cpu_trig, rtol=2e-4, atol=2e-5, msg="PMA CUDA trigger mismatch")

    def test_pma_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 5
        series = test_data['close'][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = series * (1.0 + 0.01 * j)

        cpu_pred_tm = np.zeros_like(data_tm)
        cpu_trig_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_pred_tm[:, j], cpu_trig_tm[:, j] = ti.pma(data_tm[:, j])

        pred_h, trig_h = ti.pma_cuda_many_series_one_param_dev(data_tm.astype(np.float32))
        gpu_pred_tm = cp.asnumpy(cp.asarray(pred_h))
        gpu_trig_tm = cp.asnumpy(cp.asarray(trig_h))

        assert gpu_pred_tm.shape == data_tm.shape
        assert gpu_trig_tm.shape == data_tm.shape
        assert_close(gpu_pred_tm, cpu_pred_tm, rtol=2e-4, atol=2e-5, msg="PMA CUDA predict many-series mismatch")
        assert_close(gpu_trig_tm, cpu_trig_tm, rtol=2e-4, atol=2e-5, msg="PMA CUDA trigger many-series mismatch")

