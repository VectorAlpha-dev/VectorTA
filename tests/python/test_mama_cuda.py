"""
Python CUDA binding tests for MAMA (MESA Adaptive Moving Average).
Skips gracefully when CUDA or bindings are unavailable.
"""
import numpy as np
import pytest

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
    if not hasattr(ti, "mama_cuda_batch_dev"):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        m, f = ti.mama_cuda_batch_dev(
            x.astype(np.float32),
            fast_limit_range=(0.5, 0.5, 0.0),
            slow_limit_range=(0.05, 0.05, 0.0),
        )

        _ = cp.asarray(m)
        _ = cp.asarray(f)
        return True
    except Exception as e:
        msg = str(e).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False

        return True


@pytest.mark.skipif(
    not _cuda_available(), reason="CUDA not available or cuda bindings not built"
)
class TestMamaCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_mama_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"][:2048].astype(np.float64)
        fast, slow = 0.5, 0.05


        cpu_m, cpu_f = ti.mama(close, fast, slow)


        m_handle, f_handle = ti.mama_cuda_batch_dev(
            close.astype(np.float32),
            fast_limit_range=(fast, fast, 0.0),
            slow_limit_range=(slow, slow, 0.0),
        )

        m_gpu = cp.asnumpy(cp.asarray(m_handle))[0]
        f_gpu = cp.asnumpy(cp.asarray(f_handle))[0]

        assert_close(m_gpu, cpu_m, rtol=1e-5, atol=1e-6, msg="MAMA mismatch (CUDA vs CPU)")
        assert_close(f_gpu, cpu_f, rtol=1e-5, atol=1e-6, msg="FAMA mismatch (CUDA vs CPU)")

    def test_mama_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        base = test_data["close"][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = base * (1.0 + 0.01 * j)

        fast, slow = 0.45, 0.06


        cpu_m_tm = np.zeros_like(data_tm)
        cpu_f_tm = np.zeros_like(data_tm)
        for j in range(N):
            m, f = ti.mama(data_tm[:, j], fast, slow)
            cpu_m_tm[:, j] = m
            cpu_f_tm[:, j] = f


        m_handle, f_handle = ti.mama_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32), fast, slow
        )
        m_gpu_tm = cp.asnumpy(cp.asarray(m_handle))
        f_gpu_tm = cp.asnumpy(cp.asarray(f_handle))

        assert m_gpu_tm.shape == data_tm.shape
        assert f_gpu_tm.shape == data_tm.shape
        assert_close(m_gpu_tm, cpu_m_tm, rtol=1e-5, atol=1e-6, msg="MAMA many-series mismatch")
        assert_close(f_gpu_tm, cpu_f_tm, rtol=1e-5, atol=1e-6, msg="FAMA many-series mismatch")

