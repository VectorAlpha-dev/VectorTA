"""
CUDA binding tests for VOSC (Volume Oscillator).
Skips gracefully when CUDA is unavailable or module not built.
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
    if not hasattr(ti, "vosc_cuda_batch_dev"):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle = ti.vosc_cuda_batch_dev(x, (2, 2, 0), (3, 3, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "driver" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or bindings missing")
class TestVoscCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_vosc_cuda_batch_matches_cpu(self, test_data):
        vol = test_data["volume"].astype(np.float64)
        s, l = 5, 34
        cpu = ti.vosc(vol, s, l)
        handle = ti.vosc_cuda_batch_dev(vol.astype(np.float32), (s, s, 0), (l, l, 0))
        gpu_row = cp.asnumpy(cp.asarray(handle))[0]
        assert_close(gpu_row, cpu, rtol=1e-5, atol=1e-6, msg="CUDA VOSC batch mismatch")

    def test_vosc_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 2048
        N = 6
        base = test_data["volume"][:T].astype(np.float64)
        v_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            v_tm[:, j] = base * (1.0 + 0.02 * j)

        s, l = 5, 34
        
        cpu_tm = np.zeros_like(v_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.vosc(v_tm[:, j].astype(np.float32).astype(np.float64), s, l)

        handle = ti.vosc_cuda_many_series_one_param_dev(v_tm.astype(np.float32), s, l)
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == v_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-4, atol=1e-5, msg="CUDA VOSC many-series mismatch")

