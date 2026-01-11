"""CUDA bindings tests for the DTI indicator (GPU path).

Requires: `maturin develop --features python,cuda` and a CUDA device.
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
    if not hasattr(ti, "dti_cuda_batch_dev"):
        return False
    try:
        
        h = np.array([np.nan, 11.0, 12.0, 13.0, 14.0], dtype=np.float32)
        l = h - 1.0
        handle, meta = ti.dti_cuda_batch_dev(h, l, (3, 3, 0), (2, 2, 0), (2, 2, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "driver" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or DTI CUDA bindings missing")
class TestDtiCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_dti_cuda_batch_matches_cpu(self, test_data):
        high = test_data["high"].astype(np.float64)
        low = test_data["low"].astype(np.float64)
        r, s, u = 14, 10, 5

        
        cpu = ti.dti(high.astype(np.float32).astype(np.float64),
                     low.astype(np.float32).astype(np.float64), r, s, u)

        handle, meta = ti.dti_cuda_batch_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            (r, r, 0), (s, s, 0), (u, u, 0)
        )
        assert meta["r"].tolist() == [r]
        assert meta["s"].tolist() == [s]
        assert meta["u"].tolist() == [u]

        gpu = cp.asnumpy(cp.asarray(handle))[0]
        assert_close(gpu, cpu, rtol=1e-4, atol=1e-4, msg="CUDA DTI batch mismatch")

    def test_dti_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 4096
        N = 4
        high = test_data["high"][:T].astype(np.float64)
        low = test_data["low"][:T].astype(np.float64)
        high_tm = np.zeros((T, N), dtype=np.float64)
        low_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            
            base = (1.0 + 0.02 * j)
            high_tm[:, j] = high * base
            low_tm[:, j] = (low * base).clip(max=high_tm[:, j] - 1e-6)

        r, s, u = 14, 10, 5

        
        cpu_tm = np.zeros_like(high_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.dti(
                high_tm[:, j].astype(np.float32).astype(np.float64),
                low_tm[:, j].astype(np.float32).astype(np.float64),
                r, s, u,
            )

        handle = ti.dti_cuda_many_series_one_param_dev(
            high_tm.astype(np.float32), low_tm.astype(np.float32), r, s, u
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == cpu_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-4, atol=1e-4, msg="CUDA DTI many-series mismatch")

