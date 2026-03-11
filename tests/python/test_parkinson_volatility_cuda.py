"""
Python binding tests for Parkinson volatility CUDA kernels.
Skips gracefully when CUDA or CuPy is unavailable, or when cuda bindings were not built.
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

from test_utils import assert_close, load_test_data


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "parkinson_volatility_cuda_batch_dev"):
        return False
    try:
        data = load_test_data()
        out = ti.parkinson_volatility_cuda_batch_dev(
            data["high"][:64].astype(np.float32),
            data["low"][:64].astype(np.float32),
            (8, 8, 0),
        )
        _ = cp.asarray(out["volatility"])
        _ = cp.asarray(out["variance"])
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestParkinsonVolatilityCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_parkinson_cuda_batch_matches_cpu(self, test_data):
        high = test_data["high"][:4096].astype(np.float64)
        low = test_data["low"][:4096].astype(np.float64)
        period_range = (8, 24, 8)

        cpu = ti.parkinson_volatility_batch(high, low, period_range)
        cpu_vol = cpu["volatility"].astype(np.float32)
        cpu_var = cpu["variance"].astype(np.float32)

        out = ti.parkinson_volatility_cuda_batch_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            period_range,
        )
        gpu_vol = cp.asnumpy(cp.asarray(out["volatility"]))
        gpu_var = cp.asnumpy(cp.asarray(out["variance"]))

        assert_close(gpu_vol, cpu_vol, rtol=1e-4, atol=1e-5, msg="CUDA volatility mismatch")
        assert_close(gpu_var, cpu_var, rtol=1e-4, atol=1e-5, msg="CUDA variance mismatch")

    def test_parkinson_cuda_many_series_one_param_matches_cpu(self, test_data):
        t = 1024
        n = 4
        high = test_data["high"][:t].astype(np.float64)
        low = test_data["low"][:t].astype(np.float64)
        high_tm = np.stack([high * (1.0 + 0.01 * j) for j in range(n)], axis=1)
        low_tm = np.stack([low * (1.0 + 0.01 * j) for j in range(n)], axis=1)
        period = 14

        cpu_vol = np.zeros_like(high_tm)
        cpu_var = np.zeros_like(high_tm)
        for j in range(n):
            vol, var = ti.parkinson_volatility(
                np.ascontiguousarray(high_tm[:, j]),
                np.ascontiguousarray(low_tm[:, j]),
                period,
            )
            cpu_vol[:, j] = vol
            cpu_var[:, j] = var

        out = ti.parkinson_volatility_cuda_many_series_one_param_dev(
            high_tm.astype(np.float32),
            low_tm.astype(np.float32),
            period,
        )
        gpu_vol = cp.asnumpy(cp.asarray(out["volatility"]))
        gpu_var = cp.asnumpy(cp.asarray(out["variance"]))

        assert gpu_vol.shape == high_tm.shape
        assert gpu_var.shape == low_tm.shape
        assert_close(gpu_vol, cpu_vol, rtol=1e-4, atol=1e-5, msg="CUDA TM volatility mismatch")
        assert_close(gpu_var, cpu_var, rtol=1e-4, atol=1e-5, msg="CUDA TM variance mismatch")
