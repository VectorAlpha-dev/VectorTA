"""
Python binding tests for Buff Averages CUDA kernels.
Skips gracefully when CUDA or CuPy is unavailable.
"""
import numpy as np
import pytest

try:
    import cupy as cp
except Exception:  
    
    
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
    if not hasattr(ti, "buff_averages_cuda_batch_dev"):
        return False
    try:
        price = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        volume = np.array([np.nan, 10.0, 11.0, 12.0, 13.0], dtype=np.float32)
        fast_handle, slow_handle = ti.buff_averages_cuda_batch_dev(
            price,
            volume,
            fast_range=(3, 3, 0),
            slow_range=(5, 5, 0),
        )
        _ = cp.asarray(fast_handle)
        _ = cp.asarray(slow_handle)
        return True
    except Exception as exc:  
        message = str(exc).lower()
        if "cuda not available" in message or "ptx" in message or "nvcc" in message:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestBuffAveragesCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_buff_averages_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"][:2048].astype(np.float64)
        volume = test_data["volume"][:2048].astype(np.float64)

        fast_range = (5, 15, 5)
        slow_range = (20, 40, 10)

        cpu = ti.buff_averages_batch(close, volume, fast_range, slow_range)
        fast_cpu = cpu["fast"]
        slow_cpu = cpu["slow"]

        fast_handle, slow_handle = ti.buff_averages_cuda_batch_dev(
            close.astype(np.float32),
            volume.astype(np.float32),
            fast_range,
            slow_range,
        )

        fast_gpu = cp.asnumpy(cp.asarray(fast_handle))
        slow_gpu = cp.asnumpy(cp.asarray(slow_handle))

        assert fast_gpu.shape == fast_cpu.shape
        assert slow_gpu.shape == slow_cpu.shape

        assert_close(fast_gpu, fast_cpu, rtol=1e-5, atol=1e-6, msg="CUDA fast vs CPU mismatch")
        assert_close(slow_gpu, slow_cpu, rtol=1e-5, atol=1e-6, msg="CUDA slow vs CPU mismatch")
