"""Python binding tests for VWMA CUDA kernels."""
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
    if not hasattr(ti, "vwma_cuda_batch_dev"):
        return False
    try:
        data = load_test_data()
        prices = data["close"].astype(np.float32)
        volumes = data["volume"].astype(np.float32)
        handle = ti.vwma_cuda_batch_dev(prices, volumes, (10, 10, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestVwmaCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_vwma_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"]
        volume = test_data["volume"]
        sweep = (6, 20, 2)

        cpu = ti.vwma_batch(close, volume, sweep)["values"]

        handle = ti.vwma_cuda_batch_dev(
            close.astype(np.float32),
            volume.astype(np.float32),
            sweep,
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu.shape
        assert_close(gpu, cpu, rtol=1e-5, atol=1e-6, msg="CUDA VWMA batch mismatch")

    def test_vwma_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 512
        N = 3
        base_close = test_data["close"][:T].astype(np.float64)
        base_volume = test_data["volume"][:T].astype(np.float64)

        prices_tm = np.zeros((T, N), dtype=np.float64)
        volumes_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            prices_tm[:, j] = base_close * (1.0 + 0.02 * j)
            volumes_tm[:, j] = base_volume * (1.0 + 0.05 * j)

        period = 14

        cpu_tm = np.zeros_like(prices_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.vwma(prices_tm[:, j], volumes_tm[:, j], period)

        handle = ti.vwma_cuda_many_series_one_param_dev(
            prices_tm.astype(np.float32),
            volumes_tm.astype(np.float32),
            period,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == prices_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-5, atol=1e-6, msg="CUDA VWMA many-series mismatch")
