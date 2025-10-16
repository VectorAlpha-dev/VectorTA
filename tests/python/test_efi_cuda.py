"""Python binding tests for EFI CUDA kernels."""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover
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
    if not hasattr(ti, "efi_cuda_batch_dev"):
        return False
    try:
        d = load_test_data()
        p = d["close"].astype(np.float32)
        v = d["volume"].astype(np.float32)
        h = ti.efi_cuda_batch_dev(p, v, (13, 13, 0))
        _ = cp.asarray(h)
        return True
    except Exception as exc:  # pragma: no cover - detection only
        msg = str(exc).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or bindings not built")
class TestEfiCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_efi_cuda_batch_matches_cpu(self, test_data):
        price = test_data["close"].astype(np.float64)
        volume = test_data["volume"].astype(np.float64)
        sweep = (8, 20, 3)

        cpu = ti.efi_batch(price, volume, sweep)["values"]
        handle = ti.efi_cuda_batch_dev(price.astype(np.float32), volume.astype(np.float32), sweep)
        gpu = cp.asnumpy(cp.asarray(handle))
        assert gpu.shape == cpu.shape
        assert_close(gpu, cpu, rtol=1e-4, atol=1e-5, msg="CUDA EFI batch mismatch")

    def test_efi_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 512
        N = 3
        base_p = test_data["close"][:T].astype(np.float64)
        base_v = test_data["volume"][:T].astype(np.float64)
        prices_tm = np.zeros((T, N), dtype=np.float64)
        volumes_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            prices_tm[:, j] = base_p * (1.0 + 0.02 * j)
            volumes_tm[:, j] = base_v * (1.0 + 0.05 * j)
        period = 13
        cpu_tm = np.zeros_like(prices_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.efi(prices_tm[:, j], volumes_tm[:, j], period)
        handle = ti.efi_cuda_many_series_one_param_dev(
            prices_tm.astype(np.float32), volumes_tm.astype(np.float32), period
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))
        assert gpu_tm.shape == prices_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-4, atol=1e-5, msg="CUDA EFI many-series mismatch")

