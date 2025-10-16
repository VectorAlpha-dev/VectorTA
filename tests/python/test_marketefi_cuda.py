"""Python binding tests for MarketEFI CUDA kernels."""
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
    if not hasattr(ti, "marketefi_cuda_batch_dev"):
        return False
    try:
        d = load_test_data()
        h = d["high"].astype(np.float32)
        l = d["low"].astype(np.float32)
        v = d["volume"].astype(np.float32)
        handle = ti.marketefi_cuda_batch_dev(h, l, v)
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  # pragma: no cover - detection only
        msg = str(exc).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or bindings not built")
class TestMarketefiCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_marketefi_cuda_batch_matches_cpu(self, test_data):
        h = test_data["high"].astype(np.float64)
        l = test_data["low"].astype(np.float64)
        v = test_data["volume"].astype(np.float64)

        cpu = ti.marketefi(h, l, v)
        handle = ti.marketefi_cuda_batch_dev(h.astype(np.float32), l.astype(np.float32), v.astype(np.float32))
        gpu = cp.asnumpy(cp.asarray(handle))
        assert gpu.shape == cpu.shape
        assert_close(gpu, cpu, rtol=1e-5, atol=1e-6, msg="CUDA marketefi batch mismatch")

    def test_marketefi_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 512
        N = 3
        base_h = test_data["high"][:T].astype(np.float64)
        base_l = test_data["low"][:T].astype(np.float64)
        base_v = test_data["volume"][:T].astype(np.float64)
        high_tm = np.zeros((T, N), dtype=np.float64)
        low_tm = np.zeros((T, N), dtype=np.float64)
        vol_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            high_tm[:, j] = base_h * (1.0 + 0.02 * j)
            low_tm[:, j] = base_l * (1.0 + 0.02 * j)
            vol_tm[:, j] = base_v * (1.0 + 0.05 * j)
        cpu_tm = np.zeros_like(high_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.marketefi(high_tm[:, j], low_tm[:, j], vol_tm[:, j])

        handle = ti.marketefi_cuda_many_series_one_param_dev(
            high_tm.astype(np.float32), low_tm.astype(np.float32), vol_tm.astype(np.float32)
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))
        assert gpu_tm.shape == cpu_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-5, atol=1e-6, msg="CUDA marketefi many-series mismatch")

