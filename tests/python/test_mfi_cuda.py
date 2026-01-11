"""Python binding tests for MFI CUDA kernels."""
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
    if not hasattr(ti, "mfi_cuda_batch_dev"):
        return False
    try:
        n = 64
        tp = np.linspace(100.0, 101.0, n, dtype=np.float32)
        vol = np.linspace(900.0, 1100.0, n, dtype=np.float32)
        
        tp[:4] = np.nan
        vol[:4] = np.nan
        handle = ti.mfi_cuda_batch_dev(tp, vol, (14, 14, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestMfiCuda:
    def test_mfi_cuda_batch_matches_cpu(self):
        data = load_test_data()
        h = data["high"].astype(np.float64)
        l = data["low"].astype(np.float64)
        c = data["close"].astype(np.float64)
        tp = (h + l + c) / 3.0
        vol = data["volume"].astype(np.float64)

        sweep = (5, 25, 5)
        
        tp_f32 = tp.astype(np.float32).astype(np.float64)
        vol_f32 = vol.astype(np.float32).astype(np.float64)
        cpu = ti.mfi_batch(tp_f32, vol_f32, sweep)
        cpu_values = cpu["values"]

        handle = ti.mfi_cuda_batch_dev(tp.astype(np.float32), vol.astype(np.float32), sweep)
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu_values.shape
        mask = ~np.isnan(cpu_values)
        assert_close(gpu[mask], cpu_values[mask], rtol=5e-4, atol=5e-3, msg="CUDA MFI batch mismatch vs CPU")

    def test_mfi_cuda_many_series_one_param_matches_cpu(self):
        data = load_test_data()
        h = data["high"].astype(np.float64)
        l = data["low"].astype(np.float64)
        c = data["close"].astype(np.float64)
        v = data["volume"].astype(np.float64)
        T = 1024
        N = 4
        tp_series = ((h + l + c) / 3.0)[:T]
        vol_series = v[:T]
        tp_tm = np.zeros((T, N), dtype=np.float64)
        vol_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            tp_tm[:, j] = tp_series * (1.0 + 0.01 * j)
            vol_tm[:, j] = vol_series * (1.0 + 0.02 * j)

        period = 14
        cpu_tm = np.zeros_like(tp_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.mfi(tp_tm[:, j].astype(np.float32).astype(np.float64),
                                   vol_tm[:, j].astype(np.float32).astype(np.float64), period)

        handle = ti.mfi_cuda_many_series_one_param_dev(
            tp_tm.astype(np.float32), vol_tm.astype(np.float32), N, T, period
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))
        assert gpu_tm.shape == tp_tm.shape
        mask = ~np.isnan(cpu_tm)
        assert_close(gpu_tm[mask], cpu_tm[mask], rtol=5e-4, atol=5e-3, msg="CUDA MFI many-series mismatch vs CPU")
