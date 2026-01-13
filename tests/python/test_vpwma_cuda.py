"""Python binding tests for VPWMA CUDA kernels."""
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
    if not hasattr(ti, "vpwma_cuda_batch_dev"):
        return False
    try:
        sample = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle, meta = ti.vpwma_cuda_batch_dev(
            sample,
            period_range=(3, 3, 0),
            power_range=(0.5, 0.5, 0.0),
        )
        _ = cp.asarray(handle)
        _ = np.asarray(meta["periods"])
        _ = np.asarray(meta["powers"])
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestVpwmaCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_vpwma_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"].astype(np.float64)
        period_range = (5, 25, 5)
        power_range = (0.2, 0.8, 0.2)

        cpu = ti.vpwma_batch(close, period_range, power_range)
        cpu_values = cpu["values"]
        cpu_periods = cpu["periods"]
        cpu_powers = cpu["powers"]

        handle, meta = ti.vpwma_cuda_batch_dev(
            close.astype(np.float32),
            period_range,
            power_range,
        )
        gpu = cp.asnumpy(cp.asarray(handle))
        gpu_periods = np.asarray(meta["periods"], dtype=np.int64)
        gpu_powers = np.asarray(meta["powers"], dtype=np.float64)

        assert gpu.shape == cpu_values.shape
        assert np.array_equal(gpu_periods, cpu_periods)
        assert np.allclose(gpu_powers, cpu_powers, atol=1e-9)

        mask = ~np.isnan(cpu_values)
        assert_close(
            gpu[mask],
            cpu_values[mask],
            rtol=1e-4,
            atol=7e-4,
            msg="CUDA VPWMA mismatch vs CPU baseline",
        )

    def test_vpwma_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        base = test_data["close"].astype(np.float64)
        data_tm = np.full((T, N), np.nan, dtype=np.float64)
        for j in range(N):
            data_tm[j:, j] = base[: T - j] * (1.0 + 0.01 * j)

        period = 14
        power = 0.6

        cpu_tm = np.empty_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.vpwma(data_tm[:, j], period, power)

        handle = ti.vpwma_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32), period, power
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        mask = ~np.isnan(cpu_tm)
        assert_close(
            gpu_tm[mask],
            cpu_tm[mask],
            rtol=1e-4,
            atol=7e-4,
            msg="CUDA VPWMA many-series mismatch vs CPU baseline",
        )
