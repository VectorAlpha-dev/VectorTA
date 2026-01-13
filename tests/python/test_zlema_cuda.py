"""Python binding tests for ZLEMA CUDA kernels."""
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
    if not hasattr(ti, "zlema_cuda_batch_dev"):
        return False
    try:
        sample = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32)
        handle, meta = ti.zlema_cuda_batch_dev(sample, period_range=(3, 3, 0))
        _ = cp.asarray(handle)
        _ = np.asarray(meta["periods"])
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestZlemaCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_zlema_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"].astype(np.float64)
        sweep = (5, 25, 5)

        cpu = ti.zlema_batch(close, sweep)
        cpu_values = cpu["values"]
        cpu_periods = cpu["periods"]

        handle, meta = ti.zlema_cuda_batch_dev(close.astype(np.float32), sweep)
        gpu = cp.asnumpy(cp.asarray(handle))
        gpu_periods = np.asarray(meta["periods"], dtype=np.int64)

        assert gpu.shape == cpu_values.shape
        assert np.array_equal(gpu_periods, cpu_periods)

        mask = ~np.isnan(cpu_values)
        assert_close(
            gpu[mask],
            cpu_values[mask],
            rtol=1e-4,
            atol=5e-4,
            msg="CUDA ZLEMA mismatch vs CPU baseline",
        )

    def test_zlema_cuda_many_series_one_param_matches_cpu(self):
        if not hasattr(ti, "zlema_cuda_many_series_one_param_dev"):
            pytest.skip("zlema_cuda_many_series_one_param_dev not available")

        cols = 6
        rows = 1024

        tm = np.full((rows, cols), np.nan, dtype=np.float64)
        for s in range(cols):
            for t in range(s, rows):
                x = float(t) + float(s) * 0.13
                tm[t, s] = np.sin(x * 0.003) + 0.0004 * x

        period = 13


        cpu_tm = np.full((rows, cols), np.nan, dtype=np.float64)
        for s in range(cols):
            series = tm[:, s]
            cpu_series = ti.zlema(series, period)
            cpu_tm[:, s] = cpu_series

        handle = ti.zlema_cuda_many_series_one_param_dev(tm.astype(np.float32), period)
        gpu_tm = cp.asnumpy(cp.asarray(handle)).astype(np.float64)


        mask = ~np.isnan(cpu_tm)
        assert_close(gpu_tm[mask], cpu_tm[mask], rtol=1e-4, atol=5e-4,
                     msg="CUDA ZLEMA many-series mismatch vs CPU baseline")
