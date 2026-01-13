"""Python binding tests for TrendFlex CUDA kernels."""
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
    if not hasattr(ti, "trendflex_cuda_batch_dev"):
        return False
    try:
        sample = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle, meta = ti.trendflex_cuda_batch_dev(sample, period_range=(5, 5, 0))
        _ = cp.asarray(handle)
        _ = np.asarray(meta["periods"], dtype=np.int64)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestTrendFlexCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_trendflex_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"].astype(np.float64)
        sweep = (6, 30, 6)

        cpu = ti.trendflex_batch(close, sweep)
        cpu_values = cpu["values"]
        cpu_periods = cpu["periods"]

        handle, meta = ti.trendflex_cuda_batch_dev(close.astype(np.float32), sweep)
        gpu = cp.asnumpy(cp.asarray(handle))
        gpu_periods = np.asarray(meta["periods"], dtype=np.int64)

        assert gpu.shape == cpu_values.shape
        assert np.array_equal(gpu_periods, cpu_periods)

        mask = ~np.isnan(cpu_values)
        assert_close(
            gpu[mask],
            cpu_values[mask],
            rtol=1e-4,
            atol=8e-4,
            msg="CUDA TrendFlex mismatch vs CPU baseline",
        )

    def test_trendflex_cuda_many_series_one_param_matches_cpu(self, test_data):
        close = test_data["close"].astype(np.float64)
        period = 18
        cols = 4
        rows = close.shape[0]
        matrix = np.tile(close[:, None], (1, cols))
        for idx in range(cols):
            matrix[: idx + 5, idx] = np.nan
            matrix[:, idx] += idx * 0.12

        cpu_cols = []
        for idx in range(cols):
            cpu_cols.append(ti.trendflex(matrix[:, idx], period=period))
        cpu_values = np.column_stack(cpu_cols)

        handle = ti.trendflex_cuda_many_series_one_param_dev(
            matrix.astype(np.float32),
            period,
        )
        gpu = cp.asnumpy(cp.asarray(handle))
        assert gpu.shape == cpu_values.shape == (rows, cols)

        mask = ~np.isnan(cpu_values)
        assert_close(
            gpu[mask],
            cpu_values[mask],
            rtol=1.5e-4,
            atol=9e-4,
            msg="CUDA TrendFlex many-series mismatch vs CPU baseline",
        )
