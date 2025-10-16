"""Python binding tests for LINEARREG_INTERCEPT CUDA kernels."""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency
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
    if not hasattr(ti, "linearreg_intercept_cuda_batch_dev"):
        return False
    try:
        sample = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle = ti.linearreg_intercept_cuda_batch_dev(sample, period_range=(10, 14, 2))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  # pragma: no cover - availability probe only
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(
    not _cuda_available(), reason="CUDA not available or cuda bindings not built"
)
class TestLinregInterceptCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_linearreg_intercept_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"].astype(np.float64)
        sweep = (6, 30, 6)

        cpu = ti.linearreg_intercept_batch(close, sweep)
        cpu_values = cpu["values"]

        handle = ti.linearreg_intercept_cuda_batch_dev(close.astype(np.float32), sweep)
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu_values.shape

        mask = ~np.isnan(cpu_values)
        assert_close(
            gpu[mask],
            cpu_values[mask],
            rtol=8e-4,
            atol=2.5e-3,
            msg="CUDA LINEARREG_INTERCEPT mismatch vs CPU baseline",
        )

    def test_linearreg_intercept_cuda_many_series_one_param_matches_cpu(self, test_data):
        close = test_data["close"].astype(np.float64)
        period = 18
        cols = 4
        rows = close.shape[0]
        matrix = np.tile(close[:, None], (1, cols))
        for idx in range(cols):
            matrix[: idx + 10, idx] = np.nan
            matrix[:, idx] += idx * 0.19

        cpu_cols = []
        for idx in range(cols):
            cpu_cols.append(ti.linearreg_intercept(matrix[:, idx], period=period))
        cpu_values = np.column_stack(cpu_cols)

        handle = ti.linearreg_intercept_cuda_many_series_one_param_dev(
            matrix.astype(np.float32),
            cols,
            rows,
            period,
        )
        gpu = cp.asnumpy(cp.asarray(handle))
        assert gpu.shape == cpu_values.shape == (rows, cols)

        mask = ~np.isnan(cpu_values)
        assert_close(
            gpu[mask],
            cpu_values[mask],
            rtol=8e-4,
            atol=3e-3,
            msg="CUDA LINEARREG_INTERCEPT many-series mismatch vs CPU baseline",
        )

