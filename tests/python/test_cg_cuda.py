"""Python binding tests for CG CUDA kernels."""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency
    cp = None

try:
    import my_project as ti
except ImportError:  # pragma: no cover - module not built yet
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python,cuda' first",
        allow_module_level=True,
    )

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "cg_cuda_batch_dev"):
        return False
    sample = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    try:
        handle = ti.cg_cuda_batch_dev(sample, period_range=(5, 5, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  # pragma: no cover - best effort probe
        msg = str(exc).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or CUDA bindings not built")
class TestCgCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_cg_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"][:2048]
        period = 10

        cpu = ti.cg(close, period=period)

        handle = ti.cg_cuda_batch_dev(
            close.astype(np.float32),
            period_range=(period, period, 0),
        )
        gpu = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(
            gpu,
            cpu,
            rtol=1e-4,
            atol=1e-4,
            msg="CG CUDA batch vs CPU mismatch",
        )

    def test_cg_cuda_batch_sweep_matches_cpu(self, test_data):
        close = test_data["close"][:1536]
        sweep = (5, 20, 5)

        rows = []
        for p in range(sweep[0], sweep[1] + 1, sweep[2]):
            rows.append(ti.cg(close, period=p))
        cpu = np.vstack(rows)

        handle = ti.cg_cuda_batch_dev(
            close.astype(np.float32),
            period_range=sweep,
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu.shape
        assert_close(gpu, cpu, rtol=1e-4, atol=1e-4, msg="CG CUDA sweep mismatch")

    def test_cg_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 6
        base = test_data["close"][:T]
        tm = np.full((T, N), np.nan, dtype=np.float64)
        for j in range(N):
            for t in range(j + 1, T):
                x = base[t] if np.isfinite(base[t]) else 0.0
                tm[t, j] = np.sin(0.003 * x + 0.01 * j) + 0.0001 * t

        period = 12
        cpu = np.full_like(tm, np.nan)
        for j in range(N):
            cpu[:, j] = ti.cg(tm[:, j], period=period)

        handle = ti.cg_cuda_many_series_one_param_dev(
            tm.astype(np.float32),
            cols=N,
            rows=T,
            period=period,
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == tm.shape
        assert_close(gpu, cpu, rtol=1e-4, atol=1e-4, msg="CG CUDA many-series mismatch")

