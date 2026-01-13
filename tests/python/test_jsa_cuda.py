"""Python binding tests for JSA CUDA kernels."""
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
    if not hasattr(ti, "jsa_cuda_batch_dev"):
        return False
    sample = np.array([np.nan, 1.0, 2.5, 3.0, 4.0], dtype=np.float32)
    try:
        handle = ti.jsa_cuda_batch_dev(sample, period_range=(3, 3, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or CUDA bindings not built")
class TestJsaCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_jsa_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"][:1024]
        period = 24

        cpu = ti.jsa(close, period=period)

        handle = ti.jsa_cuda_batch_dev(
            close.astype(np.float32),
            period_range=(period, period, 0),
        )
        gpu = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(
            gpu,
            cpu,
            rtol=1e-6,
            atol=1e-6,
            msg="JSA CUDA batch vs CPU mismatch",
        )

    def test_jsa_cuda_batch_sweep_matches_cpu(self, test_data):
        close = test_data["close"][:2048]
        sweep = (8, 48, 8)
        periods = list(range(sweep[0], sweep[1] + 1, sweep[2]))

        cpu_rows = [ti.jsa(close, period=p) for p in periods]
        cpu = np.vstack(cpu_rows)

        handle = ti.jsa_cuda_batch_dev(
            close.astype(np.float32),
            period_range=sweep,
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu.shape
        assert_close(
            gpu,
            cpu,
            rtol=1e-6,
            atol=1e-6,
            msg="JSA CUDA sweep mismatch",
        )

    def test_jsa_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        period = 20
        base = test_data["close"][:T]
        data_tm = np.full((T, N), np.nan, dtype=np.float64)
        for j in range(N):
            for t in range(j, T):
                x = base[t] if np.isfinite(base[t]) else 0.0
                data_tm[t, j] = np.sin(0.0017 * x + 0.01 * j) + 0.00021 * t

        cpu_tm = np.full_like(data_tm, np.nan)
        for j in range(N):
            cpu_tm[:, j] = ti.jsa(data_tm[:, j], period=period)

        handle = ti.jsa_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32),
            period=period,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=1e-6,
            atol=1e-6,
            msg="JSA CUDA many-series mismatch",
        )
