"""Python binding tests for MWDX CUDA kernels."""
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
    if not hasattr(ti, "mwdx_cuda_batch_dev"):
        return False
    sample = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    try:
        handle = ti.mwdx_cuda_batch_dev(sample, factor_range=(0.2, 0.2, 0.0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  # pragma: no cover - best effort probe
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or CUDA bindings not built")
class TestMwdxCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_mwdx_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"][:1024]
        factor = 0.35

        cpu = ti.mwdx(close, factor)

        handle = ti.mwdx_cuda_batch_dev(
            close.astype(np.float32),
            factor_range=(factor, factor, 0.0),
        )
        gpu = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(
            gpu,
            cpu,
            rtol=1e-5,
            atol=1e-5,
            msg="MWDX CUDA batch vs CPU mismatch",
        )

    def test_mwdx_cuda_batch_sweep_matches_cpu(self, test_data):
        close = test_data["close"][:1536]
        sweep = (0.1, 0.9, 0.2)

        factors = []
        start, end, step = sweep
        if step == 0.0 or abs(start - end) < 1e-12:
            factors.append(start)
        else:
            value = start
            while value <= end + 1e-12:
                factors.append(value)
                value += step
        cpu_rows = [ti.mwdx(close, factor=f) for f in factors]
        cpu = np.vstack(cpu_rows)

        handle = ti.mwdx_cuda_batch_dev(
            close.astype(np.float32),
            factor_range=sweep,
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu.shape
        assert_close(
            gpu,
            cpu,
            rtol=1e-5,
            atol=1e-5,
            msg="MWDX CUDA sweep mismatch",
        )

    def test_mwdx_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 5
        factor = 0.27

        base_series = test_data["close"][:T]
        data_tm = np.full((T, N), np.nan, dtype=np.float64)
        for j in range(N):
            for t in range(j, T):
                x = base_series[t] if np.isfinite(base_series[t]) else 0.0
                data_tm[t, j] = np.sin(0.0025 * x + 0.01 * j) + 0.00021 * t

        cpu_tm = np.full_like(data_tm, np.nan)
        for j in range(N):
            cpu_tm[:, j] = ti.mwdx(data_tm[:, j], factor)

        handle = ti.mwdx_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32),
            factor=factor,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=1e-5,
            atol=1e-5,
            msg="MWDX CUDA many-series mismatch",
        )
