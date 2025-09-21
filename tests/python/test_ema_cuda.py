"""Python binding tests for EMA CUDA kernels."""
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
    if not hasattr(ti, "ema_cuda_batch_dev"):
        return False
    sample = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    try:
        handle = ti.ema_cuda_batch_dev(sample, period_range=(5, 5, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  # pragma: no cover - best effort probe
        msg = str(exc).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or CUDA bindings not built")
class TestEmaCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_ema_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"][:1024]
        period = 21

        cpu = ti.ema(close, period=period)

        handle = ti.ema_cuda_batch_dev(
            close.astype(np.float32),
            period_range=(period, period, 0),
        )
        gpu = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(
            gpu,
            cpu,
            rtol=1e-5,
            atol=1e-5,
            msg="EMA CUDA batch vs CPU mismatch",
        )

    def test_ema_cuda_batch_sweep_matches_cpu(self, test_data):
        close = test_data["close"][:1536]
        sweep = dict(period_range=(5, 45, 5))

        periods = list(range(sweep["period_range"][0], sweep["period_range"][1] + 1, sweep["period_range"][2]))

        cpu_rows = [ti.ema(close, period=p) for p in periods]
        cpu = np.vstack(cpu_rows)

        handle = ti.ema_cuda_batch_dev(
            close.astype(np.float32),
            period_range=sweep["period_range"],
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu.shape
        assert_close(
            gpu,
            cpu,
            rtol=1e-5,
            atol=1e-5,
            msg="EMA CUDA sweep mismatch",
        )

    def test_ema_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        period = 21
        base_series = test_data["close"][:T]
        data_tm = np.full((T, N), np.nan, dtype=np.float64)
        for j in range(N):
            for t in range(j, T):
                x = base_series[t] if np.isfinite(base_series[t]) else 0.0
                data_tm[t, j] = np.sin(0.0019 * x + 0.01 * j) + 0.00037 * t

        cpu_tm = np.full_like(data_tm, np.nan)
        for j in range(N):
            cpu_tm[:, j] = ti.ema(data_tm[:, j], period=period)

        handle = ti.ema_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32),
            period=period,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=1e-5,
            atol=1e-5,
            msg="EMA CUDA many-series mismatch",
        )
