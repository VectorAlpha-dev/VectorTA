"""
Python binding tests for NAMA CUDA kernels.
Skips gracefully when CUDA is unavailable or CUDA feature not built.
"""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

try:
    import my_project as ti
except ImportError:  # pragma: no cover
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python,cuda' first",
        allow_module_level=True,
    )

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "nama_cuda_batch_dev"):
        return False
    # Probe minimal call; tolerate environment errors as unavailability
    probe = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    try:
        handle = ti.nama_cuda_batch_dev(probe, period_range=(3, 3, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  # pragma: no cover
        msg = str(exc).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or CUDA bindings not built")
class TestNamaCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_nama_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"][:2048]
        period = 30

        cpu = ti.nama(close, period=period)

        handle = ti.nama_cuda_batch_dev(
            close.astype(np.float32),
            period_range=(period, period, 0),
        )
        gpu_row = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(
            gpu_row,
            cpu,
            rtol=1e-5,
            atol=1e-5,
            msg="NAMA CUDA batch vs CPU mismatch",
        )

    def test_nama_cuda_batch_sweep_matches_cpu(self, test_data):
        close = test_data["close"][:3072]
        start, end, step = 10, 40, 5
        periods = list(range(start, end + 1, step))

        cpu_rows = [ti.nama(close, period=p) for p in periods]
        cpu = np.vstack(cpu_rows)

        handle = ti.nama_cuda_batch_dev(
            close.astype(np.float32),
            period_range=(start, end, step),
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu.shape
        assert_close(
            gpu,
            cpu,
            rtol=1e-5,
            atol=1e-5,
            msg="NAMA CUDA sweep mismatch",
        )

    def test_nama_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        period = 30
        series = test_data["close"][:T].astype(np.float64)
        data_tm = np.full((T, N), np.nan, dtype=np.float64)
        for j in range(N):
            # Create varied but deterministic columns with NaN prefixes
            for t in range(j + 5, T):
                base = series[t] if np.isfinite(series[t]) else 0.0
                data_tm[t, j] = np.sin(0.0021 * base + 0.017 * j) + 0.00029 * t

        cpu_tm = np.full_like(data_tm, np.nan)
        for j in range(N):
            cpu_tm[:, j] = ti.nama(data_tm[:, j], period=period)

        handle = ti.nama_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32), period
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=1e-5,
            atol=1e-5,
            msg="NAMA CUDA many-series mismatch",
        )

