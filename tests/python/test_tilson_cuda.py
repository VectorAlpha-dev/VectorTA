"""Python binding tests for Tilson CUDA kernels."""
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
    if not hasattr(ti, "tilson_cuda_batch_dev"):
        return False
    sample = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    try:
        handle = ti.tilson_cuda_batch_dev(
            sample,
            period_range=(5, 5, 0),
            volume_factor_range=(0.0, 0.0, 0.0),
        )
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  # pragma: no cover - best effort probe
        msg = str(exc).lower()
        if "cuda not available" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or CUDA bindings not built")
class TestTilsonCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_tilson_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"][:1024]
        period = 12
        volume_factor = 0.4

        cpu = ti.tilson(close, period=period, volume_factor=volume_factor)

        handle = ti.tilson_cuda_batch_dev(
            close.astype(np.float32),
            period_range=(period, period, 0),
            volume_factor_range=(volume_factor, volume_factor, 0.0),
        )
        gpu = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(
            gpu,
            cpu,
            rtol=2e-4,
            atol=2e-4,
            msg="Tilson CUDA batch vs CPU mismatch",
        )

    def test_tilson_cuda_batch_sweep_matches_cpu(self, test_data):
        close = test_data["close"][:1536]
        period_range = (5, 20, 3)
        volume_range = (0.0, 0.8, 0.2)

        cpu_rows = []
        for period in range(period_range[0], period_range[1] + 1, period_range[2]):
            vf = volume_range[0]
            while vf <= volume_range[1] + 1e-12:
                cpu_rows.append(ti.tilson(close, period=period, volume_factor=vf))
                vf += volume_range[2]
        cpu = np.vstack(cpu_rows)

        handle = ti.tilson_cuda_batch_dev(
            close.astype(np.float32),
            period_range=period_range,
            volume_factor_range=volume_range,
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu.shape
        assert_close(
            gpu,
            cpu,
            rtol=2e-4,
            atol=2e-4,
            msg="Tilson CUDA sweep mismatch",
        )

    def test_tilson_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 5
        base = test_data["close"][:T].astype(np.float64)
        data_tm = np.full((T, N), np.nan, dtype=np.float64)
        for j in range(N):
            for t in range(j, T):
                val = base[t] if np.isfinite(base[t]) else 0.0
                data_tm[t, j] = val * (1.0 + 0.01 * j) + 0.0003 * t

        period = 10
        volume_factor = 0.35

        cpu_tm = np.full_like(data_tm, np.nan)
        for j in range(N):
            cpu_tm[:, j] = ti.tilson(
                data_tm[:, j],
                period=period,
                volume_factor=volume_factor,
            )

        handle = ti.tilson_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32),
            period=period,
            volume_factor=volume_factor,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=2e-4,
            atol=2e-4,
            msg="Tilson CUDA many-series mismatch",
        )
