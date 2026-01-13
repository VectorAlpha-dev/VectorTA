"""Python binding tests for VAMA CUDA kernels."""
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
    if not hasattr(ti, "vama_cuda_batch_dev"):
        return False
    sample = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    try:
        handle = ti.vama_cuda_batch_dev(
            sample,
            base_period_range=(5, 5, 0),
            vol_period_range=(3, 3, 0),
        )
        _ = cp.asarray(handle)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or CUDA bindings not built")
class TestVamaCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_vama_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"][:1024]
        base_period = 21
        vol_period = 13

        cpu = ti.vama(
            close,
            base_period=base_period,
            vol_period=vol_period,
            smoothing=False,
        )

        handle = ti.vama_cuda_batch_dev(
            close.astype(np.float32),
            base_period_range=(base_period, base_period, 0),
            vol_period_range=(vol_period, vol_period, 0),
        )

        gpu = cp.asnumpy(cp.asarray(handle))[0]
        assert_close(
            gpu,
            cpu,
            rtol=1e-5,
            atol=1e-5,
            msg="VAMA CUDA batch vs CPU mismatch",
        )

    def test_vama_cuda_batch_sweep_matches_cpu(self, test_data):
        close = test_data["close"][:1536]
        base_range = (9, 21, 4)
        vol_range = (5, 17, 4)

        cpu_rows = []
        for base in range(base_range[0], base_range[1] + 1, base_range[2]):
            for vol in range(vol_range[0], vol_range[1] + 1, vol_range[2]):
                cpu_rows.append(
                    ti.vama(
                        close,
                        base_period=base,
                        vol_period=vol,
                        smoothing=False,
                    )
                )
        cpu = np.vstack(cpu_rows)

        handle = ti.vama_cuda_batch_dev(
            close.astype(np.float32),
            base_period_range=base_range,
            vol_period_range=vol_range,
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu.shape
        assert_close(
            gpu,
            cpu,
            rtol=1e-5,
            atol=1e-5,
            msg="VAMA CUDA sweep mismatch",
        )

    def test_vama_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        base_series = test_data["close"][:T].astype(np.float64)
        data_tm = np.full((T, N), np.nan, dtype=np.float64)
        for j in range(N):
            start = j
            for t in range(start, T):
                x = base_series[t] if np.isfinite(base_series[t]) else 0.0
                data_tm[t, j] = x * (1.0 + 0.02 * j) + 0.001 * t

        base_period = 21
        vol_period = 13

        cpu_tm = np.full_like(data_tm, np.nan)
        for j in range(N):
            cpu_tm[:, j] = ti.vama(
                data_tm[:, j],
                base_period=base_period,
                vol_period=vol_period,
                smoothing=False,
            )

        handle = ti.vama_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32),
            base_period=base_period,
            vol_period=vol_period,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=1e-5,
            atol=1e-5,
            msg="VAMA CUDA many-series mismatch",
        )
