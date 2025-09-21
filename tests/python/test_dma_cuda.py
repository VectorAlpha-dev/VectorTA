"""
Python binding tests for DMA CUDA kernels.
Mirror the scalar DMA tests to ensure the GPU batch path stays in sync.
"""

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
    if not hasattr(ti, "dma_cuda_batch_dev"):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32)
        handle = ti.dma_cuda_batch_dev(
            x,
            hull_length_range=(3, 3, 0),
            ema_length_range=(5, 5, 0),
            ema_gain_limit_range=(10, 10, 0),
            hull_ma_type="WMA",
        )
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        msg = str(exc).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or bindings missing")
class TestDmaCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_dma_cuda_batch_wma_matches_cpu(self, test_data):
        close = test_data["close"].astype(np.float64)

        sweep = dict(
            hull_length_range=(7, 15, 4),
            ema_length_range=(16, 28, 4),
            ema_gain_limit_range=(10, 30, 10),
            hull_ma_type="WMA",
        )

        cpu = ti.dma_batch(close, **sweep)
        cpu_values = cpu["values"]

        handle = ti.dma_cuda_batch_dev(
            close.astype(np.float32),
            sweep["hull_length_range"],
            sweep["ema_length_range"],
            sweep["ema_gain_limit_range"],
            sweep["hull_ma_type"],
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu_values.shape
        assert_close(gpu, cpu_values, rtol=1e-3, atol=1e-3, msg="DMA CUDA WMA mismatch")

    def test_dma_cuda_batch_ema_matches_cpu(self, test_data):
        close = test_data["close"].astype(np.float64)

        sweep = dict(
            hull_length_range=(10, 18, 4),
            ema_length_range=(20, 32, 6),
            ema_gain_limit_range=(5, 25, 5),
            hull_ma_type="EMA",
        )

        cpu = ti.dma_batch(close, **sweep)
        cpu_values = cpu["values"]

        handle = ti.dma_cuda_batch_dev(
            close.astype(np.float32),
            sweep["hull_length_range"],
            sweep["ema_length_range"],
            sweep["ema_gain_limit_range"],
            sweep["hull_ma_type"],
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu_values.shape
        assert_close(gpu, cpu_values, rtol=1e-3, atol=1e-3, msg="DMA CUDA EMA mismatch")

    def test_dma_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        base = test_data["close"][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = base * (1.0 + 0.02 * j)

        hull_length = 21
        ema_length = 34
        ema_gain_limit = 30
        hull_ma_type = "WMA"

        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.dma(
                data_tm[:, j],
                hull_length,
                ema_length,
                ema_gain_limit,
                hull_ma_type,
            )

        handle = ti.dma_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32),
            hull_length,
            ema_length,
            ema_gain_limit,
            hull_ma_type,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=1e-3,
            atol=1e-3,
            msg="DMA CUDA many-series mismatch",
        )
