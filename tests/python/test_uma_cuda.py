"""Python binding tests for UMA CUDA kernels."""
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
    if not hasattr(ti, "uma_cuda_batch_dev"):
        return False
    try:
        data = load_test_data()
        prices = data["close"].astype(np.float32)
        handle = ti.uma_cuda_batch_dev(
            prices,
            (1.0, 1.0, 0.0),
            (5, 5, 0),
            (20, 20, 0),
            (3, 3, 0),
        )
        _ = cp.asarray(handle)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestUmaCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_uma_cuda_batch_no_volume_matches_cpu(self, test_data):
        prices = test_data["close"].astype(np.float64)
        sweep = (
            (1.0, 1.5, 0.5),
            (5, 7, 1),
            (18, 22, 2),
            (2, 4, 1),
        )

        cpu = ti.uma_batch(
            prices,
            sweep[0],
            sweep[1],
            sweep[2],
            sweep[3],
        )["values"]

        handle = ti.uma_cuda_batch_dev(
            prices.astype(np.float32),
            sweep[0],
            sweep[1],
            sweep[2],
            sweep[3],
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu.shape
        assert_close(gpu, cpu, rtol=1e-4, atol=1e-5, msg="CUDA UMA batch mismatch (no volume)")

    def test_uma_cuda_batch_with_volume_matches_cpu(self, test_data):
        prices = test_data["close"].astype(np.float64)
        volumes = test_data["volume"].astype(np.float64)
        sweep = (
            (1.0, 1.0, 0.0),
            (6, 6, 0),
            (20, 24, 2),
            (3, 3, 0),
        )

        cpu = ti.uma_batch(
            prices,
            sweep[0],
            sweep[1],
            sweep[2],
            sweep[3],
            volume=volumes,
        )["values"]

        handle = ti.uma_cuda_batch_dev(
            prices.astype(np.float32),
            sweep[0],
            sweep[1],
            sweep[2],
            sweep[3],
            volume_f32=volumes.astype(np.float32),
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu.shape
        assert_close(gpu, cpu, rtol=1e-4, atol=1e-5, msg="CUDA UMA batch mismatch (with volume)")

    def test_uma_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 256
        N = 4
        prices_tm = np.full((T, N), np.nan, dtype=np.float64)
        volumes_tm = np.full_like(prices_tm, np.nan)
        for j in range(N):
            first_valid = j + 3
            for t in range(first_valid, T):
                x = float(t) + 0.25 * j
                prices_tm[t, j] = np.sin(x * 0.0021) + 0.0005 * x
                volumes_tm[t, j] = 400.0 + np.cos(x * 0.0017) * (j + 1) * 25.0

        accelerator = 1.3
        min_length = 6
        max_length = 24
        smooth_length = 3

        cpu_tm = np.full_like(prices_tm, np.nan)
        for j in range(N):
            cpu_tm[:, j] = ti.uma(
                prices_tm[:, j],
                accelerator,
                min_length,
                max_length,
                smooth_length,
                volume=volumes_tm[:, j],
            )

        handle = ti.uma_cuda_many_series_one_param_dev(
            prices_tm.astype(np.float32),
            accelerator,
            min_length,
            max_length,
            smooth_length,
            volume_tm_f32=volumes_tm.astype(np.float32),
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == cpu_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-4, atol=1e-5, msg="CUDA UMA many-series mismatch (with volume)")

    def test_uma_cuda_many_series_one_param_no_volume_matches_cpu(self, test_data):
        T = 192
        N = 3
        prices_tm = np.full((T, N), np.nan, dtype=np.float64)
        for j in range(N):
            first_valid = 2 * j + 4
            for t in range(first_valid, T):
                x = float(t) + 0.4 * j
                prices_tm[t, j] = np.cos(x * 0.0018) + 0.0003 * x

        accelerator = 1.1
        min_length = 5
        max_length = 18
        smooth_length = 2

        cpu_tm = np.full_like(prices_tm, np.nan)
        for j in range(N):
            cpu_tm[:, j] = ti.uma(
                prices_tm[:, j],
                accelerator,
                min_length,
                max_length,
                smooth_length,
            )

        handle = ti.uma_cuda_many_series_one_param_dev(
            prices_tm.astype(np.float32),
            accelerator,
            min_length,
            max_length,
            smooth_length,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == cpu_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-4, atol=1e-5, msg="CUDA UMA many-series mismatch (no volume)")
