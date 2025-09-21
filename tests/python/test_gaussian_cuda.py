"""Python binding tests for Gaussian CUDA kernels."""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional CUDA runtime
    cp = None

try:
    import my_project as ti
except ImportError:  # pragma: no cover - module not built yet
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python,cuda' first",
        allow_module_level=True,
    )

from test_utils import assert_close, load_test_data


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "gaussian_cuda_batch_dev"):
        return False
    try:
        sample = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle = ti.gaussian_cuda_batch_dev(sample, (8, 8, 0), (2, 2, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  # pragma: no cover - defensive skip path
        msg = str(exc).lower()
        if "cuda not available" in msg or "no cuda device" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestGaussianCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_gaussian_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"].astype(np.float64)
        period = 16
        poles = 3

        cpu = ti.gaussian(close, period, poles)

        handle = ti.gaussian_cuda_batch_dev(
            close.astype(np.float32),
            (period, period, 0),
            (poles, poles, 0),
        )
        gpu = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(gpu, cpu, rtol=2e-4, atol=1e-4, msg="CUDA Gaussian batch mismatch")

    def test_gaussian_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 512
        N = 4
        base = test_data["close"][:T].astype(np.float64)
        data_tm = np.full((T, N), np.nan, dtype=np.float64)
        for j in range(N):
            data_tm[j:, j] = base[: T - j] * (1.0 + 0.02 * j)

        period = 20
        poles = 2

        cpu_tm = np.full_like(data_tm, np.nan)
        for j in range(N):
            cpu_tm[:, j] = ti.gaussian(data_tm[:, j], period, poles)

        handle = ti.gaussian_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32), period, poles
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=2e-4,
            atol=1e-4,
            msg="CUDA Gaussian many-series mismatch",
        )
