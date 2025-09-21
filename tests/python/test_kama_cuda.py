"""Python binding tests for KAMA CUDA kernels."""
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
    if not hasattr(ti, 'kama_cuda_batch_dev'):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        handle = ti.kama_cuda_batch_dev(
            x,
            period_range=(3, 3, 0),
        )
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  # pragma: no cover - defensive skip
        msg = str(exc).lower()
        if "cuda not available" in msg or "no cuda" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or KAMA CUDA bindings missing")
class TestKamaCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_kama_cuda_batch_matches_cpu(self, test_data):
        close = test_data['close']
        period = 20

        cpu = ti.kama(close.astype(np.float64), period)

        handle = ti.kama_cuda_batch_dev(
            close.astype(np.float32),
            period_range=(period, period, 0),
        )
        gpu = cp.asarray(handle)
        gpu_first = cp.asnumpy(gpu)[0]

        assert_close(
            gpu_first,
            cpu,
            rtol=1e-5,
            atol=1e-6,
            msg="KAMA CUDA batch vs CPU mismatch",
        )

    def test_kama_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 5
        period = 18
        base = test_data['close'][:T].astype(np.float64)

        data_tm = np.empty((T, N), dtype=np.float64)
        for j in range(N):
            drift = 1.0 + 0.015 * j
            data_tm[:, j] = base * drift + j * 0.03

        cpu_tm = np.empty_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.kama(data_tm[:, j], period)

        handle = ti.kama_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32),
            period,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=1e-5,
            atol=1e-6,
            msg="KAMA CUDA many-series vs CPU mismatch",
        )
