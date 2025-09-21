"""Python binding tests for EPMA CUDA kernels."""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency for CUDA path
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
    if not hasattr(ti, 'epma_cuda_batch_dev'):
        return False
    try:
        data = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        handle = ti.epma_cuda_batch_dev(
            data,
            period_range=(4, 4, 0),
            offset_range=(1, 1, 0),
        )
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  # pragma: no cover - defensive skip
        msg = str(exc).lower()
        if 'cuda not available' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or bindings missing")
class TestEpmaCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_epma_cuda_batch_matches_cpu(self, test_data):
        close = test_data['close'].astype(np.float64)
        period_range = (6, 30, 4)
        offset_range = (1, 9, 2)

        cpu_out = ti.epma_batch(close, period_range, offset_range)
        cpu_vals = np.asarray(cpu_out['values'], dtype=np.float64)

        handle = ti.epma_cuda_batch_dev(
            close.astype(np.float32),
            period_range=period_range,
            offset_range=offset_range,
        )
        gpu_vals = cp.asnumpy(cp.asarray(handle))

        assert gpu_vals.shape == cpu_vals.shape
        assert_close(
            gpu_vals,
            cpu_vals,
            rtol=2e-5,
            atol=1e-6,
            msg="EPMA CUDA batch vs CPU mismatch",
        )

    def test_epma_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 5
        period = 24
        offset = 6
        base = test_data['close'][:T].astype(np.float64)
        data_tm = np.full((T, N), np.nan, dtype=np.float64)
        for j in range(N):
            shifted = base * (1.0 + 0.05 * j)
            data_tm[:, j] = shifted

        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.epma(data_tm[:, j], period, offset)

        handle = ti.epma_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32), period, offset
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=2e-5,
            atol=1e-6,
            msg="EPMA CUDA many-series vs CPU mismatch",
        )
