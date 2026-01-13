"""Python binding tests for SuperSmoother 3-Pole CUDA kernels."""
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
    if not hasattr(ti, 'supersmoother_3_pole_cuda_batch_dev'):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.5, 4.2, 5.0], dtype=np.float32)
        handle = ti.supersmoother_3_pole_cuda_batch_dev(
            x,
            period_range=(6, 6, 0),
        )
        _ = cp.asarray(handle)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if 'cuda not available' in msg or 'no cuda' in msg:
            return False
        return True


@pytest.mark.skipif(
    not _cuda_available(),
    reason="CUDA not available or supersmoother_3_pole CUDA bindings missing",
)
class TestSupersmoother3PoleCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_supersmoother_3_pole_cuda_batch_matches_cpu(self, test_data):
        close = test_data['close']
        period = 32

        close32 = close.astype(np.float32)
        close_quant = close32.astype(np.float64)

        cpu = ti.supersmoother_3_pole_batch(close_quant, (period, period, 0))

        handle = ti.supersmoother_3_pole_cuda_batch_dev(
            close32,
            period_range=(period, period, 0),
        )

        gpu = cp.asnumpy(cp.asarray(handle))
        gpu_first = gpu[0]

        assert gpu_first.shape == close.shape
        assert_close(
            gpu_first,
            cpu['values'][0],
            rtol=1e-5,
            atol=1e-6,
            msg="SuperSmoother3Pole CUDA batch vs CPU mismatch",
        )

    def test_supersmoother_3_pole_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        period = 24

        base = test_data['close'][:T].astype(np.float64)
        tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            scale = 1.0 + 0.04 * j
            tm[:, j] = base * scale

        tm32 = tm.astype(np.float32)
        tm_quant = tm32.astype(np.float64)

        cpu_tm = np.zeros_like(tm_quant)
        for j in range(N):
            cpu_tm[:, j] = ti.supersmoother_3_pole(tm_quant[:, j], period)

        handle = ti.supersmoother_3_pole_cuda_many_series_one_param_dev(
            tm32,
            period,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=1e-5,
            atol=1e-6,
            msg="SuperSmoother3Pole CUDA many-series vs CPU mismatch",
        )
