"""Python binding tests for TrAdjEMA CUDA kernels."""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency for CUDA validation
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
    if not hasattr(ti, 'tradjema_cuda_batch_dev'):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle = ti.tradjema_cuda_batch_dev(
            x,
            x,
            x,
            length_range=(4, 4, 0),
            mult_range=(10.0, 10.0, 0.0),
        )
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  # pragma: no cover - defensive skip
        msg = str(exc).lower()
        if "cuda not available" in msg or "no cuda" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or tradjema CUDA bindings missing")
class TestTradjemaCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_tradjema_cuda_batch_matches_cpu(self, test_data):
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        length = 32
        mult = 8.0

        high32 = high.astype(np.float32)
        low32 = low.astype(np.float32)
        close32 = close.astype(np.float32)

        cpu = ti.tradjema(high32.astype(np.float64), low32.astype(np.float64), close32.astype(np.float64), length, mult)

        handle = ti.tradjema_cuda_batch_dev(
            high32,
            low32,
            close32,
            length_range=(length, length, 0),
            mult_range=(mult, mult, 0.0),
        )

        gpu = cp.asarray(handle)
        gpu_first = cp.asnumpy(gpu)[0]

        assert_close(
            gpu_first,
            cpu,
            rtol=1e-5,
            atol=1e-5,
            msg="TrAdjEMA CUDA batch vs CPU mismatch",
        )

    def test_tradjema_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        length = 28
        mult = 9.5

        high_src = test_data['high'][:T].astype(np.float64)
        low_src = test_data['low'][:T].astype(np.float64)
        close_src = test_data['close'][:T].astype(np.float64)

        high_tm = np.empty((T, N), dtype=np.float64)
        low_tm = np.empty((T, N), dtype=np.float64)
        close_tm = np.empty((T, N), dtype=np.float64)
        for j in range(N):
            scale = 1.0 + 0.05 * j
            high_tm[:, j] = high_src * scale
            low_tm[:, j] = low_src * scale
            close_tm[:, j] = close_src * scale

        high_tm32 = high_tm.astype(np.float32)
        low_tm32 = low_tm.astype(np.float32)
        close_tm32 = close_tm.astype(np.float32)

        cpu_tm = np.empty_like(close_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.tradjema(
                high_tm32[:, j].astype(np.float64),
                low_tm32[:, j].astype(np.float64),
                close_tm32[:, j].astype(np.float64),
                length,
                mult,
            )

        handle = ti.tradjema_cuda_many_series_one_param_dev(
            high_tm32,
            low_tm32,
            close_tm32,
            length,
            mult,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == close_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=1e-5,
            atol=1e-5,
            msg="TrAdjEMA CUDA many-series vs CPU mismatch",
        )
