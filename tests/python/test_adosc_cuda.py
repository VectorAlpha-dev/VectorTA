"""
Python binding tests for ADOSC CUDA kernels.
Skips gracefully when CUDA or the CUDA bindings are unavailable.
"""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

try:
    import my_project as ti
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python,cuda' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, 'adosc_cuda_batch_dev'):
        return False
    try:
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        # minimal OHLCV that avoids div-by-zero (high>low) and volume>0
        high = x + 1
        low = x - 1
        close = x
        volume = np.ones_like(x)
        handle = ti.adosc_cuda_batch_dev(high, low, close, volume, (3, 3, 0), (10, 10, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestAdoscCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_adosc_cuda_batch_matches_cpu(self, test_data):
        high = test_data['high'].astype(np.float64)
        low = test_data['low'].astype(np.float64)
        close = test_data['close'].astype(np.float64)
        volume = test_data['volume'].astype(np.float64)

        # CPU baseline
        cpu = ti.adosc(high, low, close, volume, short_period=3, long_period=10)

        # CUDA single-combo batch
        handle = ti.adosc_cuda_batch_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
            volume.astype(np.float32),
            (3, 3, 0),
            (10, 10, 0),
        )
        gpu_row = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(gpu_row, cpu, rtol=2e-3, atol=1e-5, msg="ADOSC CUDA batch vs CPU mismatch")

    def test_adosc_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 2048
        N = 4
        high = np.zeros((T, N), dtype=np.float64)
        low = np.zeros((T, N), dtype=np.float64)
        close = np.zeros((T, N), dtype=np.float64)
        volume = np.zeros((T, N), dtype=np.float64)
        base = test_data['close'][:T].astype(np.float64)
        for j in range(N):
            off = 0.1 + 0.01 * j
            high[:, j] = base + off
            low[:, j] = base - off
            close[:, j] = base
            volume[:, j] = 1000.0 + 5.0 * j

        short, long = 5, 21

        cpu_tm = np.zeros_like(close)
        for j in range(N):
            cpu_tm[:, j] = ti.adosc(high[:, j], low[:, j], close[:, j], volume[:, j], short, long)

        handle = ti.adosc_cuda_many_series_one_param_dev(
            high.astype(np.float32).ravel(),
            low.astype(np.float32).ravel(),
            close.astype(np.float32).ravel(),
            volume.astype(np.float32).ravel(),
            N,
            T,
            short,
            long,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == cpu_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=2e-3, atol=1e-5, msg="ADOSC CUDA TM vs CPU mismatch")

