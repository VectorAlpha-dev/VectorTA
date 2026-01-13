"""
Python binding tests for ERI CUDA kernels.
Skips gracefully when CUDA is unavailable or CUDA feature not built.
"""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import my_project as ti
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python,cuda' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, 'eri_cuda_batch_dev'):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float64)
        h = x + 0.25
        l = x - 0.25
        bull, bear = ti.eri_cuda_batch_dev(
            h.astype(np.float32),
            l.astype(np.float32),
            x.astype(np.float32),
            period_range=(3, 3, 0),
            ma_type="ema",
        )
        _ = cp.asarray(bull); _ = cp.asarray(bear)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestEriCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_eri_cuda_batch_matches_cpu(self, test_data):
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        period = 13
        ma_type = "ema"


        bull_cpu, bear_cpu = ti.eri(high, low, close, period=period, ma_type=ma_type)


        bull_dev, bear_dev = ti.eri_cuda_batch_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
            period_range=(period, period, 0),
            ma_type=ma_type,
        )

        bull_gpu = cp.asnumpy(cp.asarray(bull_dev))[0]
        bear_gpu = cp.asnumpy(cp.asarray(bear_dev))[0]


        assert_close(bull_gpu, bull_cpu, rtol=1e-4, atol=1e-5, msg="ERI bull: CUDA batch vs CPU mismatch")
        assert_close(bear_gpu, bear_cpu, rtol=1e-4, atol=1e-5, msg="ERI bear: CUDA batch vs CPU mismatch")

    def test_eri_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        high = test_data['high'][:T].astype(np.float64)
        low = test_data['low'][:T].astype(np.float64)
        close = test_data['close'][:T].astype(np.float64)


        high_tm = np.zeros((T, N), dtype=np.float64)
        low_tm = np.zeros((T, N), dtype=np.float64)
        close_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            high_tm[:, j] = high * (1.0 + 0.01 * j)
            low_tm[:, j] = low * (1.0 + 0.01 * j)
            close_tm[:, j] = close * (1.0 + 0.01 * j)

        period = 14
        ma_type = "ema"


        bull_cpu = np.zeros_like(close_tm)
        bear_cpu = np.zeros_like(close_tm)
        for j in range(N):
            bull, bear = ti.eri(high_tm[:, j], low_tm[:, j], close_tm[:, j], period=period, ma_type=ma_type)
            bull_cpu[:, j] = bull
            bear_cpu[:, j] = bear


        bull_dev, bear_dev = ti.eri_cuda_many_series_one_param_dev(
            high_tm.astype(np.float32).ravel(),
            low_tm.astype(np.float32).ravel(),
            close_tm.astype(np.float32).ravel(),
            close_tm.shape[1],
            close_tm.shape[0],
            period,
            ma_type,
        )
        bull_gpu = cp.asnumpy(cp.asarray(bull_dev))
        bear_gpu = cp.asnumpy(cp.asarray(bear_dev))

        assert bull_gpu.shape == (T, N)
        assert bear_gpu.shape == (T, N)
        assert_close(bull_gpu, bull_cpu, rtol=1e-4, atol=1e-5, msg="ERI bull: CUDA many-series vs CPU mismatch")
        assert_close(bear_gpu, bear_cpu, rtol=1e-4, atol=1e-5, msg="ERI bear: CUDA many-series vs CPU mismatch")

