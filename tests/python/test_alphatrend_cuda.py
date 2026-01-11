"""
Python binding tests for AlphaTrend CUDA kernels.
Skips gracefully when CUDA is unavailable or module is not built.
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
    if not hasattr(ti, 'alphatrend_cuda_batch_dev'):
        return False
    
    try:
        data = load_test_data()
        h = data['high'][:64].astype(np.float32)
        l = data['low'][:64].astype(np.float32)
        c = data['close'][:64].astype(np.float32)
        v = data['volume'][:64].astype(np.float32)
        handle = ti.alphatrend_cuda_batch_dev(
            h, l, c, v,
            coeff_range=(1.0, 1.0, 0.0),
            period_range=(14, 14, 0),
            no_volume=True,
        )
        _ = cp.asarray(handle['k1'])
        _ = cp.asarray(handle['k2'])
        return True
    except Exception as e:  
        msg = str(e).lower()
        return not (('cuda not available' in msg) or ('ptx' in msg) or ('nvcc' in msg))


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestAlphaTrendCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_alphatrend_cuda_batch_matches_cpu(self, test_data):
        h = test_data['high'].astype(np.float64)
        l = test_data['low'].astype(np.float64)
        c = test_data['close'].astype(np.float64)
        v = test_data['volume'].astype(np.float64)
        coeff = 1.0
        period = 14
        no_volume = True

        
        k1_cpu, k2_cpu = ti.alphatrend(h, l, l, c, v, coeff, period, no_volume)  

        
        out = ti.alphatrend_cuda_batch_dev(
            h.astype(np.float32),
            l.astype(np.float32),
            c.astype(np.float32),
            v.astype(np.float32),
            coeff_range=(coeff, coeff, 0.0),
            period_range=(period, period, 0),
            no_volume=no_volume,
        )
        k1_gpu = cp.asnumpy(cp.asarray(out['k1']))[0]
        k2_gpu = cp.asnumpy(cp.asarray(out['k2']))[0]

        assert_close(k1_gpu, k1_cpu, rtol=2e-3, atol=2e-5, msg="AlphaTrend k1 CUDA vs CPU mismatch")
        assert_close(k2_gpu, k2_cpu, rtol=2e-3, atol=2e-5, msg="AlphaTrend k2 CUDA vs CPU mismatch")

    def test_alphatrend_cuda_many_series_one_param_matches_cpu(self, test_data):
        
        T = 1024
        N = 3
        h = np.zeros((T, N), dtype=np.float64)
        l = np.zeros((T, N), dtype=np.float64)
        c = np.zeros((T, N), dtype=np.float64)
        v = np.zeros((T, N), dtype=np.float64)
        base_h = test_data['high'][:T]
        base_l = test_data['low'][:T]
        base_c = test_data['close'][:T]
        base_v = test_data['volume'][:T]
        for j in range(N):
            h[:, j] = base_h * (1.0 + 0.01 * j)
            l[:, j] = base_l * (1.0 + 0.01 * j)
            c[:, j] = base_c * (1.0 + 0.01 * j)
            v[:, j] = base_v

        coeff = 1.0
        period = 14
        no_volume = True

        
        k1_cpu = np.zeros_like(c)
        k2_cpu = np.zeros_like(c)
        for j in range(N):
            k1_j, k2_j = ti.alphatrend(h[:, j], h[:, j], l[:, j], c[:, j], v[:, j], coeff, period, no_volume)
            k1_cpu[:, j] = k1_j
            k2_cpu[:, j] = k2_j

        
        k1_h, k2_h = ti.alphatrend_cuda_many_series_one_param_dev(
            h.astype(np.float32).reshape(-1),
            l.astype(np.float32).reshape(-1),
            c.astype(np.float32).reshape(-1),
            v.astype(np.float32).reshape(-1),
            cols=N,
            rows=T,
            coeff=coeff,
            period=period,
            no_volume=no_volume,
        )
        k1_gpu = cp.asnumpy(cp.asarray(k1_h)).reshape(T, N)
        k2_gpu = cp.asnumpy(cp.asarray(k2_h)).reshape(T, N)

        assert_close(k1_gpu, k1_cpu, rtol=2e-3, atol=2e-5, msg="AlphaTrend many-series k1 mismatch")
        assert_close(k2_gpu, k2_cpu, rtol=2e-3, atol=2e-5, msg="AlphaTrend many-series k2 mismatch")

