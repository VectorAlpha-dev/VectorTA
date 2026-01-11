"""
Python CUDA binding tests for FVG Trailing Stop.
Skips gracefully when CUDA is unavailable or bindings not built.
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
    if not hasattr(ti, 'fvg_trailing_stop_cuda_batch_dev'):
        return False
    try:
        
        x = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32)
        h = x + 0.1
        l = x - 0.1
        u,lw,ut,lt = ti.fvg_trailing_stop_cuda_batch_dev(
            h, l, x,
            lookback_range=(5,5,0),
            smoothing_range=(9,9,0),
            reset_toggle=(True, False),
        )
        _ = cp.asarray(u)
        _ = cp.asarray(lw)
        _ = cp.asarray(ut)
        _ = cp.asarray(lt)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestFvgTrailingStopCuda:
    @pytest.fixture(scope='class')
    def test_ohlc(self):
        data = load_test_data()
        close = data['close']
        
        high = close + 0.15
        low  = close - 0.14
        return high, low, close

    def test_batch_dev_matches_cpu(self, test_ohlc):
        high, low, close = test_ohlc
        params = dict(unmitigated_fvg_lookback=5, smoothing_length=9, reset_on_cross=False)
        
        cpu_u, cpu_l, cpu_ut, cpu_lt = ti.fvg_trailing_stop(close, **params, kernel=None)
        
        u, l, ut, lt = ti.fvg_trailing_stop_cuda_batch_dev(
            high.astype(np.float32), low.astype(np.float32), close.astype(np.float32),
            lookback_range=(params['unmitigated_fvg_lookback'], params['unmitigated_fvg_lookback'], 0),
            smoothing_range=(params['smoothing_length'], params['smoothing_length'], 0),
            reset_toggle=(params['reset_on_cross'], params['reset_on_cross']),
        )
        gu = cp.asnumpy(cp.asarray(u))[0]
        gl = cp.asnumpy(cp.asarray(l))[0]
        gut= cp.asnumpy(cp.asarray(ut))[0]
        glt= cp.asnumpy(cp.asarray(lt))[0]
        
        assert_close(gu, cpu_u, rtol=1e-3, atol=1e-3, msg="upper mismatch")
        assert_close(gl, cpu_l, rtol=1e-3, atol=1e-3, msg="lower mismatch")
        assert_close(gut, cpu_ut, rtol=1e-3, atol=1e-3, msg="upper_ts mismatch")
        assert_close(glt, cpu_lt, rtol=1e-3, atol=1e-3, msg="lower_ts mismatch")

    def test_many_series_one_param_matches_cpu(self, test_ohlc):
        high, low, close = test_ohlc
        T = 2048
        N = 4
        
        h_tm = np.zeros((T, N), dtype=np.float64)
        l_tm = np.zeros_like(h_tm)
        c_tm = np.zeros_like(h_tm)
        for j in range(N):
            h_tm[:, j] = high[:T] * (1.0 + 0.0*j)
            l_tm[:, j] = low [:T] * (1.0 + 0.0*j)
            c_tm[:, j] = close[:T] * (1.0 + 0.0*j)

        params = dict(unmitigated_fvg_lookback=5, smoothing_length=9, reset_on_cross=False)
        
        cpu_u = np.zeros_like(c_tm)
        cpu_l = np.zeros_like(c_tm)
        cpu_ut= np.zeros_like(c_tm)
        cpu_lt= np.zeros_like(c_tm)
        for j in range(N):
            u, l, ut, lt = ti.fvg_trailing_stop(c_tm[:, j], **params, kernel=None)
            cpu_u[:, j] = u; cpu_l[:, j] = l; cpu_ut[:, j] = ut; cpu_lt[:, j] = lt

        u, l, ut, lt = ti.fvg_trailing_stop_cuda_many_series_one_param_dev(
            h_tm.astype(np.float32), l_tm.astype(np.float32), c_tm.astype(np.float32),
            cols=N, rows=T,
            unmitigated_fvg_lookback=params['unmitigated_fvg_lookback'],
            smoothing_length=params['smoothing_length'],
            reset_on_cross=params['reset_on_cross'],
        )
        gu = cp.asnumpy(cp.asarray(u))
        gl = cp.asnumpy(cp.asarray(l))
        gut= cp.asnumpy(cp.asarray(ut))
        glt= cp.asnumpy(cp.asarray(lt))
        assert_close(gu, cpu_u, rtol=1e-3, atol=1e-3, msg="upper mismatch")
        assert_close(gl, cpu_l, rtol=1e-3, atol=1e-3, msg="lower mismatch")
        assert_close(gut, cpu_ut, rtol=1e-3, atol=1e-3, msg="upper_ts mismatch")
        assert_close(glt, cpu_lt, rtol=1e-3, atol=1e-3, msg="lower_ts mismatch")

