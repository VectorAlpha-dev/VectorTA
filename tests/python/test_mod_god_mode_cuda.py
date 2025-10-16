"""
Python binding tests for MOD_GOD_MODE CUDA kernels.
Skips gracefully when CUDA is unavailable or bindings are not present.
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
    if not hasattr(ti, 'mod_god_mode_cuda_batch_dev'):
        return False
    try:
        data = load_test_data()
        out = ti.mod_god_mode_cuda_batch_dev(
            data['high'].astype(np.float32),
            data['low'].astype(np.float32),
            data['close'].astype(np.float32),
            (17, 17, 0), (6, 6, 0), (4, 4, 0),
            mode='tradition_mg',
            use_volume=False,
        )
        _ = cp.asarray(out['wavetrend'])  # ensure CuPy can wrap the handle
        return True
    except Exception as e:
        msg = str(e).lower()
        return 'cuda not available' not in msg


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestModGodModeCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_mgm_cuda_batch_matches_cpu(self, test_data):
        h = test_data['high']
        l = test_data['low']
        c = test_data['close']
        n1, n2, n3 = 17, 6, 4
        mode = 'tradition_mg'

        # CPU baseline
        wt_cpu, sig_cpu, hist_cpu = ti.mod_god_mode(h, l, c, None, n1, n2, n3, mode, False)

        # CUDA single-combo batch
        out = ti.mod_god_mode_cuda_batch_dev(
            h.astype(np.float32), l.astype(np.float32), c.astype(np.float32),
            (n1, n1, 0), (n2, n2, 0), (n3, n3, 0), mode, False
        )
        wt = cp.asnumpy(cp.asarray(out['wavetrend']))[0]
        sig = cp.asnumpy(cp.asarray(out['signal']))[0]
        hist = cp.asnumpy(cp.asarray(out['histogram']))[0]

        # Allow fp32 tolerance
        assert_close(wt, wt_cpu, rtol=8e-3, atol=1e-3, msg='wt mismatch')
        assert_close(sig, sig_cpu, rtol=8e-3, atol=1e-3, msg='sig mismatch')
        assert_close(hist, hist_cpu, rtol=8e-3, atol=1e-3, msg='hist mismatch')

    def test_mgm_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 3
        h = test_data['high'][:T].astype(np.float64)
        l = test_data['low'][:T].astype(np.float64)
        c = test_data['close'][:T].astype(np.float64)
        h_tm = np.stack([h * (1 + 0.01*j) for j in range(N)], axis=1)
        l_tm = np.stack([l * (1 + 0.01*j) for j in range(N)], axis=1)
        c_tm = np.stack([c * (1 + 0.01*j) for j in range(N)], axis=1)
        n1, n2, n3 = 17, 6, 4
        mode = 'tradition_mg'

        wt_cpu = np.zeros_like(c_tm)
        sig_cpu = np.zeros_like(c_tm)
        hist_cpu = np.zeros_like(c_tm)
        for j in range(N):
            wt, sig, hist = ti.mod_god_mode(h_tm[:, j], l_tm[:, j], c_tm[:, j], None, n1, n2, n3, mode, False)
            wt_cpu[:, j] = wt
            sig_cpu[:, j] = sig
            hist_cpu[:, j] = hist

        out = ti.mod_god_mode_cuda_many_series_one_param_dev(
            h_tm.astype(np.float32).ravel(),
            l_tm.astype(np.float32).ravel(),
            c_tm.astype(np.float32).ravel(),
            N, T, n1, n2, n3, mode, False,
        )
        wt = cp.asnumpy(cp.asarray(out['wavetrend']))
        sig = cp.asnumpy(cp.asarray(out['signal']))
        hist = cp.asnumpy(cp.asarray(out['histogram']))

        assert wt.shape == (T, N)
        assert_close(wt, wt_cpu, rtol=8e-3, atol=1e-3, msg='wt TM mismatch')
        assert_close(sig, sig_cpu, rtol=8e-3, atol=1e-3, msg='sig TM mismatch')
        assert_close(hist, hist_cpu, rtol=8e-3, atol=1e-3, msg='hist TM mismatch')

