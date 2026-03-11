"""
Python binding tests for Garman-Klass volatility CUDA kernels.
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
    if not hasattr(ti, 'garman_klass_volatility_cuda_batch_dev'):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32)
        handle, _ = ti.garman_klass_volatility_cuda_batch_dev(
            x, x + 0.5, np.maximum(x - 0.4, 0.01), x + 0.1, lookback_range=(3, 3, 0)
        )
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestGarmanKlassVolatilityCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_cuda_batch_matches_cpu(self, test_data):
        open_ = test_data['open']
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        lookback = 20

        cpu = ti.garman_klass_volatility(open_, high, low, close, lookback=lookback)

        handle, meta = ti.garman_klass_volatility_cuda_batch_dev(
            open_.astype(np.float32),
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
            lookback_range=(lookback, lookback, 0),
        )

        gpu = cp.asarray(handle)
        gpu_first = cp.asnumpy(gpu)[0]

        assert tuple(meta["lookbacks"]) == (lookback,)
        assert_close(gpu_first, cpu, rtol=5e-3, atol=2e-5, msg="CUDA batch vs CPU mismatch")

    def test_cuda_dlpack_matches_cpu(self, test_data):
        open_ = test_data['open']
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        lookback = 20

        cpu = ti.garman_klass_volatility(open_, high, low, close, lookback=lookback)

        handle, _ = ti.garman_klass_volatility_cuda_batch_dev(
            open_.astype(np.float32),
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
            lookback_range=(lookback, lookback, 0),
        )

        gpu = cp.from_dlpack(handle)
        gpu_first = cp.asnumpy(gpu)[0]

        assert_close(
            gpu_first,
            cpu,
            rtol=5e-3,
            atol=2e-5,
            msg="Garman-Klass DLPack export mismatch vs CPU",
        )

    def test_cuda_dlpack_device_hint_errors(self, test_data):
        open_ = test_data['open']
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        lookback = 20

        handle, _ = ti.garman_klass_volatility_cuda_batch_dev(
            open_.astype(np.float32),
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
            lookback_range=(lookback, lookback, 0),
        )

        kdl, dev = handle.__dlpack_device__()
        wrong_dev = dev + 1

        with pytest.raises(ValueError, match="dl_device mismatch for __dlpack__"):
            handle.__dlpack__(dl_device=(kdl, wrong_dev), copy=False)

        with pytest.raises(ValueError, match="device copy not implemented for __dlpack__"):
            handle.__dlpack__(dl_device=(kdl, wrong_dev), copy=True)

    def test_cuda_many_series_one_param_matches_cpu(self, test_data):
        rows = 1024
        cols = 4
        lookback = 14

        base_open = test_data['open'][:rows].astype(np.float64)
        base_high = test_data['high'][:rows].astype(np.float64)
        base_low = test_data['low'][:rows].astype(np.float64)
        base_close = test_data['close'][:rows].astype(np.float64)

        open_tm = np.zeros((rows, cols), dtype=np.float64)
        high_tm = np.zeros((rows, cols), dtype=np.float64)
        low_tm = np.zeros((rows, cols), dtype=np.float64)
        close_tm = np.zeros((rows, cols), dtype=np.float64)
        for j in range(cols):
            scale = 1.0 + 0.01 * j
            open_tm[:, j] = base_open * scale
            high_tm[:, j] = base_high * scale
            low_tm[:, j] = np.maximum(base_low * scale, 0.01)
            close_tm[:, j] = base_close * scale

        cpu_tm = np.zeros_like(open_tm)
        for j in range(cols):
            cpu_tm[:, j] = ti.garman_klass_volatility(
                np.ascontiguousarray(open_tm[:, j]),
                np.ascontiguousarray(high_tm[:, j]),
                np.ascontiguousarray(low_tm[:, j]),
                np.ascontiguousarray(close_tm[:, j]),
                lookback=lookback
            )

        handle = ti.garman_klass_volatility_cuda_many_series_one_param_dev(
            open_tm.astype(np.float32).reshape(-1),
            high_tm.astype(np.float32).reshape(-1),
            low_tm.astype(np.float32).reshape(-1),
            close_tm.astype(np.float32).reshape(-1),
            cols=cols,
            rows=rows,
            lookback=lookback,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == (rows, cols)
        assert_close(gpu_tm, cpu_tm, rtol=5e-3, atol=2e-5, msg="CUDA many-series vs CPU mismatch")
