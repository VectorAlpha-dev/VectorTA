"""Python CUDA binding tests for Keltner Channels."""
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
    if not hasattr(ti, "keltner_cuda_batch_dev"):
        return False
    try:
        data = load_test_data()
        high = data["high"][:128].astype(np.float32)
        low = data["low"][:128].astype(np.float32)
        close = data["close"][:128].astype(np.float32)
        out = ti.keltner_cuda_batch_dev(high, low, close, close, (14, 14, 0), (2.0, 2.0, 0.0))
        _ = cp.asarray(out["upper"])  
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "device" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestKeltnerCuda:
    @pytest.fixture(scope="class")
    def ds(self):
        return load_test_data()

    def test_keltner_cuda_batch_matches_cpu(self, ds):
        high = ds["high"].astype(np.float64)
        low = ds["low"].astype(np.float64)
        close = ds["close"].astype(np.float64)
        src = close

        period = 20
        mult = 2.0

        up_cpu, mid_cpu, low_cpu = ti.keltner(high, low, close, src, period, mult, "ema")

        gpu = ti.keltner_cuda_batch_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
            src.astype(np.float32),
            (period, period, 0),
            (mult, mult, 0.0),
            "ema",
        )

        up_gpu = cp.asnumpy(cp.asarray(gpu["upper"]))[0]
        mid_gpu = cp.asnumpy(cp.asarray(gpu["middle"]))[0]
        low_gpu = cp.asnumpy(cp.asarray(gpu["lower"]))[0]

        assert_close(mid_gpu, mid_cpu, rtol=1e-4, atol=2e-3, msg="middle mismatch")
        assert_close(up_gpu, up_cpu, rtol=1e-4, atol=2e-3, msg="upper mismatch")
        assert_close(low_gpu, low_cpu, rtol=1e-4, atol=2e-3, msg="lower mismatch")

    def test_keltner_cuda_many_series_one_param_matches_cpu(self, ds):
        rows = 2048
        cols = 4
        close = ds["close"][:rows].astype(np.float64)
        series = np.vstack([close * (1.0 + 0.01 * j) for j in range(cols)]).T

        
        high = series + 0.12 + 0.004 * np.sin(np.arange(rows) * 0.002)[:, None]
        low = series - 0.12 - 0.004 * np.sin(np.arange(rows) * 0.002)[:, None]

        period = 20
        mult = 2.0

        up_cpu = np.zeros_like(series)
        mid_cpu = np.zeros_like(series)
        low_cpu = np.zeros_like(series)
        for j in range(cols):
            u, m, l = ti.keltner(high[:, j], low[:, j], series[:, j], series[:, j], period, mult, "ema")
            up_cpu[:, j] = u
            mid_cpu[:, j] = m
            low_cpu[:, j] = l

        up_dev, mid_dev, low_dev = ti.keltner_cuda_many_series_one_param_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            series.astype(np.float32),
            series.astype(np.float32),
            cols,
            rows,
            period,
            float(mult),
            "ema",
        )

        up_gpu = cp.asnumpy(cp.asarray(up_dev))
        mid_gpu = cp.asnumpy(cp.asarray(mid_dev))
        low_gpu = cp.asnumpy(cp.asarray(low_dev))

        assert_close(mid_gpu, mid_cpu, rtol=1e-4, atol=2e-3, msg="middle TM mismatch")
        assert_close(up_gpu, up_cpu, rtol=1e-4, atol=2e-3, msg="upper TM mismatch")
        assert_close(low_gpu, low_cpu, rtol=1e-4, atol=2e-3, msg="lower TM mismatch")

