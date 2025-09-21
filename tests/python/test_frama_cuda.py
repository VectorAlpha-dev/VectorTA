"""CUDA bindings tests for the Fractal Adaptive Moving Average (FRAMA)."""

import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency
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
    if not hasattr(ti, "frama_cuda_batch_dev"):
        return False
    try:
        high = np.array(
            [np.nan, 1.2, 1.3, 1.4, 1.45, 1.5, 1.6, 1.7], dtype=np.float32
        )
        low = high - 0.4
        close = (high + low) * 0.5
        handle, _meta = ti.frama_cuda_batch_dev(
            high,
            low,
            close,
            window_range=(4, 4, 0),
            sc_range=(200, 200, 0),
            fc_range=(1, 1, 0),
        )
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  # pragma: no cover - probing path
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "driver" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings missing")
class TestFramaCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_frama_cuda_batch_matches_cpu(self, test_data):
        high = test_data["high"].astype(np.float64)
        low = test_data["low"].astype(np.float64)
        close = test_data["close"].astype(np.float64)

        window = 12
        sc = 260
        fc = 2

        cpu = ti.frama(high, low, close, window, sc, fc)

        handle, meta = ti.frama_cuda_batch_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
            window_range=(window, window, 0),
            sc_range=(sc, sc, 0),
            fc_range=(fc, fc, 0),
        )
        assert meta["windows"].tolist() == [window]
        assert meta["scs"].tolist() == [sc]
        assert meta["fcs"].tolist() == [fc]

        gpu = cp.asnumpy(cp.asarray(handle))[0]
        assert_close(gpu, cpu, rtol=5e-4, atol=5e-4, msg="CUDA FRAMA batch mismatch")

    def test_frama_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1500
        N = 3
        base_close = test_data["close"][:T].astype(np.float64)
        base_high = test_data["high"][:T].astype(np.float64)
        base_low = test_data["low"][:T].astype(np.float64)

        high_tm = np.zeros((T, N), dtype=np.float64)
        low_tm = np.zeros((T, N), dtype=np.float64)
        close_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            scale = 1.0 + 0.04 * j
            bias = 0.15 * j
            close_tm[:, j] = base_close * scale + bias
            high_tm[:, j] = base_high * scale + bias + 0.2
            low_tm[:, j] = base_low * scale + bias - 0.2

        window = 14
        sc = 220
        fc = 2

        cpu_tm = np.zeros_like(close_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.frama(
                high_tm[:, j],
                low_tm[:, j],
                close_tm[:, j],
                window,
                sc,
                fc,
            )

        handle = ti.frama_cuda_many_series_one_param_dev(
            high_tm.astype(np.float32),
            low_tm.astype(np.float32),
            close_tm.astype(np.float32),
            window,
            sc,
            fc,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == close_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=5e-4,
            atol=5e-4,
            msg="CUDA FRAMA many-series mismatch",
        )
