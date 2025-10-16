"""CUDA bindings tests for Stochastic Oscillator (Stoch)."""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover
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
    if not hasattr(ti, "stoch_cuda_batch_dev"):
        return False
    try:
        close = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        high = close + 0.1
        low = close - 0.1
        k, d = ti.stoch_cuda_batch_dev(
            high, low, close, fastk_period=(3, 3, 0), slowk_period=(1, 1, 0), slowd_period=(1, 1, 0)
        )
        _ = cp.asarray(k)
        _ = cp.asarray(d)
        return True
    except Exception as exc:  # pragma: no cover - probing path
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "driver" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings missing")
class TestStochCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def _synth_hlc(self, close: np.ndarray):
        off = 0.15 + 0.01 * np.sin(np.arange(close.size) * 0.01)
        return close + off, close - off

    def test_stoch_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"].astype(np.float64)
        close32 = close.astype(np.float32)
        high32, low32 = self._synth_hlc(close32)

        fastk = 14
        slowk = 3
        slowd = 3
        cpu = ti.stoch(
            high32.astype(np.float64),
            low32.astype(np.float64),
            close,
            fastk,
            slowk,
            "sma",
            slowd,
            "ema",
        )

        k_dev, d_dev = ti.stoch_cuda_batch_dev(
            high32,
            low32,
            close32,
            fastk_period=(fastk, fastk, 0),
            slowk_period=(slowk, slowk, 0),
            slowd_period=(slowd, slowd, 0),
            slowk_ma_type="sma",
            slowd_ma_type="ema",
        )
        k = cp.asnumpy(cp.asarray(k_dev))[0]
        d = cp.asnumpy(cp.asarray(d_dev))[0]

        assert_close(k, cpu["k"], rtol=1e-3, atol=1e-2, msg="Stoch K mismatch")
        assert_close(d, cpu["d"], rtol=1e-3, atol=1e-2, msg="Stoch D mismatch")

    def test_stoch_cuda_many_series_time_major_matches_cpu(self, test_data):
        T = 2048
        N = 4
        base = test_data["close"][:T].astype(np.float64)
        close_tm = np.zeros((T, N), dtype=np.float32)
        for j in range(N):
            close_tm[:, j] = (base * (1.0 + 0.02 * j)).astype(np.float32)
        high_tm, low_tm = self._synth_hlc(close_tm)

        fastk, slowk, slowd = 14, 3, 3

        # CPU reference
        ref_k = np.zeros_like(close_tm, dtype=np.float64)
        ref_d = np.zeros_like(close_tm, dtype=np.float64)
        for j in range(N):
            out = ti.stoch(
                high_tm[:, j].astype(np.float64),
                low_tm[:, j].astype(np.float64),
                close_tm[:, j].astype(np.float64),
                fastk,
                slowk,
                "sma",
                slowd,
                "sma",
            )
            ref_k[:, j] = out["k"]
            ref_d[:, j] = out["d"]

        k_dev, d_dev = ti.stoch_cuda_many_series_one_param_dev(
            high_tm, low_tm, close_tm, close_tm.shape[1], close_tm.shape[0], fastk, slowk, slowd
        )
        k = cp.asnumpy(cp.asarray(k_dev))
        d = cp.asnumpy(cp.asarray(d_dev))

        assert_close(k, ref_k, rtol=1e-3, atol=1e-2, msg="Stoch K TM mismatch")
        assert_close(d, ref_d, rtol=1e-3, atol=1e-2, msg="Stoch D TM mismatch")

