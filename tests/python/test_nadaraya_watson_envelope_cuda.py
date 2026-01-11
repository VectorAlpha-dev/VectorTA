"""Python binding tests for Nadaraya–Watson Envelope CUDA kernels."""
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
    if not hasattr(ti, "nadaraya_watson_envelope_cuda_batch_dev"):
        return False
    probe = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    try:
        handle = ti.nadaraya_watson_envelope_cuda_batch_dev(
            probe,
            bandwidth_range=(8.0, 8.0, 0.0),
            multiplier_range=(3.0, 3.0, 0.0),
            lookback_range=(5, 5, 0),
        )
        _ = cp.asarray(handle["upper"])  
        _ = cp.asarray(handle["lower"])  
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "device" in msg:
            return False
        return True


@pytest.mark.skipif(
    not _cuda_available(), reason="CUDA not available or CUDA bindings not built"
)
class TestNweCuda:
    @pytest.fixture(scope="class")
    def close(self):
        
        d = load_test_data()["close"].astype(np.float64)
        return d[:2048]

    def test_nwe_cuda_batch_matches_cpu(self, close):
        bw = 8.0
        mult = 3.0
        lb = 200

        upper_cpu, lower_cpu = ti.nadaraya_watson_envelope(close, bandwidth=bw, multiplier=mult, lookback=lb)
        gpu = ti.nadaraya_watson_envelope_cuda_batch_dev(
            close.astype(np.float32),
            bandwidth_range=(bw, bw, 0.0),
            multiplier_range=(mult, mult, 0.0),
            lookback_range=(lb, lb, 0),
        )
        up_gpu = cp.asnumpy(cp.asarray(gpu["upper"]))[0]
        lo_gpu = cp.asnumpy(cp.asarray(gpu["lower"]))[0]

        
        assert_close(up_gpu, upper_cpu, rtol=1e-3, atol=2e-3, msg="NWE upper CUDA vs CPU mismatch")
        assert_close(lo_gpu, lower_cpu, rtol=1e-3, atol=2e-3, msg="NWE lower CUDA vs CPU mismatch")

    def test_nwe_cuda_batch_sweep_shapes(self, close):
        sweep = dict(
            bandwidth_range=(6.0, 10.0, 2.0),
            multiplier_range=(2.0, 3.0, 0.5),
            lookback_range=(128, 256, 64),
        )
        gpu = ti.nadaraya_watson_envelope_cuda_batch_dev(
            close.astype(np.float32),
            **sweep,
        )
        up = cp.asnumpy(cp.asarray(gpu["upper"]))
        lo = cp.asnumpy(cp.asarray(gpu["lower"]))
        assert up.shape == lo.shape
        assert up.shape[1] == close.shape[0]

    def test_nwe_cuda_many_series_one_param_matches_cpu(self, close):
        T = 1536
        N = 5
        bw = 8.0
        mult = 3.0
        lb = 180
        base = close[:T]
        data_tm = np.full((T, N), np.nan, dtype=np.float64)
        for j in range(N):
            for t in range(j + 12, T):
                x = base[t] if np.isfinite(base[t]) else 0.0
                data_tm[t, j] = np.sin(0.0019 * x + 0.013 * j) + 0.00031 * t

        upper_cpu = np.full_like(data_tm, np.nan)
        lower_cpu = np.full_like(data_tm, np.nan)
        for j in range(N):
            up_j, lo_j = ti.nadaraya_watson_envelope(data_tm[:, j], bandwidth=bw, multiplier=mult, lookback=lb)
            upper_cpu[:, j] = up_j
            lower_cpu[:, j] = lo_j

        gpu = ti.nadaraya_watson_envelope_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32),
            bandwidth=bw,
            multiplier=mult,
            lookback=lb,
        )
        up_gpu = cp.asnumpy(cp.asarray(gpu["upper"]))
        lo_gpu = cp.asnumpy(cp.asarray(gpu["lower"]))
        assert up_gpu.shape == data_tm.shape
        assert lo_gpu.shape == data_tm.shape
        assert_close(up_gpu, upper_cpu, rtol=1e-3, atol=2e-3, msg="NWE upper many-series mismatch")
        assert_close(lo_gpu, lower_cpu, rtol=1e-3, atol=2e-3, msg="NWE lower many-series mismatch")

