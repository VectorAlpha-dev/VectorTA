"""Python binding tests for highpass_2_pole CUDA kernels."""
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
    if not hasattr(ti, "highpass_2_pole_cuda_batch_dev"):
        return False
    sample = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    try:
        handle = ti.highpass_2_pole_cuda_batch_dev(
            sample,
            period_range=(6, 6, 0),
            k_range=(0.7, 0.7, 0.0),
        )
        _ = cp.asarray(handle)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(
    not _cuda_available(), reason="CUDA not available or CUDA bindings not built"
)
class TestHighpass2Cuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_highpass2_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"][:1024]
        params = dict(period=48, k=0.707)

        cpu = ti.highpass_2_pole(close, **params)

        handle = ti.highpass_2_pole_cuda_batch_dev(
            close.astype(np.float32),
            period_range=(params["period"], params["period"], 0),
            k_range=(params["k"], params["k"], 0.0),
        )
        gpu = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(
            gpu,
            cpu,
            rtol=1e-5,
            atol=1e-5,
            msg="highpass_2_pole CUDA batch vs CPU mismatch",
        )

    def test_highpass2_cuda_batch_sweep_matches_cpu(self, test_data):
        close = test_data["close"][:1536]
        sweep = dict(period_range=(6, 64, 7), k_range=(0.2, 0.9, 0.15))

        def _axis_usize(rng):
            start, end, step = rng
            if step == 0 or start == end:
                return [start]
            return list(range(start, end + 1, step))

        def _axis_f64(rng):
            start, end, step = rng
            if abs(step) < 1e-12 or abs(start - end) < 1e-12:
                return [start]
            values = []
            current = start
            while current <= end + 1e-12:
                values.append(current)
                current += step
            return values

        periods = _axis_usize(sweep["period_range"])
        ks = _axis_f64(sweep["k_range"])

        cpu_rows = []
        for period in periods:
            for kval in ks:
                cpu_rows.append(ti.highpass_2_pole(close, period=period, k=kval))
        cpu = np.vstack(cpu_rows)

        handle = ti.highpass_2_pole_cuda_batch_dev(
            close.astype(np.float32),
            period_range=sweep["period_range"],
            k_range=sweep["k_range"],
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu.shape
        assert_close(
            gpu,
            cpu,
            rtol=1e-5,
            atol=1e-5,
            msg="highpass_2_pole CUDA sweep mismatch",
        )

    def test_highpass2_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        params = dict(period=48, k=0.707)
        base_series = test_data["close"][:T]
        data_tm = np.full((T, N), np.nan, dtype=np.float64)
        for j in range(N):
            for t in range(j * 2, T):
                x = base_series[t] if np.isfinite(base_series[t]) else float(t)
                data_tm[t, j] = np.cos(0.0023 * x + 0.01 * j) + 0.00041 * t

        cpu_tm = np.full_like(data_tm, np.nan)
        for j in range(N):
            cpu_tm[:, j] = ti.highpass_2_pole(
                data_tm[:, j],
                period=params["period"],
                k=params["k"],
            )

        handle = ti.highpass_2_pole_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32),
            period=params["period"],
            k=params["k"],
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == cpu_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=1e-5,
            atol=1e-5,
            msg="highpass_2_pole CUDA many-series mismatch",
        )
