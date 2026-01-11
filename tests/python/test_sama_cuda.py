"""Python binding tests for SAMA CUDA kernels."""
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
    if not hasattr(ti, "sama_cuda_batch_dev"):
        return False
    sample = np.array([np.nan, 1.0, 2.5, 3.0, 4.0], dtype=np.float32)
    try:
        handle = ti.sama_cuda_batch_dev(
            sample,
            length_range=(32, 32, 0),
            maj_length_range=(10, 10, 0),
            min_length_range=(4, 4, 0),
        )
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or CUDA bindings not built")
class TestSamaCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_sama_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"][:1024]
        params = dict(length=64, maj_length=18, min_length=8)

        cpu = ti.sama(close, **params)

        handle = ti.sama_cuda_batch_dev(
            close.astype(np.float32),
            length_range=(params["length"], params["length"], 0),
            maj_length_range=(params["maj_length"], params["maj_length"], 0),
            min_length_range=(params["min_length"], params["min_length"], 0),
        )
        gpu = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(
            gpu,
            cpu,
            rtol=1e-5,
            atol=1e-5,
            msg="SAMA CUDA batch vs CPU mismatch",
        )

    def test_sama_cuda_batch_sweep_matches_cpu(self, test_data):
        close = test_data["close"][:1536]
        sweep = dict(length_range=(32, 96, 16), maj_length_range=(10, 22, 6), min_length_range=(4, 12, 4))

        def _axis(rng):
            start, end, step = rng
            if step == 0 or start == end:
                return [start]
            return list(range(start, end + 1, step))

        lengths = _axis(sweep["length_range"])
        maj_lengths = _axis(sweep["maj_length_range"])
        min_lengths = _axis(sweep["min_length_range"])

        cpu_rows = []
        for length in lengths:
            for maj in maj_lengths:
                for mn in min_lengths:
                    cpu_rows.append(
                        ti.sama(close, length=length, maj_length=maj, min_length=mn)
                    )
        cpu = np.vstack(cpu_rows)

        handle = ti.sama_cuda_batch_dev(
            close.astype(np.float32),
            length_range=sweep["length_range"],
            maj_length_range=sweep["maj_length_range"],
            min_length_range=sweep["min_length_range"],
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu.shape
        assert_close(
            gpu,
            cpu,
            rtol=1e-5,
            atol=1e-5,
            msg="SAMA CUDA sweep mismatch",
        )

    def test_sama_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        params = dict(length=64, maj_length=18, min_length=8)
        base_series = test_data["close"][:T]
        data_tm = np.full((T, N), np.nan, dtype=np.float64)
        for j in range(N):
            for t in range(j, T):
                x = base_series[t] if np.isfinite(base_series[t]) else 0.0
                data_tm[t, j] = np.sin(0.0021 * x + 0.01 * j) + 0.00029 * t

        cpu_tm = np.full_like(data_tm, np.nan)
        for j in range(N):
            cpu_tm[:, j] = ti.sama(
                data_tm[:, j],
                length=params["length"],
                maj_length=params["maj_length"],
                min_length=params["min_length"],
            )

        handle = ti.sama_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32),
            length=params["length"],
            maj_length=params["maj_length"],
            min_length=params["min_length"],
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=1e-5,
            atol=1e-5,
            msg="SAMA CUDA many-series mismatch",
        )
