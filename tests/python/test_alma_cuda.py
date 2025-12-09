"""
Python binding tests for ALMA CUDA kernels.
These mirror the CPU tests and use the same reference values where applicable.
Skips gracefully when CUDA is unavailable or CUDA feature not built.
"""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency for CUDA path
    cp = None

try:
    import my_project as ti
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python,cuda' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, 'alma_cuda_batch_dev'):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float64)
        handle = ti.alma_cuda_batch_dev(
            x.astype(np.float32),
            period_range=(3, 3, 0),
            offset_range=(0.85, 0.85, 0.0),
            sigma_range=(6.0, 6.0, 0.0),
        )
        _ = cp.asarray(handle)  # ensure CuPy can wrap the handle
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        # Other errors mean CUDA path exists; consider available
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestAlmaCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_alma_cuda_batch_matches_cpu(self, test_data):
        close = test_data['close']
        period, offset, sigma = 9, 0.85, 6.0

        # CPU baseline
        cpu = ti.alma(close, period, offset, sigma)

        # CUDA single-combo batch
        handle = ti.alma_cuda_batch_dev(
            close.astype(np.float32),
            period_range=(period, period, 0),
            offset_range=(offset, offset, 0.0),
            sigma_range=(sigma, sigma, 0.0),
        )

        gpu = cp.asarray(handle)
        gpu_first = cp.asnumpy(gpu)[0]

        # Compare entire row with modest tolerance (fp32 vs fp64)
        assert_close(gpu_first, cpu, rtol=1e-5, atol=1e-6, msg="CUDA batch vs CPU mismatch")

    def test_alma_cuda_dlpack_matches_cpu(self, test_data):
        """Ensure DLPack export for ALMA CUDA handle is correct."""
        close = test_data['close']
        period, offset, sigma = 9, 0.85, 6.0

        # CPU baseline (fp64)
        cpu = ti.alma(close, period, offset, sigma)

        # CUDA handle
        handle = ti.alma_cuda_batch_dev(
            close.astype(np.float32),
            period_range=(period, period, 0),
            offset_range=(offset, offset, 0.0),
            sigma_range=(sigma, sigma, 0.0),
        )

        # Consume via DLPack path explicitly
        gpu = cp.fromDlpack(handle)
        gpu_first = cp.asnumpy(gpu)[0]

        assert_close(
            gpu_first,
            cpu,
            rtol=1e-5,
            atol=1e-6,
            msg="ALMA DLPack export mismatch vs CPU",
        )

    def test_alma_cuda_dlpack_device_hint_errors(self, test_data):
        """Shared DLPack helper: validate dl_device/copy handling."""
        close = test_data['close']
        period, offset, sigma = 9, 0.85, 6.0

        handle = ti.alma_cuda_batch_dev(
            close.astype(np.float32),
            period_range=(period, period, 0),
            offset_range=(offset, offset, 0.0),
            sigma_range=(sigma, sigma, 0.0),
        )

        kdl, dev = handle.__dlpack_device__()
        wrong_dev = dev + 1

        with pytest.raises(ValueError, match="dl_device mismatch for __dlpack__"):
            handle.__dlpack__(dl_device=(kdl, wrong_dev), copy=False)

        with pytest.raises(ValueError, match="device copy not implemented for __dlpack__"):
            handle.__dlpack__(dl_device=(kdl, wrong_dev), copy=True)

    # multi-stream variant removed

    def test_alma_cuda_many_series_one_param_matches_cpu(self, test_data):
        # Build small time-major matrix (T,N) with varied columns
        T = 1024
        N = 4
        series = test_data['close'][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = series * (1.0 + 0.01 * j)

        period = 14
        offset = 0.85
        sigma = 6.0

        # CPU baseline per series
        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.alma(data_tm[:, j], period, offset, sigma)

        # CUDA
        handle = ti.alma_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32), period, offset, sigma
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-5, atol=1e-6, msg="CUDA many-series vs CPU mismatch")
