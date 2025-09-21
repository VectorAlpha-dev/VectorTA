"""
Python binding tests for WaveTrend CUDA kernels.
Validates GPU batch output against the scalar CPU implementation and ensures
CuPy interop works through __cuda_array_interface__.
"""
import numpy as np
import pytest

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency for CUDA path
    cp = None

try:
    import my_project as ti
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python,cuda' first",
        allow_module_level=True,
    )

from test_utils import assert_close, load_test_data


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "wavetrend_cuda_batch_dev"):
        return False
    try:
        probe = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle = ti.wavetrend_cuda_batch_dev(
            probe,
            channel_length_range=(6, 6, 0),
            average_length_range=(9, 9, 0),
            ma_length_range=(3, 3, 0),
            factor_range=(0.015, 0.015, 0.0),
        )
        _ = cp.asarray(handle["wt1"])
        return True
    except Exception as exc:  # pragma: no cover - defensive path
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestWavetrendCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_wavetrend_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"].astype(np.float64)
        channel = (6, 12, 3)
        average = (9, 15, 3)
        ma = (3, 5, 1)
        factor = (0.010, 0.018, 0.004)

        cpu = ti.wavetrend_batch(close, channel, average, ma, factor)
        cpu_wt1 = cpu["wt1"]
        cpu_wt2 = cpu["wt2"]
        cpu_diff = cpu["wt_diff"]

        handle = ti.wavetrend_cuda_batch_dev(
            close.astype(np.float32), channel, average, ma, factor
        )
        gpu_wt1 = cp.asnumpy(cp.asarray(handle["wt1"]))
        gpu_wt2 = cp.asnumpy(cp.asarray(handle["wt2"]))
        gpu_diff = cp.asnumpy(cp.asarray(handle["wt_diff"]))

        assert gpu_wt1.shape == cpu_wt1.shape
        assert gpu_wt2.shape == cpu_wt2.shape
        assert gpu_diff.shape == cpu_diff.shape

        mask = ~np.isnan(cpu_wt1)
        assert_close(
            gpu_wt1[mask],
            cpu_wt1[mask],
            rtol=5e-3,
            atol=5e-4,
            msg="WT1 GPU vs CPU mismatch",
        )
        mask = ~np.isnan(cpu_wt2)
        assert_close(
            gpu_wt2[mask],
            cpu_wt2[mask],
            rtol=5e-3,
            atol=5e-4,
            msg="WT2 GPU vs CPU mismatch",
        )
        mask = ~np.isnan(cpu_diff)
        assert_close(
            gpu_diff[mask],
            cpu_diff[mask],
            rtol=5e-3,
            atol=5e-4,
            msg="WT diff GPU vs CPU mismatch",
        )

    def test_wavetrend_cuda_metadata(self, test_data):
        close = test_data["close"].astype(np.float64)
        channel = (5, 7, 1)
        average = (9, 11, 1)
        ma = (2, 3, 1)
        factor = (0.012, 0.014, 0.001)

        handle = ti.wavetrend_cuda_batch_dev(
            close.astype(np.float32), channel, average, ma, factor
        )
        channels = np.array(handle["channel_lengths"])  # host arrays
        averages = np.array(handle["average_lengths"])
        mas = np.array(handle["ma_lengths"])
        factors = np.array(handle["factors"])

        expected_channels = np.arange(channel[0], channel[1] + 1, channel[2] or 1)
        expected_averages = np.arange(average[0], average[1] + 1, average[2] or 1)
        expected_mas = np.arange(ma[0], ma[1] + 1, ma[2] or 1)
        expected_factors = np.arange(factor[0], factor[1] + 1e-9, factor[2] or 1.0)

        assert set(channels.tolist()) == set(expected_channels.tolist())
        assert set(averages.tolist()) == set(expected_averages.tolist())
        assert set(mas.tolist()) == set(expected_mas.tolist())
        assert np.allclose(factors, expected_factors, atol=1e-9)
