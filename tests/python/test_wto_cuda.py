"""Python binding tests for WTO CUDA kernels."""
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

from test_utils import assert_close, load_test_data


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "wto_cuda_batch_dev"):
        return False
    try:
        series = np.array([np.nan, 1.0, 2.0, 3.5, 4.0, 5.0], dtype=np.float32)
        handle = ti.wto_cuda_batch_dev(series, (4, 4, 0), (7, 7, 0))
        _ = cp.asarray(handle["wt1"])  
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "device" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestWtoCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        data = load_test_data()
        return data["close"].astype(np.float64)

    def test_wto_cuda_batch_matches_cpu(self, test_data):
        channel = 10
        average = 21

        wt1_cpu, wt2_cpu, hist_cpu = ti.wto(test_data, channel, average)

        gpu = ti.wto_cuda_batch_dev(
            test_data.astype(np.float32),
            (channel, channel, 0),
            (average, average, 0),
        )
        wt1_gpu = cp.asnumpy(cp.asarray(gpu["wt1"]))[0]
        wt2_gpu = cp.asnumpy(cp.asarray(gpu["wt2"]))[0]
        hist_gpu = cp.asnumpy(cp.asarray(gpu["hist"]))[0]

        assert_close(wt1_gpu, wt1_cpu, rtol=1e-4, atol=1e-5, msg="WT1 mismatch")
        assert_close(wt2_gpu, wt2_cpu, rtol=1e-4, atol=1e-5, msg="WT2 mismatch")
        assert_close(hist_gpu, hist_cpu, rtol=1e-4, atol=1e-5, msg="Histogram mismatch")

    def test_wto_cuda_batch_multi_combo_matches_cpu(self, test_data):
        channel_range = (8, 12, 2)
        average_range = (18, 24, 3)

        cpu = ti.wto_batch(test_data, channel_range, average_range)
        gpu = ti.wto_cuda_batch_dev(
            test_data.astype(np.float32), channel_range, average_range
        )

        wt1_gpu = cp.asnumpy(cp.asarray(gpu["wt1"]))
        wt2_gpu = cp.asnumpy(cp.asarray(gpu["wt2"]))
        hist_gpu = cp.asnumpy(cp.asarray(gpu["hist"]))

        assert wt1_gpu.shape == cpu["wt1"].shape
        assert wt2_gpu.shape == cpu["wt2"].shape
        assert hist_gpu.shape == cpu["hist"].shape

        assert_close(wt1_gpu, cpu["wt1"], rtol=1e-4, atol=1e-5, msg="WT1 grid mismatch")
        assert_close(wt2_gpu, cpu["wt2"], rtol=1e-4, atol=1e-5, msg="WT2 grid mismatch")
        assert_close(hist_gpu, cpu["hist"], rtol=1e-4, atol=1e-5, msg="Hist grid mismatch")

        assert np.array_equal(gpu["channel_lengths"], cpu["channel_lengths"])
        assert np.array_equal(gpu["average_lengths"], cpu["average_lengths"])

    def test_wto_cuda_many_series_one_param_matches_cpu(self, test_data):
        rows = 1024
        cols = 4
        series = np.vstack(
            [
                test_data[:rows] * (1.0 + 0.01 * j)
                for j in range(cols)
            ]
        ).T

        channel = 9
        average = 21

        cpu_wt1 = np.zeros_like(series)
        cpu_wt2 = np.zeros_like(series)
        cpu_hist = np.zeros_like(series)
        for j in range(cols):
            wt1_j, wt2_j, hist_j = ti.wto(series[:, j], channel, average)
            cpu_wt1[:, j] = wt1_j
            cpu_wt2[:, j] = wt2_j
            cpu_hist[:, j] = hist_j

        gpu = ti.wto_cuda_many_series_one_param_dev(
            series.astype(np.float32), channel, average
        )
        wt1_gpu = cp.asnumpy(cp.asarray(gpu["wt1"]))
        assert wt1_gpu.shape == series.shape

        assert_close(wt1_gpu, cpu_wt1, rtol=1e-4, atol=1e-5, msg="WT1 many-series mismatch")

        wt2_gpu = cp.asnumpy(cp.asarray(gpu["wt2"]))
        hist_gpu = cp.asnumpy(cp.asarray(gpu["hist"]))

        assert_close(wt2_gpu, cpu_wt2, rtol=1e-4, atol=1e-5, msg="WT2 many-series mismatch")
        assert_close(hist_gpu, cpu_hist, rtol=1e-4, atol=1e-5, msg="Hist many-series mismatch")
