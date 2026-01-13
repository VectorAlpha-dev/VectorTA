"""Python binding tests for Ehlers PMA CUDA kernels."""
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
    if not hasattr(ti, "ehlers_pma_cuda_batch_dev"):
        return False
    try:
        sample = np.linspace(1.0, 3.0, 32, dtype=np.float32)
        handles = ti.ehlers_pma_cuda_batch_dev(
            sample, (0, 0, 0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        )
        predict_handle, trigger_handle = handles
        _ = cp.asarray(predict_handle)
        _ = cp.asarray(trigger_handle)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "no cuda device" in msg:
            return False
        return True


@pytest.mark.skipif(
    not _cuda_available(), reason="CUDA not available or cuda bindings not built"
)
class TestEhlersPmaCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_ehlers_pma_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"].astype(np.float64)
        cpu_predict, cpu_trigger = ti.ehlers_pma(close)

        predict_handle, trigger_handle = ti.ehlers_pma_cuda_batch_dev(
            close.astype(np.float32), (0, 0, 0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        )
        gpu_predict = cp.asnumpy(cp.asarray(predict_handle))[0]
        gpu_trigger = cp.asnumpy(cp.asarray(trigger_handle))[0]

        assert_close(
            gpu_predict,
            cpu_predict,
            rtol=5e-5,
            atol=5e-5,
            msg="CUDA Ehlers PMA predict mismatch",
        )
        assert_close(
            gpu_trigger,
            cpu_trigger,
            rtol=5e-5,
            atol=5e-5,
            msg="CUDA Ehlers PMA trigger mismatch",
        )

    def test_ehlers_pma_cuda_many_series_one_param_matches_cpu(self, test_data):
        rows = 720
        cols = 4
        base = test_data["close"].astype(np.float64)
        data_tm = np.full((rows, cols), np.nan, dtype=np.float64)
        for j in range(cols):
            start = j + 2
            data_tm[start:, j] = (
                base[: rows - start] * (1.0 + 0.015 * j)
                + 0.001 * np.linspace(0.0, rows - start - 1, rows - start)
            )

        cpu_predict = np.full_like(data_tm, np.nan)
        cpu_trigger = np.full_like(data_tm, np.nan)
        for j in range(cols):
            predict, trigger = ti.ehlers_pma(data_tm[:, j])
            cpu_predict[:, j] = predict
            cpu_trigger[:, j] = trigger

        predict_handle, trigger_handle = ti.ehlers_pma_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32)
        )
        gpu_predict = cp.asnumpy(cp.asarray(predict_handle))
        gpu_trigger = cp.asnumpy(cp.asarray(trigger_handle))

        assert_close(
            gpu_predict,
            cpu_predict,
            rtol=5e-5,
            atol=5e-5,
            msg="CUDA Ehlers PMA many-series predict mismatch",
        )
        assert_close(
            gpu_trigger,
            cpu_trigger,
            rtol=5e-5,
            atol=5e-5,
            msg="CUDA Ehlers PMA many-series trigger mismatch",
        )
