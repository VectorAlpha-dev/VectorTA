"""Python CUDA binding tests for Chande (Chandelier Exit)."""
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
    if not hasattr(ti, "chande_cuda_batch_dev"):
        return False
    try:
        data = load_test_data()
        h = data["high"][:128].astype(np.float32)
        l = data["low"][:128].astype(np.float32)
        c = data["close"][:128].astype(np.float32)
        handle = ti.chande_cuda_batch_dev(h, l, c, (22, 22, 0), (3.0, 3.0, 0.0), "long")
        _ = cp.asarray(handle)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or bindings not built")
class TestChandeCuda:
    @pytest.fixture(scope="class")
    def dataset(self):
        return load_test_data()

    def test_batch_matches_cpu(self, dataset):
        high = dataset["high"].astype(np.float64)
        low = dataset["low"].astype(np.float64)
        close = dataset["close"].astype(np.float64)

        periods = [10, 20, 30]
        mults = [2.0, 3.0]

        handle = ti.chande_cuda_batch_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
            (periods[0], periods[-1], 10),
            (mults[0], mults[-1], 1.0),
            "long",
        )
        gpu = cp.asnumpy(cp.asarray(handle))
        assert gpu.shape == (len(periods) * len(mults), high.shape[0])


        combos = [(p, m) for p in periods for m in mults]
        for row, (p, m) in enumerate(combos):
            cpu = ti.chande(high, low, close, p, m, "long")

            assert_close(gpu[row], cpu, rtol=1e-6, atol=1e-3,
                         msg=f"Chande CUDA batch mismatch (p={p}, m={m})")

