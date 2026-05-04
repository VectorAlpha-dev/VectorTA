import numpy as np
import pytest

try:
    import cupy as cp
except Exception:
    cp = None

try:
    import vector_ta as ti
except Exception:
    ti = None


PLAN_SYMBOLS = [
    "medium_ad_cuda_batch_plan_create",
    "frama_cuda_batch_plan_create",
    "vpwma_cuda_batch_plan_create",
    "vpci_cuda_batch_plan_create",
    "vwma_cuda_batch_plan_create",
    "rsmk_cuda_batch_plan_create",
    "mab_cuda_batch_plan_create",
    "vwmacd_cuda_batch_plan_create",
]


def _cuda_plan_available():
    if ti is None or cp is None:
        return False
    return all(hasattr(ti, name) for name in PLAN_SYMBOLS)


pytestmark = pytest.mark.skipif(
    not _cuda_plan_available(),
    reason="CUDA plan bindings or CuPy are unavailable",
)


def _series(n=512):
    close = np.linspace(1.0, 2.0, n, dtype=np.float32)
    volume = np.linspace(2.0, 3.0, n, dtype=np.float32)
    close[:10] = np.nan
    volume[:10] = np.nan
    high = close + np.float32(0.5)
    low = close - np.float32(0.5)
    return close, volume, high, low


def _device_to_numpy(handle):
    return cp.asnumpy(cp.asarray(handle))


def _assert_same(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5, equal_nan=True)


def test_cuda_batch_plans_match_existing_cuda_batch_outputs():
    close, volume, high, low = _series()
    n = close.size
    first_valid = 10

    period = (5, 7, 2)
    root = _device_to_numpy(ti.medium_ad_cuda_batch_dev(close, period))
    plan = ti.medium_ad_cuda_batch_plan_create(n, first_valid, period)
    _assert_same(plan.execute(close)["values"], root)

    frama_ranges = ((6, 8, 2), (20, 20, 0), (198, 198, 0))
    root, _ = ti.frama_cuda_batch_dev(high, low, close, *frama_ranges)
    plan = ti.frama_cuda_batch_plan_create(n, first_valid, *frama_ranges)
    _assert_same(plan.execute(high, low, close)["values"], _device_to_numpy(root))

    root, _ = ti.vpwma_cuda_batch_dev(close, period, (1.0, 2.0, 1.0))
    plan = ti.vpwma_cuda_batch_plan_create(n, first_valid, period, (1.0, 2.0, 1.0))
    _assert_same(plan.execute(close)["values"], _device_to_numpy(root))

    root = ti.vpci_cuda_batch_dev(close, volume, period, (10, 12, 2))
    plan = ti.vpci_cuda_batch_plan_create(n, first_valid, period, (10, 12, 2))
    plan_out = plan.execute(close, volume)
    _assert_same(plan_out["vpci"], _device_to_numpy(root["vpci"]))
    _assert_same(plan_out["vpcis"], _device_to_numpy(root["vpcis"]))

    root = ti.vwma_cuda_batch_dev(close, volume, period)
    plan = ti.vwma_cuda_batch_plan_create(n, first_valid, period)
    _assert_same(plan.execute(close, volume), _device_to_numpy(root))

    root_indicator, root_signal = ti.rsmk_cuda_batch_dev(
        close, volume, (20, 22, 2), (3, 5, 2), (10, 12, 2)
    )
    plan = ti.rsmk_cuda_batch_plan_create(
        n, first_valid, (20, 22, 2), (3, 5, 2), (10, 12, 2)
    )
    plan_out = plan.execute(close, volume)
    _assert_same(plan_out["indicator"], _device_to_numpy(root_indicator))
    _assert_same(plan_out["signal"], _device_to_numpy(root_signal))

    root_upper, root_middle, root_lower = ti.mab_cuda_batch_dev(
        close, period, (10, 12, 2), (1.0, 1.0, 0.0), (1.0, 1.0, 0.0)
    )
    plan = ti.mab_cuda_batch_plan_create(
        n, first_valid, period, (10, 12, 2), (1.0, 1.0, 0.0), (1.0, 1.0, 0.0)
    )
    plan_out = plan.execute(close)
    _assert_same(plan_out["upper"], _device_to_numpy(root_upper))
    _assert_same(plan_out["middle"], _device_to_numpy(root_middle))
    _assert_same(plan_out["lower"], _device_to_numpy(root_lower))

    root = ti.vwmacd_cuda_batch_dev(close, volume, period, (12, 14, 2), (4, 6, 2))
    plan = ti.vwmacd_cuda_batch_plan_create(n, first_valid, period, (12, 14, 2), (4, 6, 2))
    plan_out = plan.execute(close, volume)
    _assert_same(plan_out["macd"], _device_to_numpy(root["macd"]))
    _assert_same(plan_out["signal"], _device_to_numpy(root["signal"]))
    _assert_same(plan_out["hist"], _device_to_numpy(root["hist"]))
