"""Python binding tests for Ehlers ECEMA CUDA kernels."""
import numpy as np
import pytest

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency
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
    if not hasattr(ti, "ehlers_ecema_cuda_batch_dev"):
        return False
    try:
        demo = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle = ti.ehlers_ecema_cuda_batch_dev(
            demo,
            length_range=(4, 4, 0),
            gain_limit_range=(20, 20, 0),
        )
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  # pragma: no cover - defensive skip
        msg = str(exc).lower()
        if "cuda not available" in msg or "no cuda" in msg:
            return False
        return True


@pytest.mark.skipif(
    not _cuda_available(), reason="CUDA not available or ECEMA CUDA bindings missing"
)
class TestEhlersEcemaCuda:
    @pytest.fixture(scope="class")
    def close_series(self):
        data = load_test_data()["close"].astype(np.float64)
        return data

    def test_ehlers_ecema_cuda_batch_matches_cpu(self, close_series):
        lengths = [12, 20]
        gain_limits = [30, 40]

        handle = ti.ehlers_ecema_cuda_batch_dev(
            close_series.astype(np.float32),
            length_range=(lengths[0], lengths[-1], lengths[1] - lengths[0]),
            gain_limit_range=(gain_limits[0], gain_limits[-1], gain_limits[1] - gain_limits[0]),
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape[0] == len(lengths) * len(gain_limits)
        assert gpu.shape[1] == close_series.shape[0]

        row = 0
        for length in lengths:
            for gain in gain_limits:
                cpu = ti.ehlers_ecema(
                    close_series,
                    length=length,
                    gain_limit=gain,
                )
                assert_close(
                    gpu[row],
                    cpu,
                    rtol=3e-5,
                    atol=3e-6,
                    msg=f"ECEMA CUDA batch mismatch (length={length}, gain={gain})",
                )
                row += 1

    def test_ehlers_ecema_cuda_batch_pine_confirmed(self, close_series):
        lengths = [10, 18]
        gain_limits = [25, 35]

        handle = ti.ehlers_ecema_cuda_batch_dev(
            close_series.astype(np.float32),
            length_range=(lengths[0], lengths[-1], lengths[1] - lengths[0]),
            gain_limit_range=(gain_limits[0], gain_limits[-1], gain_limits[1] - gain_limits[0]),
            pine_compatible=True,
            confirmed_only=True,
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        row = 0
        for length in lengths:
            for gain in gain_limits:
                cpu = ti.ehlers_ecema(
                    close_series,
                    length=length,
                    gain_limit=gain,
                    pine_compatible=True,
                    confirmed_only=True,
                )
                assert_close(
                    gpu[row],
                    cpu,
                    rtol=3e-5,
                    atol=3e-6,
                    msg=f"ECEMA CUDA pine batch mismatch (length={length}, gain={gain})",
                )
                row += 1

    def test_ehlers_ecema_cuda_many_series_one_param_matches_cpu(self, close_series):
        T = 1024
        N = 4
        base = close_series[:T]

        data_tm = np.empty((T, N), dtype=np.float64)
        for j in range(N):
            drift = 1.0 + 0.02 * j
            data_tm[:, j] = base * drift + j * 0.1

        for pine, confirmed, length, gain in (
            (False, False, 22, 45),
            (True, True, 16, 35),
        ):
            cpu_tm = np.empty_like(data_tm)
            for j in range(N):
                cpu_tm[:, j] = ti.ehlers_ecema(
                    data_tm[:, j],
                    length=length,
                    gain_limit=gain,
                    pine_compatible=pine,
                    confirmed_only=confirmed,
                )

            handle = ti.ehlers_ecema_cuda_many_series_one_param_dev(
                data_tm.astype(np.float32),
                length,
                gain,
                pine_compatible=pine,
                confirmed_only=confirmed,
            )
            gpu_tm = cp.asnumpy(cp.asarray(handle))

            assert_close(
                gpu_tm,
                cpu_tm,
                rtol=3e-5,
                atol=3e-6,
                msg=f"ECEMA CUDA many-series mismatch (pine={pine}, confirmed={confirmed})",
            )
