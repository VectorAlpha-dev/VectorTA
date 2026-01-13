import numpy as np
import pytest

try:
    import ta_indicators as ti
except Exception:
    ti = None


@pytest.mark.skipif(ti is None or not hasattr(ti, "kst_cuda_batch_dev"), reason="CUDA KST not available")
def test_kst_cuda_batch_matches_cpu():
    rng = np.random.default_rng(42)
    n = 4096
    data = np.full(n, np.nan, dtype=np.float64)
    x = np.arange(n, dtype=np.float64)
    data[5:] = np.sin(x[5:] * 0.0012) + 0.0002 * x[5:]


    line_cpu, sig_cpu = ti.kst(
        data,
        sma_period1=5, sma_period2=5, sma_period3=5, sma_period4=10,
        roc_period1=5, roc_period2=7, roc_period3=10, roc_period4=15,
        signal_period=5,
    )


    s1 = (5, 7, 2); s2=(5,5,0); s3=(5,5,0); s4=(10,10,0)
    r1 = (5,5,0); r2=(7,7,0); r3=(10,10,0); r4=(15,15,0)
    sg = (5,5,0)

    dev_line, dev_sig = ti.kst_cuda_batch_dev(
        data.astype(np.float32),
        s1, s2, s3, s4,
        r1, r2, r3, r4,
        sg,
    )

    gpu_line = np.empty_like(line_cpu, dtype=np.float32)
    gpu_sig  = np.empty_like(sig_cpu,  dtype=np.float32)
    dev_line.buf.copy_to(gpu_line)
    dev_sig.buf.copy_to(gpu_sig)

    tol = 1e-3
    np.testing.assert_allclose(gpu_line.astype(np.float64), line_cpu, rtol=0, atol=tol, equal_nan=True)
    np.testing.assert_allclose(gpu_sig.astype(np.float64), sig_cpu, rtol=0, atol=tol, equal_nan=True)


@pytest.mark.skipif(ti is None or not hasattr(ti, "kst_cuda_many_series_one_param_dev"), reason="CUDA KST not available")
def test_kst_cuda_many_series_one_param_matches_cpu():
    cols = 6
    rows = 4096
    data_tm = np.full((rows, cols), np.nan, dtype=np.float64)
    for s in range(cols):
        t = np.arange(rows, dtype=np.float64)
        data_tm[5:, s] = np.sin((t[5:] + 0.1*s) * 0.0011) + 0.00015 * (t[5:] + 0.2*s)


    line_cpu = np.empty_like(data_tm)
    sig_cpu  = np.empty_like(data_tm)
    for s in range(cols):
        line, sig = ti.kst(
            data_tm[:, s],
            10, 10, 10, 15,
            10, 15, 20, 30,
            9,
        )
        line_cpu[:, s] = line
        sig_cpu[:, s]  = sig

    dev_line, dev_sig = ti.kst_cuda_many_series_one_param_dev(
        data_tm.astype(np.float32).ravel(),
        cols, rows,
        10, 10, 10, 15,
        10, 15, 20, 30,
        9,
    )
    gpu_line = np.empty(rows*cols, dtype=np.float32)
    gpu_sig  = np.empty(rows*cols, dtype=np.float32)
    dev_line.buf.copy_to(gpu_line)
    dev_sig.buf.copy_to(gpu_sig)

    gpu_line = gpu_line.reshape(rows, cols).astype(np.float64)
    gpu_sig  = gpu_sig.reshape(rows, cols).astype(np.float64)
    tol = 1.2e-3
    np.testing.assert_allclose(gpu_line, line_cpu, rtol=0, atol=tol, equal_nan=True)
    np.testing.assert_allclose(gpu_sig,  sig_cpu,  rtol=0, atol=tol, equal_nan=True)

