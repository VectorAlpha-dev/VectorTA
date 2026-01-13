"""
Python benchmarks for the HighPass CUDA bindings.
Requires a CUDA-enabled build of the Python module.

Usage:
  maturin develop --features "python,cuda" --release
  python -OO benchmarks/bench_highpass_cuda.py
"""
import time
import numpy as np

try:
    import cupy as cp
except ImportError as exc:
    raise SystemExit(
        "CuPy is required for CUDA benchmarks. Install with `pip install cupy-cuda12x`."
    ) from exc


def _import_module():
    try:
        import ta_indicators as mod
    except Exception:
        try:
            import my_project as mod
        except Exception as inner:
            raise SystemExit(
                "Module not built. Run: maturin develop --features \"python,cuda\" --release"
            ) from inner
    if not hasattr(mod, 'highpass_cuda_batch_dev'):
        for alt in ('my_project', 'ta_indicators'):
            try:
                candidate = __import__(alt)
            except Exception:
                continue
            if hasattr(candidate, 'highpass_cuda_batch_dev'):
                return candidate
    return mod


ti = _import_module()

if not hasattr(ti, 'highpass_cuda_batch_dev'):
    raise SystemExit(
        "Installed module lacks HighPass CUDA bindings. Rebuild with: maturin develop --features \"python,cuda\" --release"
    )


def bench(name, fn, iters=10):
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    dt = (t1 - t0) / iters
    print(f"{name:55s} {dt * 1e3:9.3f} ms")


def main():
    series_len = 200_000
    data = np.empty(series_len, dtype=np.float32)
    for i in range(series_len):
        t = float(i)
        data[i] = np.sin(t * 0.0024) + 0.00029 * t

    def run_highpass_batch():
        handle = ti.highpass_cuda_batch_dev(
            data,
            period_range=(HIGHPASS_PERIOD_START, HIGHPASS_PERIOD_END, HIGHPASS_PERIOD_STEP),
        )
        _ = cp.asarray(handle)

    T = 16_384
    N = 512
    tm = np.empty((T, N), dtype=np.float32)
    for j in range(N):
        shift = 0.35 * j
        scale = 1.0 + 0.01 * j
        for t in range(T):
            val = float(t)
            tm[t, j] = np.sin(val * 0.0031 + shift) * scale + 0.00037 * val

    def run_highpass_many_series():
        handle = ti.highpass_cuda_many_series_one_param_dev(tm, HIGHPASS_PERIOD)
        _ = cp.asarray(handle)

    print("HighPass CUDA Python Benchmarks (avg over 10 iters)")
    bench(
        "highpass_cuda_batch_dev(200k x periods)",
        run_highpass_batch,
    )
    bench(
        "highpass_cuda_many_series_one_param_dev(512 x 16k)",
        run_highpass_many_series,
    )


HIGHPASS_PERIOD_START = 8
HIGHPASS_PERIOD_END = 160
HIGHPASS_PERIOD_STEP = 4
HIGHPASS_PERIOD = 48


if __name__ == "__main__":
    main()
