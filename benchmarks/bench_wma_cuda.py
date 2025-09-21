"""
Python benchmarks for the WMA CUDA bindings.
Requires a CUDA-enabled build of the Python module.

Usage:
  maturin develop --features "python,cuda" --release
  python -OO benchmarks/bench_wma_cuda.py
"""
import time
import numpy as np

try:
    import cupy as cp
except ImportError as exc:  # pragma: no cover - benchmark helper
    raise SystemExit(
        "CuPy is required for CUDA benchmarks. Install with `pip install cupy-cuda12x`."
    ) from exc


def _import_module():
    try:
        import ta_indicators as mod  # type: ignore
    except Exception:
        try:
            import my_project as mod  # type: ignore
        except Exception as exc:  # pragma: no cover - import helper
            raise SystemExit(
                "Module not built. Run: maturin develop --features \"python,cuda\" --release"
            ) from exc
    if not hasattr(mod, 'wma_cuda_batch_dev'):
        try:
            import my_project as alt  # type: ignore
            if hasattr(alt, 'wma_cuda_batch_dev'):
                return alt
        except Exception:
            pass
        try:
            import ta_indicators as alt  # type: ignore
            if hasattr(alt, 'wma_cuda_batch_dev'):
                return alt
        except Exception:
            pass
    return mod


ti = _import_module()

if not hasattr(ti, 'wma_cuda_batch_dev'):
    raise SystemExit(
        "Installed module lacks WMA CUDA bindings. Rebuild with: maturin develop --features \"python,cuda\" --release"
    )


def bench(name, fn, iters=10):
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    dt = (t1 - t0) / iters
    print(f"{name:50s} {dt * 1e3:9.3f} ms")


def main():
    series_len = 200_000
    data = np.full(series_len, np.nan, dtype=np.float32)
    for i in range(4, series_len):
        v = float(i)
        data[i] = np.sin(v * 0.0015) + 0.0002 * v

    def run_wma_batch():
        handle = ti.wma_cuda_batch_dev(data, period_range=(4, 192, 2))
        _ = cp.asarray(handle)

    T = 16_384
    N = 512
    tm = np.full((T, N), np.nan, dtype=np.float32)
    for j in range(N):
        for t in range(j, T):
            val = float(t) + 0.05 * float(j)
            tm[t, j] = np.cos(val * 0.0025) + 0.00015 * val

    def run_wma_many_series():
        handle = ti.wma_cuda_many_series_one_param_dev(tm, 64)
        _ = cp.asarray(handle)

    print("WMA CUDA Python Benchmarks (avg over 10 iters)")
    bench("wma_cuda_batch_dev(200k x 95 periods)", run_wma_batch)
    bench("wma_cuda_many_series_one_param_dev(512 x 16k)", run_wma_many_series)


if __name__ == "__main__":
    main()
