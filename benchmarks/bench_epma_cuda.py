"""
Python benchmarks for the EPMA CUDA bindings.
Requires a CUDA-enabled build of the Python module.

Usage:
  maturin develop --features "python,cuda" --release
  python -OO benchmarks/bench_epma_cuda.py
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
        except Exception as exc:  
            raise SystemExit(
                "Module not built. Run: maturin develop --features \"python,cuda\" --release"
            ) from exc
    if not hasattr(mod, 'epma_cuda_batch_dev'):
        for name in ('my_project', 'ta_indicators'):
            try:
                imported = __import__(name)  
                if hasattr(imported, 'epma_cuda_batch_dev'):
                    return imported
            except Exception:
                continue
    return mod


ti = _import_module()

if not hasattr(ti, 'epma_cuda_batch_dev'):
    raise SystemExit(
        "Installed module lacks EPMA CUDA bindings. Rebuild with: maturin develop --features \"python,cuda\" --release"
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
    data = np.full(series_len, np.nan, dtype=np.float32)
    for i in range(10, series_len):
        x = float(i)
        data[i] = np.sin(x * 0.0018) + 0.00012 * x

    def run_epma_batch():
        handle = ti.epma_cuda_batch_dev(
            data,
            period_range=(10, 160, 5),
            offset_range=(1, 6, 1),
        )
        _ = cp.asarray(handle)

    T = 16_384
    N = 512
    tm = np.full((T, N), np.nan, dtype=np.float32)
    for j in range(N):
        for t in range(j + 3, T):
            val = float(t) + 0.05 * float(j)
            tm[t, j] = np.cos(val * 0.0031) + 0.00017 * val

    def run_epma_many_series():
        handle = ti.epma_cuda_many_series_one_param_dev(tm, 48, 8)
        _ = cp.asarray(handle)

    print("EPMA CUDA Python Benchmarks (avg over 10 iters)")
    bench("epma_cuda_batch_dev(200k x period/offset grid)", run_epma_batch)
    bench("epma_cuda_many_series_one_param_dev(512 x 16k)", run_epma_many_series)


if __name__ == "__main__":
    main()
