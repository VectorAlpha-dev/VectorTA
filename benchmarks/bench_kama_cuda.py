"""
Python benchmarks for the KAMA CUDA bindings.
Requires a CUDA-enabled build of the Python module.

Usage:
  maturin develop --features "python,cuda" --release
  python -OO benchmarks/bench_kama_cuda.py
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
    if not hasattr(mod, 'kama_cuda_batch_dev'):
        for alt_name in ('my_project', 'ta_indicators'):
            try:
                alt = __import__(alt_name)  
            except Exception:
                continue
            if hasattr(alt, 'kama_cuda_batch_dev'):
                return alt
    return mod


ti = _import_module()

if not hasattr(ti, 'kama_cuda_batch_dev'):
    raise SystemExit(
        "Installed module lacks KAMA CUDA bindings. Rebuild with: maturin develop --features \"python,cuda\" --release"
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
    for i in range(12, series_len):
        v = float(i)
        data[i] = np.sin(v * 0.0021) + 0.00017 * v

    def run_kama_batch():
        handle = ti.kama_cuda_batch_dev(data, period_range=(5, 160, 5))
        _ = cp.asarray(handle)

    T = 16_384
    N = 512
    tm = np.full((T, N), np.nan, dtype=np.float32)
    for j in range(N):
        for t in range(j + 8, T):
            val = float(t) + 0.11 * float(j)
            tm[t, j] = np.sin(val * 0.0027) + 0.00019 * val

    def run_kama_many_series():
        handle = ti.kama_cuda_many_series_one_param_dev(tm, 42)
        _ = cp.asarray(handle)

    print("KAMA CUDA Python Benchmarks (avg over 10 iters)")
    bench("kama_cuda_batch_dev(200k x 32 periods)", run_kama_batch)
    bench("kama_cuda_many_series_one_param_dev(512 x 16k)", run_kama_many_series)


if __name__ == "__main__":
    main()
