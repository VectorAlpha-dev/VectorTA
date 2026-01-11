"""
Python benchmarks for the NAMA CUDA bindings.
Requires a CUDA-enabled build of the Python module.

Usage:
  maturin develop --features "python,cuda" --release
  python -OO benchmarks/bench_nama_cuda.py
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
    return mod


ti = _import_module()

if not hasattr(ti, 'nama_cuda_batch_dev'):
    raise SystemExit(
        "Installed module lacks NAMA CUDA bindings. Rebuild with: maturin develop --features \"python,cuda\" --release"
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
        x = float(i)
        data[i] = np.sin(x * 0.0027) + 0.00031 * x

    def run_nama_batch():
        handle = ti.nama_cuda_batch_dev(data, period_range=(6, 120, 3))
        _ = cp.asarray(handle)

    T = 16_384
    N = 512
    tm = np.full((T, N), np.nan, dtype=np.float32)
    for j in range(N):
        for t in range(j + 6, T):
            arg = float(t) * 0.0026 + j * 0.13
            tm[t, j] = np.cos(arg) + 0.00028 * t + 0.012 * j

    def run_nama_many_series():
        handle = ti.nama_cuda_many_series_one_param_dev(tm, 40)
        _ = cp.asarray(handle)

    print("NAMA CUDA Python Benchmarks (avg over 10 iters)")
    bench("nama_cuda_batch_dev(200k x 39 periods)", run_nama_batch)
    bench("nama_cuda_many_series_one_param_dev(512 x 16k)", run_nama_many_series)


if __name__ == "__main__":
    main()
