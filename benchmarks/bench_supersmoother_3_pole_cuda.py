"""
Python benchmarks for the SuperSmoother 3-Pole CUDA bindings.
Requires a CUDA-enabled build of the Python module.

Usage:
  maturin develop --features "python,cuda" --release
  python -OO benchmarks/bench_supersmoother_3_pole_cuda.py
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
    if not hasattr(mod, 'supersmoother_3_pole_cuda_batch_dev'):
        try:
            import my_project as alt  
            if hasattr(alt, 'supersmoother_3_pole_cuda_batch_dev'):
                return alt
        except Exception:
            pass
        try:
            import ta_indicators as alt  
            if hasattr(alt, 'supersmoother_3_pole_cuda_batch_dev'):
                return alt
        except Exception:
            pass
    return mod


ti = _import_module()

if not hasattr(ti, 'supersmoother_3_pole_cuda_batch_dev'):
    raise SystemExit(
        "Installed module lacks supersmoother_3_pole CUDA bindings. Rebuild with: maturin develop --features \"python,cuda\" --release"
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
    for i in range(6, series_len):
        v = float(i)
        data[i] = np.sin(v * 0.0017) + 0.0002 * v

    def run_ss3p_batch():
        handle = ti.supersmoother_3_pole_cuda_batch_dev(
            data,
            period_range=(6, 96, 2),
        )
        _ = cp.asarray(handle)

    rows = 16_384
    cols = 256
    tm = np.full((rows, cols), np.nan, dtype=np.float32)
    for j in range(cols):
        for t in range(j, rows):
            val = float(t) + 0.12 * float(j)
            tm[t, j] = np.cos(val * 0.0021) + 0.00018 * val

    def run_ss3p_many_series():
        handle = ti.supersmoother_3_pole_cuda_many_series_one_param_dev(
            tm,
            24,
        )
        _ = cp.asarray(handle)

    print("SuperSmoother 3-Pole CUDA Python Benchmarks (avg over 10 iters)")
    bench("supersmoother_3_pole_cuda_batch_dev(200k x 46 periods)", run_ss3p_batch)
    bench(
        "supersmoother_3_pole_cuda_many_series_one_param_dev(256 x 16k)",
        run_ss3p_many_series,
    )


if __name__ == "__main__":
    main()
