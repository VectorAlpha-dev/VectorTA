"""
Simple Python benchmarks for ALMA CUDA bindings.
Run with a CUDA-enabled build of the Python module.

Usage (Windows):
  - Activate venv and build: maturin develop --features "python,cuda" --release
  - python -OO benchmarks/bench_alma_cuda.py
"""
import time
import numpy as np

try:
    import cupy as cp
except ImportError:
    raise SystemExit("CuPy is required for CUDA benchmarks. Install with `pip install cupy-cuda12x`. ")


def _import_module():
    # Try ta_indicators first (pymodule name), then crate name fallback
    mod = None
    try:
        import ta_indicators as mod
    except Exception:
        try:
            import my_project as mod
        except Exception:
            pass
    if mod is None:
        raise SystemExit("Module not built. Run: maturin develop --features \"python,cuda\" --release")
    # If CUDA attrs missing on the first import, try alternate name once more
    if not hasattr(mod, 'alma_cuda_batch_dev'):
        try:
            import my_project as alt
            if hasattr(alt, 'alma_cuda_batch_dev'):
                return alt
        except Exception:
            pass
        try:
            import ta_indicators as alt
            if hasattr(alt, 'alma_cuda_batch_dev'):
                return alt
        except Exception:
            pass
    return mod

ti = _import_module()

if not hasattr(ti, 'alma_cuda_batch_dev'):
    raise SystemExit("Installed module lacks CUDA bindings. Rebuild with: maturin develop --features \"python,cuda\" --release")


def bench(name, fn, iters=10):
    # Warmup
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    dt = (t1 - t0) / iters
    print(f"{name:48s} {dt*1e3:9.3f} ms")


def main():
    # One series × many params
    series_len = 50_000
    x = np.full(series_len, np.nan, dtype=np.float32)
    for i in range(3, series_len):
        v = float(i)
        x[i] = np.sin(v * 0.001) + 0.0001 * v

    def run_batch():
        handle = ti.alma_cuda_batch_dev(
            x,
            period_range=(9, 240, 12),
            offset_range=(0.05, 0.95, 0.10),
            sigma_range=(1.5, 11.0, 0.5),
        )
        _ = cp.asarray(handle)

    # Many series × one param (time-major)
    T = 50_000
    N = 256
    tm = np.full((T, N), np.nan, dtype=np.float32)
    for j in range(N):
        for t in range(j, T):
            xv = float(t) + 0.1 * float(j)
            tm[t, j] = np.cos(xv * 0.003) + 0.001 * xv

    def run_many_series():
        handle = ti.alma_cuda_many_series_one_param_dev(tm, 14, 0.85, 6.0)
        _ = cp.asarray(handle)

    # Multi-stream batch with moderate combos
    X2 = 100_000
    x2 = np.full(X2, np.nan, dtype=np.float32)
    for i in range(3, X2):
        v = float(i)
        x2[i] = np.sin(v * 0.001) + 0.0001 * v


    print("ALMA CUDA Python Benchmarks (avg over 10 iters)")
    bench("alma_cuda_batch_dev(50k x ~4k params)", run_batch)
    bench("alma_cuda_many_series_one_param_dev(256 x 50k)", run_many_series)
    # multi-stream variant removed

    # New: 1,000,000 x 240 params (period sweep only)
    X3 = 1_000_000
    x3 = np.full(X3, np.nan, dtype=np.float32)
    for i in range(3, X3):
        v = float(i)
        x3[i] = np.sin(v * 0.001) + 0.0001 * v

    def run_batch_1m_240():
        handle = ti.alma_cuda_batch_dev(
            x3,
            period_range=(1, 240, 1),
            offset_range=(0.85, 0.85, 0.0),
            sigma_range=(6.0, 6.0, 0.0),
        )
        _ = cp.asarray(handle)

    bench("alma_cuda_batch_dev(1M x 240 params)", run_batch_1m_240)

    # Optional very large case: 250k x ~4k params (may require >= 6-8GB VRAM)
    try:
        X4 = 250_000
        x4 = np.full(X4, np.nan, dtype=np.float32)
        for i in range(3, X4):
            v = float(i)
            x4[i] = np.sin(v * 0.001) + 0.0001 * v

        def run_batch_250k_4k():
            handle = ti.alma_cuda_batch_dev(
                x4,
                period_range=(1, 240, 1),
                offset_range=(0.25, 0.85, 0.20),  # 4 values
                sigma_range=(3.0, 12.0, 3.0),     # 4 values (approx 3840 combos)
            )
            _ = cp.asarray(handle)

        bench("alma_cuda_batch_dev(250k x ~4k params)", run_batch_250k_4k)
    except Exception as e:
        print(f"[skip large] {e}")


if __name__ == "__main__":
    main()
