"""
Python benchmarks for the TrAdjEMA CUDA bindings.
Requires a CUDA-enabled build of the Python module.

Usage:
  maturin develop --features "python,cuda" --release
  python -OO benchmarks/bench_tradjema_cuda.py
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
    if not hasattr(mod, 'tradjema_cuda_batch_dev'):
        for candidate in ('my_project', 'ta_indicators'):
            try:
                module = __import__(candidate)  
                if hasattr(module, 'tradjema_cuda_batch_dev'):
                    return module
            except Exception:
                continue
    return mod


ti = _import_module()

if not hasattr(ti, 'tradjema_cuda_batch_dev'):
    raise SystemExit(
        "Installed module lacks TrAdjEMA CUDA bindings. Rebuild with: maturin develop --features \"python,cuda\" --release"
    )


def bench(name, fn, iters=10):
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    dt = (t1 - t0) / iters
    print(f"{name:55s} {dt * 1e3:9.3f} ms")


def _gen_one_series(series_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high = np.full(series_len, np.nan, dtype=np.float32)
    low = np.full(series_len, np.nan, dtype=np.float32)
    close = np.full(series_len, np.nan, dtype=np.float32)
    for i in range(8, series_len):
        t = float(i)
        trend = 0.0004 * t
        wave = np.sin(t * 0.0023) * 0.7
        base = trend + wave
        close[i] = base
        high[i] = base + 0.28 + 0.012 * (i % 11)
        low[i] = base - 0.30 - 0.015 * (i % 7)
    return high, low, close


def _gen_many_series(time_len: int, num_series: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high = np.full((time_len, num_series), np.nan, dtype=np.float32)
    low = np.full((time_len, num_series), np.nan, dtype=np.float32)
    close = np.full((time_len, num_series), np.nan, dtype=np.float32)
    for j in range(num_series):
        scale = 1.0 + 0.04 * j
        for t in range(j + 6, time_len):
            tf = float(t)
            wave = np.sin(tf * 0.0035 + j * 0.25)
            drift = 0.00035 * tf + 0.02 * j
            base = (drift + 0.6 * wave) * scale
            close[t, j] = base
            high[t, j] = base + 0.26 + 0.01 * j
            low[t, j] = base - 0.29 - 0.008 * j
    return high, low, close


def main():
    series_len = 150_000
    high, low, close = _gen_one_series(series_len)

    def run_tradjema_batch():
        handle = ti.tradjema_cuda_batch_dev(
            high,
            low,
            close,
            length_range=(16, 128, 8),
            mult_range=(5.0, 15.0, 1.0),
        )
        _ = cp.asarray(handle)

    T = 12_288
    N = 384
    high_tm, low_tm, close_tm = _gen_many_series(T, N)

    def run_tradjema_many_series():
        handle = ti.tradjema_cuda_many_series_one_param_dev(
            high_tm,
            low_tm,
            close_tm,
            34,
            8.0,
        )
        _ = cp.asarray(handle)

    print("TrAdjEMA CUDA Python Benchmarks (avg over 10 iters)")
    bench("tradjema_cuda_batch_dev(150k x 165 combos)", run_tradjema_batch)
    bench("tradjema_cuda_many_series_one_param_dev(384 x 12k)", run_tradjema_many_series)


if __name__ == "__main__":
    main()
