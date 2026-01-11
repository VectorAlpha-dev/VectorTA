"""
Simple Python benchmarks for Buff Averages CUDA bindings.
Run with a CUDA-enabled build of the Python module.

Usage (Windows):
  - Activate venv and build: maturin develop --features "python,cuda" --release
  - python -OO benchmarks/bench_buff_averages_cuda.py
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
    module = None
    try:
        import ta_indicators as candidate

        module = candidate
    except Exception:
        try:
            import my_project as candidate

            module = candidate
        except Exception:
            pass
    if module is None:
        raise SystemExit(
            "Module not built. Run: maturin develop --features \"python,cuda\" --release"
        )
    if not hasattr(module, "buff_averages_cuda_batch_dev"):
        try:
            import my_project as candidate

            if hasattr(candidate, "buff_averages_cuda_batch_dev"):
                return candidate
        except Exception:
            pass
        try:
            import ta_indicators as candidate

            if hasattr(candidate, "buff_averages_cuda_batch_dev"):
                return candidate
        except Exception:
            pass
    return module


ti = _import_module()

if not hasattr(ti, "buff_averages_cuda_batch_dev"):
    raise SystemExit(
        "Installed module lacks Buff Averages CUDA bindings. Rebuild with: maturin develop --features \"python,cuda\" --release"
    )


def bench(name, fn, iters=10):
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    dt = (t1 - t0) / iters
    print(f"{name:58s} {dt * 1e3:9.3f} ms")


def main():
    series_len = 50_000
    price = np.full(series_len, np.nan, dtype=np.float32)
    volume = np.full(series_len, np.nan, dtype=np.float32)
    for i in range(3, series_len):
        x = float(i)
        price[i] = np.sin(x * 0.001) + 0.0001 * x
        volume[i] = abs(np.cos(x * 0.0007)) + 0.5

    def run_medium_batch():
        fast_handle, slow_handle = ti.buff_averages_cuda_batch_dev(
            price,
            volume,
            fast_range=(5, 45, 5),
            slow_range=(50, 200, 10),
        )
        _ = cp.asarray(fast_handle)
        _ = cp.asarray(slow_handle)

    large_len = 120_000
    price_large = np.full(large_len, np.nan, dtype=np.float32)
    volume_large = np.full(large_len, np.nan, dtype=np.float32)
    for i in range(4, large_len):
        x = float(i)
        price_large[i] = np.cos(x * 0.0009) + 0.00015 * x
        volume_large[i] = abs(np.sin(x * 0.0005)) + 0.75

    def run_large_batch():
        fast_handle, slow_handle = ti.buff_averages_cuda_batch_dev(
            price_large,
            volume_large,
            fast_range=(4, 20, 4),
            slow_range=(24, 120, 12),
        )
        _ = cp.asarray(fast_handle)
        _ = cp.asarray(slow_handle)

    print("Buff Averages CUDA Python Benchmarks (avg over 10 iters)")
    bench("buff_averages_cuda_batch_dev(50k x 40 combos)", run_medium_batch)
    bench("buff_averages_cuda_batch_dev(120k x 64 combos)", run_large_batch)


if __name__ == "__main__":
    main()
