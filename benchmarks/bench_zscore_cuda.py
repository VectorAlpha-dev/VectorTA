"""
Simple Python benchmarks for Zscore CUDA bindings.
Run with a CUDA-enabled build of the Python module.

Usage (Windows):
  - Activate venv and build: maturin develop --features "python,cuda" --release
  - python -OO benchmarks/bench_zscore_cuda.py
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
    if not hasattr(module, "zscore_cuda_batch_dev"):
        for name in ("my_project", "ta_indicators"):
            try:
                candidate = __import__(name)
                if hasattr(candidate, "zscore_cuda_batch_dev"):
                    return candidate
            except Exception:
                continue
    return module


ti = _import_module()

if not hasattr(ti, "zscore_cuda_batch_dev"):
    raise SystemExit(
        "Installed module lacks Zscore CUDA bindings. Rebuild with: maturin develop --features \"python,cuda\" --release"
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
    data = np.full(series_len, np.nan, dtype=np.float32)
    for i in range(5, series_len):
        x = float(i)
        data[i] = np.sin(x * 0.00041) + np.cos(x * 0.00019) + 0.0005 * (i % 9)

    def run_medium_batch():
        handle, _meta = ti.zscore_cuda_batch_dev(
            data,
            period_range=(10, 40, 10),
            nbdev_range=(0.5, 1.5, 0.5),
        )
        _ = cp.asarray(handle)

    large_len = 120_000
    data_large = np.full(large_len, np.nan, dtype=np.float32)
    for i in range(6, large_len):
        x = float(i)
        data_large[i] = np.cos(x * 0.00033) - np.sin(x * 0.00027) + 0.0004 * (i % 11)

    def run_large_batch():
        handle, _meta = ti.zscore_cuda_batch_dev(
            data_large,
            period_range=(12, 48, 12),
            nbdev_range=(0.25, 2.0, 0.25),
        )
        _ = cp.asarray(handle)

    print("Zscore CUDA Python Benchmarks (avg over 10 iters)")
    bench("zscore_cuda_batch_dev(50k x 9 combos)", run_medium_batch)
    bench("zscore_cuda_batch_dev(120k x 28 combos)", run_large_batch)


if __name__ == "__main__":
    main()

