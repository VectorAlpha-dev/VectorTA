"""
Python benchmarks for the Ehlers ECEMA CUDA bindings.
Requires a CUDA-enabled build of the Python extension.

Usage:
  maturin develop --features "python,cuda" --release
  python -OO benchmarks/bench_ehlers_ecema_cuda.py
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
    candidates = ("my_project", "ta_indicators")
    for name in candidates:
        try:
            mod = __import__(name)  # type: ignore
        except Exception:
            continue
        if hasattr(mod, "ehlers_ecema_cuda_batch_dev"):
            return mod
    raise SystemExit(
        "Module not built with CUDA bindings. Run: maturin develop --features \"python,cuda\" --release"
    )


ti = _import_module()


def bench(label, fn, iters=10):
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    print(f"{label:60s} {(t1 - t0) * 1e3 / iters:9.3f} ms")


def main():
    if not hasattr(ti, "ehlers_ecema_cuda_batch_dev"):
        raise SystemExit("ECEMA CUDA bindings missing from loaded module")

    series_len = 200_000
    data = np.full(series_len, np.nan, dtype=np.float32)
    for i in range(6, series_len):
        v = float(i)
        data[i] = np.sin(v * 0.0019) + np.cos(v * 0.0009) * 0.4 + 0.00027 * v

    def run_ecema_batch():
        handle = ti.ehlers_ecema_cuda_batch_dev(
            data,
            length_range=(6, 126, 6),
            gain_limit_range=(10, 70, 10),
        )
        _ = cp.asarray(handle)

    T = 32_768
    N = 256
    tm = np.full((T, N), np.nan, dtype=np.float32)
    for j in range(N):
        for t in range(j + 8, T):
            val = float(t) + 0.11 * float(j)
            tm[t, j] = np.sin(val * 0.0021) + np.cos(val * 0.0011) * 0.3 + 0.0004 * val

    def run_ecema_many_series():
        handle = ti.ehlers_ecema_cuda_many_series_one_param_dev(
            tm,
            length=26,
            gain_limit=50,
        )
        _ = cp.asarray(handle)

    print("Ehlers ECEMA CUDA Python Benchmarks (avg over 10 iters)")
    bench("ehlers_ecema_cuda_batch_dev(200k x 120 combos)", run_ecema_batch)
    bench("ehlers_ecema_cuda_many_series_one_param_dev(256 x 32k)", run_ecema_many_series)


if __name__ == "__main__":
    main()
