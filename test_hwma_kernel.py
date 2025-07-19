import numpy as np
import my_project as mp

# Test data
data = np.random.randn(1000).astype(np.float64)

# Test different kernels
kernels = [None, "scalar", "avx2", "avx512", "auto"]

print("Testing HWMA with different kernels:")
for kernel in kernels:
    try:
        if kernel is None:
            result = mp.hwma(data, 0.2, 0.1, 0.1)
            print(f"  Default kernel: Success, shape={result.shape}")
        else:
            result = mp.hwma(data, 0.2, 0.1, 0.1, kernel=kernel)
            print(f"  Kernel '{kernel}': Success, shape={result.shape}")
    except Exception as e:
        print(f"  Kernel '{kernel}': Error - {e}")

# Test batch with kernel
print("\nTesting HWMA batch with kernel:")
try:
    result = mp.hwma_batch(data, (0.1, 0.3, 0.1), (0.1, 0.2, 0.1), (0.1, 0.2, 0.1), kernel="avx2")
    print(f"  Batch with AVX2: Success")
    print(f"  Values shape: {result['values'].shape}")
    print(f"  Parameter arrays: na={len(result['na_values'])}, nb={len(result['nb_values'])}, nc={len(result['nc_values'])}")
except Exception as e:
    print(f"  Batch error: {e}")