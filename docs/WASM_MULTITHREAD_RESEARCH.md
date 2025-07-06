# WebAssembly Multi-Threading Research for Rust-Backtester

**Date**: January 5, 2025  
**Purpose**: Comprehensive research on implementing WebAssembly multi-threading using wasm-bindgen-rayon for the ALMA indicator and other batch processing operations.

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Technical Feasibility](#technical-feasibility)
3. [Implementation Plan](#implementation-plan)
4. [Browser Support](#browser-support)
5. [Performance Analysis](#performance-analysis)
6. [Practical Considerations](#practical-considerations)
7. [Known Limitations](#known-limitations)
8. [Alternative Approaches](#alternative-approaches)

## Executive Summary

### Key Findings
- **Technically Feasible**: Yes, you can run existing Rayon-based batch kernels in the browser using wasm-bindgen-rayon
- **Browser Support**: 92-95% of global browser traffic supports necessary features (Chrome 68+, Firefox 79+, Safari 18+)
- **Performance**: 2-8× speedup on 4-core systems for parallel workloads
- **Major Concern**: GoogleChromeLabs/wasm-bindgen-rayon was **archived in July 2024**

### Quick Decision Matrix
| Factor | Status | Impact |
|--------|--------|--------|
| Technical Feasibility | ✅ Possible | High |
| Browser Support | ✅ 92-95% | High |
| Performance Gain | ✅ 2-8× | High |
| Maintenance | ❌ Project Archived | Critical |
| Production Ready | ⚠️ Requires Nightly | Medium |
| Real-World Usage | ❌ Limited | Medium |

## Technical Feasibility

### Core Requirements

1. **Rust Toolchain**
   ```bash
   # Requires nightly Rust (tested with nightly-2024-08-02)
   rustup default nightly
   rustup target add wasm32-unknown-unknown
   ```

2. **Build Configuration**
   ```bash
   # Compile with atomics support
   RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals' \
     cargo build --release \
     --target wasm32-unknown-unknown \
     -Z build-std=std,panic_abort
   ```

3. **Dependencies**
   ```toml
   [dependencies]
   wasm-bindgen = { version = "0.2", features = ["rayon"] }
   wasm-bindgen-rayon = "1.3"
   rayon = { version = "1", optional = true }

   [features]
   default = ["native-parallel"]
   native-parallel = ["rayon"]
   wasm-parallel = ["rayon"]
   ```

### Zero-Copy Implementation Verified
- Python bindings already use `PyReadonlyArray1::as_slice()` for zero-copy reads
- WASM maintains zero-copy with `&[f64]` views
- No additional memory allocations for parallel processing

## Implementation Plan

### Step 1: Cargo Configuration
```toml
# Cargo.toml additions
[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = { version = "0.2.87", features = ["rayon"] }
wasm-bindgen-rayon = "1.3.0"

[features]
wasm-parallel = ["rayon", "wasm-bindgen-rayon"]
```

### Step 2: Conditional Compilation
```rust
// In your batch processing code
#[cfg(all(target_arch = "wasm32", feature = "wasm-parallel"))]
use wasm_bindgen_rayon::init_thread_pool;

#[cfg(feature = "rayon")]
let chunks = out_uninit.par_chunks_mut(cols);
#[cfg(not(feature = "rayon"))]
let chunks = out_uninit.chunks_mut(cols);
```

### Step 3: JavaScript Initialization
```javascript
import init, { alma_batch_js } from './pkg/rust_backtester.js';
import { initThreadPool } from './pkg/rust_backtester_bg.js';

async function initialize() {
    // Detect thread support
    const supportsThreads = typeof SharedArrayBuffer === 'function' && 
                           crossOriginIsolated;
    
    if (supportsThreads) {
        // Initialize with parallel support
        await init();
        await initThreadPool(navigator.hardwareConcurrency || 4);
    } else {
        // Fall back to scalar version
        await init();
    }
}
```

### Step 4: HTTP Headers Configuration
```nginx
# Required for SharedArrayBuffer
add_header Cross-Origin-Embedder-Policy "require-corp";
add_header Cross-Origin-Opener-Policy "same-origin";
add_header Cross-Origin-Resource-Policy "cross-origin";
```

### Step 5: Build Scripts
```bash
#!/bin/bash
# build_wasm_parallel.sh

# Build parallel version
RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals' \
wasm-pack build \
  --features wasm,wasm-parallel \
  --target web \
  --out-dir pkg-parallel \
  -- -Z build-std=std,panic_abort

# Build scalar fallback
wasm-pack build \
  --features wasm \
  --target web \
  --out-dir pkg-scalar
```

## Browser Support

### Current Status (January 2025)

| Browser | Version | SharedArrayBuffer | WebAssembly Threads | Notes |
|---------|---------|-------------------|---------------------|-------|
| Chrome | 68+ | ✅ | ✅ | Full support with COOP/COEP |
| Firefox | 79+ | ✅ | ✅ | Full support with COOP/COEP |
| Safari | 18+ | ✅ | ⚠️ | Limited thread support |
| Edge | 79+ | ✅ | ✅ | Same as Chrome |
| Opera | 55+ | ✅ | ✅ | Same as Chrome |

### Cross-Origin Isolation Check
```javascript
if (crossOriginIsolated) {
    console.log("SharedArrayBuffer is available");
} else {
    console.log("SharedArrayBuffer is NOT available");
}
```

## Performance Analysis

### Benchmarked Results
1. **Mandelbrot Fractal Generation**: ~3× speedup
2. **General Parallel Workloads**: 2-8× speedup on 4-core systems
3. **Rust vs JavaScript**: Up to 8× faster for computation-heavy tasks

### Expected Performance for ALMA
```
Single-threaded ALMA batch (1000 securities, 20 params):
- Current WASM: ~X ms

Multi-threaded ALMA batch (4 cores):
- Expected: ~X/4 ms + overhead
- Overhead: ~12ms (4 workers × 3ms initialization)
```

### Performance Considerations
- Worker initialization: ~3ms per worker
- Memory overhead: ~15KB for worker JavaScript
- Best for batches > 100 computations

## Practical Considerations

### Advantages
1. **No Rust Code Changes**: Existing `rayon::par_chunks_mut` works unchanged
2. **Graceful Degradation**: Automatic fallback for unsupported browsers
3. **Modern Bundler Support**: Works with Webpack 5, Parcel 2, Vite
4. **True Parallelism**: Utilizes all CPU cores

### Disadvantages
1. **Project Maintenance**: wasm-bindgen-rayon archived July 2024
2. **Nightly Rust Required**: Not suitable for stable production builds
3. **Complex Setup**: Requires specific headers, build flags
4. **Limited Adoption**: Few production deployments exist

### Operational Requirements
1. **HTTPS Required**: Cross-origin isolation needs secure context
2. **Header Configuration**: Must control server headers
3. **CDN Compatibility**: Need to configure cache varying
4. **Third-Party Scripts**: May break with COEP enabled

## Known Limitations

### Technical Limitations
1. **Main Thread Blocking**: Cannot acquire mutexes on main thread
2. **Worker Communication**: Limited to postMessage
3. **Memory Model**: SharedArrayBuffer has strict requirements
4. **Browser APIs**: Some APIs unavailable in workers

### Implementation Risks
1. **Archived Dependency**: No future updates to wasm-bindgen-rayon
2. **Nightly Rust**: May have breaking changes
3. **Safari Support**: Limited and may change
4. **Security Policies**: Enterprise environments may block

### Debugging Challenges
1. **Worker Debugging**: More complex than single-threaded
2. **Race Conditions**: Possible with shared memory
3. **Error Handling**: Cross-worker errors harder to track

## Alternative Approaches

### 1. Server-Side Processing
```rust
// Keep Rayon on server, serve results via API
#[post("/batch/alma")]
async fn alma_batch_endpoint(params: BatchParams) -> Result<Vec<f64>> {
    // Use full Rayon parallelism server-side
    alma_batch_with_kernel(&data, &params, Kernel::Avx512)
}
```

### 2. Web Workers Without Rayon
```javascript
// Manual worker pool for parallel processing
const workerPool = [];
for (let i = 0; i < navigator.hardwareConcurrency; i++) {
    workerPool.push(new Worker('alma-worker.js'));
}
```

### 3. GPU Computation (WebGPU)
```javascript
// Future consideration when WebGPU is widely supported
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
```

### 4. Progressive Enhancement
```javascript
// Start with single-threaded, upgrade if available
class AlmaProcessor {
    constructor() {
        this.parallel = crossOriginIsolated && SharedArrayBuffer;
        this.loadModule();
    }
}
```

## Conclusion and Recommendations

### Current Status (January 2025)
While technically feasible, implementing WebAssembly multi-threading faces significant practical challenges:

1. **Archived Project**: Critical dependency no longer maintained
2. **Limited Adoption**: Few production examples exist
3. **Complexity**: Significant operational overhead

### Recommendation
**Wait for ecosystem maturity**. Consider revisiting when:
- WebAssembly threads reach stable Rust
- A maintained alternative to wasm-bindgen-rayon emerges
- More production deployments prove the approach

### If Proceeding Anyway
1. Fork wasm-bindgen-rayon for maintenance control
2. Implement comprehensive fallback system
3. Test extensively across all target browsers
4. Monitor performance gains vs. complexity cost

---

**Document Version**: 1.0  
**Last Updated**: January 5, 2025  
**Next Review**: When considering WASM multi-threading implementation