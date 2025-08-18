/**
 * WASM binding tests for BOP indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { 
    loadTestData, 
    assertArrayClose, 
    assertClose,
    isNaN,
    assertAllNaN,
    assertNoNaN,
    EXPECTED_OUTPUTS 
} from './test_utils.js';
import { compareWithRust } from './rust-comparison.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
    // Load WASM module
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('BOP partial params', () => {
    // Test with standard parameters - mirrors check_bop_partial_params
    const open = new Float64Array(testData.open);
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // BOP has no parameters, just OHLC inputs
    const result = wasm.bop_js(open, high, low, close);
    assert.strictEqual(result.length, close.length);
});

test('BOP accuracy', async () => {
    // Test BOP matches expected values from Rust tests - mirrors check_bop_accuracy
    const open = new Float64Array(testData.open);
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.bop;
    
    const result = wasm.bop_js(open, high, low, close);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-10,
        "BOP last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('bop', result, 'ohlc', expected.defaultParams);
});

test('BOP default candles', () => {
    // Test BOP with default parameters - mirrors check_bop_default_candles
    const open = new Float64Array(testData.open);
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.bop_js(open, high, low, close);
    assert.strictEqual(result.length, close.length);
});

test('BOP with empty data', () => {
    // Test BOP fails with empty data - mirrors check_bop_with_empty_data
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.bop_js(empty, empty, empty, empty);
    }, /Data is empty/);
});

test('BOP with inconsistent lengths', () => {
    // Test BOP fails with inconsistent input lengths - mirrors check_bop_with_inconsistent_lengths
    const open = new Float64Array([1.0, 2.0, 3.0]);
    const high = new Float64Array([1.5, 2.5]);  // Wrong length
    const low = new Float64Array([0.8, 1.8, 2.8]);
    const close = new Float64Array([1.2, 2.2, 3.2]);
    
    assert.throws(() => {
        wasm.bop_js(open, high, low, close);
    }, /Input lengths mismatch/);
});

test('BOP very small dataset', () => {
    // Test BOP with single data point - mirrors check_bop_very_small_dataset
    const open = new Float64Array([10.0]);
    const high = new Float64Array([12.0]);
    const low = new Float64Array([9.5]);
    const close = new Float64Array([11.0]);
    
    const result = wasm.bop_js(open, high, low, close);
    assert.strictEqual(result.length, 1);
    // (11.0 - 10.0) / (12.0 - 9.5) = 1.0 / 2.5 = 0.4
    assertClose(result[0], 0.4, 1e-10, "BOP single value calculation");
});

test('BOP with slice data reinput', () => {
    // Test BOP with slice data re-input - mirrors check_bop_with_slice_data_reinput
    const open = new Float64Array(testData.open);
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.bop_js(open, high, low, close);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - use first result as close, zeros for others
    const dummy = new Float64Array(firstResult.length);
    const secondResult = wasm.bop_js(dummy, dummy, dummy, firstResult);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // All values should be 0.0 since (first_result - 0) / (0 - 0) = 0.0
    for (let i = 0; i < secondResult.length; i++) {
        assertClose(secondResult[i], 0.0, 1e-15, 
                   `Expected BOP=0.0 for dummy data at idx ${i}`);
    }
});

test('BOP NaN handling', () => {
    // Test BOP handles values correctly without NaN - mirrors check_bop_nan_handling
    const open = new Float64Array(testData.open);
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.bop_js(open, high, low, close);
    assert.strictEqual(result.length, close.length);
    
    // BOP should not produce NaN values after any warmup period
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // Actually, BOP has no warmup period - it calculates from the first value
    assertNoNaN(result, "BOP should not produce any NaN values");
});

test('BOP zero range handling', () => {
    // Test BOP when high equals low (zero range)
    // When high == low, BOP should return 0.0
    const open = new Float64Array([10.0, 20.0, 30.0]);
    const high = new Float64Array([15.0, 25.0, 35.0]);
    const low = new Float64Array([15.0, 25.0, 35.0]);  // Same as high
    const close = new Float64Array([15.0, 25.0, 35.0]);
    
    const result = wasm.bop_js(open, high, low, close);
    
    // All values should be 0.0 since denominator is 0
    for (let i = 0; i < result.length; i++) {
        assertClose(result[i], 0.0, 1e-15, 
                   `Expected BOP=0.0 when high=low at idx ${i}`);
    }
});

test('BOP batch single', () => {
    // Test batch processing with BOP (no parameters)
    const open = new Float64Array(testData.open);
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // BOP has no parameters, so batch is just regular calculation
    const batchResult = wasm.bop_batch_js(open, high, low, close);
    
    // Should match single calculation
    const singleResult = wasm.bop_js(open, high, low, close);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('BOP batch metadata', () => {
    // Test metadata function returns empty array (no parameters)
    const metadata = wasm.bop_batch_metadata_js();
    
    // BOP has no parameters, so metadata should be empty
    assert.strictEqual(metadata.length, 0);
});

test('BOP batch unified API', () => {
    // Test unified batch API with config object
    const open = new Float64Array(testData.open.slice(0, 100));
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // BOP has no parameters, but we pass empty config for API consistency
    const config = {};  // Empty config for BOP
    
    const result = wasm.bop_batch(open, high, low, close, config);
    
    // Should have structure with values, rows, cols
    assert(result.values);
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, 100);
    assert.strictEqual(result.values.length, 100);
    
    // Values should match regular calculation
    const regularResult = wasm.bop_js(open, high, low, close);
    assertArrayClose(result.values, regularResult, 1e-10, "Unified API mismatch");
});

test('BOP extreme values', () => {
    // Test BOP with extreme price movements
    const open = new Float64Array([100.0, 1000.0, 10.0]);
    const high = new Float64Array([200.0, 2000.0, 20.0]);
    const low = new Float64Array([50.0, 500.0, 5.0]);
    const close = new Float64Array([150.0, 1500.0, 15.0]);
    
    const result = wasm.bop_js(open, high, low, close);
    
    // Calculate expected values manually
    // BOP = (close - open) / (high - low)
    const expected = [
        (150.0 - 100.0) / (200.0 - 50.0),   // 50/150 = 0.333...
        (1500.0 - 1000.0) / (2000.0 - 500.0), // 500/1500 = 0.333...
        (15.0 - 10.0) / (20.0 - 5.0)         // 5/15 = 0.333...
    ];
    
    assertArrayClose(result, expected, 1e-10, "Extreme values mismatch");
});

test('BOP negative values', () => {
    // Test BOP when close < open (negative BOP)
    const open = new Float64Array([100.0, 200.0, 300.0]);
    const high = new Float64Array([110.0, 210.0, 310.0]);
    const low = new Float64Array([80.0, 180.0, 280.0]);
    const close = new Float64Array([90.0, 190.0, 290.0]);
    
    const result = wasm.bop_js(open, high, low, close);
    
    // All should be negative: (90-100)/(110-80) = -10/30 = -0.333...
    for (let i = 0; i < result.length; i++) {
        assert(result[i] < 0, `Expected negative BOP at index ${i}`);
        assertClose(result[i], -1/3, 1e-10, `BOP value at index ${i}`);
    }
});

// Fast API tests
test('BOP zero-copy API (bop_into)', () => {
    const data = new Float64Array([
        10.0, 20.0, 30.0, 40.0, 50.0,
        11.0, 21.0, 31.0, 41.0, 51.0
    ]);
    const high = new Float64Array([
        12.0, 22.0, 32.0, 42.0, 52.0,
        13.0, 23.0, 33.0, 43.0, 53.0
    ]);
    const low = new Float64Array([
        9.0, 19.0, 29.0, 39.0, 49.0,
        10.0, 20.0, 30.0, 40.0, 50.0
    ]);
    const close = new Float64Array([
        11.5, 21.5, 31.5, 41.5, 51.5,
        12.5, 22.5, 32.5, 42.5, 52.5
    ]);
    
    // Allocate buffers
    const openPtr = wasm.bop_alloc(data.length);
    const highPtr = wasm.bop_alloc(data.length);
    const lowPtr = wasm.bop_alloc(data.length);
    const closePtr = wasm.bop_alloc(data.length);
    const outPtr = wasm.bop_alloc(data.length);
    
    assert(openPtr !== 0, 'Failed to allocate open buffer');
    assert(highPtr !== 0, 'Failed to allocate high buffer');
    assert(lowPtr !== 0, 'Failed to allocate low buffer');
    assert(closePtr !== 0, 'Failed to allocate close buffer');
    assert(outPtr !== 0, 'Failed to allocate output buffer');
    
    try {
        // Create views into WASM memory
        const openView = new Float64Array(wasm.__wasm.memory.buffer, openPtr, data.length);
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, data.length);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, data.length);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, data.length);
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, data.length);
        
        // Copy data into WASM memory
        openView.set(data);
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        
        // Compute BOP using fast API
        wasm.bop_into(openPtr, highPtr, lowPtr, closePtr, outPtr, data.length);
        
        // Verify results match regular API
        const regularResult = wasm.bop_js(data, high, low, close);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(outView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - outView[i]) < 1e-10,
                   `Fast API mismatch at index ${i}: regular=${regularResult[i]}, fast=${outView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.bop_free(openPtr, data.length);
        wasm.bop_free(highPtr, data.length);
        wasm.bop_free(lowPtr, data.length);
        wasm.bop_free(closePtr, data.length);
        wasm.bop_free(outPtr, data.length);
    }
});

test('BOP in-place computation with aliasing', () => {
    const data = new Float64Array([
        10.0, 20.0, 30.0, 40.0, 50.0
    ]);
    const high = new Float64Array([
        12.0, 22.0, 32.0, 42.0, 52.0
    ]);
    const low = new Float64Array([
        9.0, 19.0, 29.0, 39.0, 49.0
    ]);
    const close = new Float64Array([
        11.5, 21.5, 31.5, 41.5, 51.5
    ]);
    
    // Test aliasing with close buffer
    const closePtr = wasm.bop_alloc(data.length);
    const highPtr = wasm.bop_alloc(data.length);
    const lowPtr = wasm.bop_alloc(data.length);
    const openPtr = wasm.bop_alloc(data.length);
    
    try {
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, data.length);
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, data.length);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, data.length);
        const openView = new Float64Array(wasm.__wasm.memory.buffer, openPtr, data.length);
        
        // Copy data
        closeView.set(close);
        highView.set(high);
        lowView.set(low);
        openView.set(data);
        
        // Store original close values for comparison
        const originalClose = Array.from(close);
        
        // Compute BOP in-place (output overwrites close)
        wasm.bop_into(openPtr, highPtr, lowPtr, closePtr, closePtr, data.length);
        
        // Recreate view in case memory moved
        const resultView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, data.length);
        
        // Verify results match regular API
        const regularResult = wasm.bop_js(data, high, low, originalClose);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(resultView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - resultView[i]) < 1e-10,
                   `Aliasing mismatch at index ${i}: regular=${regularResult[i]}, aliased=${resultView[i]}`);
        }
    } finally {
        wasm.bop_free(closePtr, data.length);
        wasm.bop_free(highPtr, data.length);
        wasm.bop_free(lowPtr, data.length);
        wasm.bop_free(openPtr, data.length);
    }
});

test('BOP memory management', () => {
    // Test multiple allocations and frees
    const sizes = [10, 100, 1000, 10000];
    
    for (const size of sizes) {
        const ptrs = [];
        
        // Allocate 4 buffers (open, high, low, close)
        for (let i = 0; i < 4; i++) {
            const ptr = wasm.bop_alloc(size);
            assert(ptr !== 0, `Failed to allocate buffer ${i} of size ${size}`);
            ptrs.push(ptr);
        }
        
        // Write test pattern to verify memory integrity
        for (let i = 0; i < ptrs.length; i++) {
            const view = new Float64Array(wasm.__wasm.memory.buffer, ptrs[i], size);
            for (let j = 0; j < Math.min(10, size); j++) {
                view[j] = i * 100 + j;
            }
        }
        
        // Verify patterns
        for (let i = 0; i < ptrs.length; i++) {
            const view = new Float64Array(wasm.__wasm.memory.buffer, ptrs[i], size);
            for (let j = 0; j < Math.min(10, size); j++) {
                assert.strictEqual(view[j], i * 100 + j, 
                    `Memory corruption at buffer ${i}, index ${j}`);
            }
        }
        
        // Free all buffers
        for (const ptr of ptrs) {
            wasm.bop_free(ptr, size);
        }
    }
});

test('BOP fast API error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.bop_into(0, 0, 0, 0, 0, 10);
    }, /Null pointer/i);
    
    // Test with one valid pointer
    const ptr = wasm.bop_alloc(10);
    try {
        assert.throws(() => {
            wasm.bop_into(ptr, 0, 0, 0, ptr, 10);
        }, /Null pointer/i);
    } finally {
        wasm.bop_free(ptr, 10);
    }
});

test('BOP large dataset performance', () => {
    const size = 100000;
    const open = new Float64Array(size);
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);
    
    // Generate test data
    for (let i = 0; i < size; i++) {
        const base = 100 + Math.sin(i * 0.01) * 50;
        low[i] = base - Math.random() * 5;
        high[i] = base + Math.random() * 5;
        open[i] = low[i] + Math.random() * (high[i] - low[i]);
        close[i] = low[i] + Math.random() * (high[i] - low[i]);
    }
    
    // Allocate large buffers
    const openPtr = wasm.bop_alloc(size);
    const highPtr = wasm.bop_alloc(size);
    const lowPtr = wasm.bop_alloc(size);
    const closePtr = wasm.bop_alloc(size);
    const outPtr = wasm.bop_alloc(size);
    
    assert(openPtr !== 0 && highPtr !== 0 && lowPtr !== 0 && closePtr !== 0 && outPtr !== 0, 
           'Failed to allocate large buffers');
    
    try {
        // Create views
        const openView = new Float64Array(wasm.__wasm.memory.buffer, openPtr, size);
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, size);
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, size);
        
        // Copy data
        openView.set(open);
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        
        // Time the fast API
        const startFast = performance.now();
        wasm.bop_into(openPtr, highPtr, lowPtr, closePtr, outPtr, size);
        const timeFast = performance.now() - startFast;
        
        // Time the regular API
        const startRegular = performance.now();
        const regularResult = wasm.bop_js(open, high, low, close);
        const timeRegular = performance.now() - startRegular;
        
        console.log(`Large dataset (${size} elements):`);
        console.log(`  Fast API: ${timeFast.toFixed(2)}ms`);
        console.log(`  Regular API: ${timeRegular.toFixed(2)}ms`);
        console.log(`  Speedup: ${(timeRegular / timeFast).toFixed(2)}x`);
        
        // Recreate the view after potential memory growth from regular API call
        const outViewFinal = new Float64Array(wasm.__wasm.memory.buffer, outPtr, size);
        
        // Verify first few values match
        for (let i = 0; i < 100; i++) {
            if (isNaN(regularResult[i]) && isNaN(outViewFinal[i])) {
                continue;
            }
            assert(Math.abs(regularResult[i] - outViewFinal[i]) < 1e-10,
                   `Mismatch at index ${i}`);
        }
    } finally {
        wasm.bop_free(openPtr, size);
        wasm.bop_free(highPtr, size);
        wasm.bop_free(lowPtr, size);
        wasm.bop_free(closePtr, size);
        wasm.bop_free(outPtr, size);
    }
});

test.after(() => {
    console.log('BOP WASM tests completed');
});

if (process.argv.includes('--run')) {
    // This allows running the file directly with node
    test.run();
}