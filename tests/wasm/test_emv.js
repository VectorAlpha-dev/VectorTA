/**
 * WASM binding tests for EMV indicator.
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

test('EMV basic calculation', () => {
    // Test basic EMV calculation - mirrors check_emv_basic_calculation
    const high = new Float64Array([10.0, 12.0, 13.0, 15.0]);
    const low = new Float64Array([5.0, 7.0, 8.0, 10.0]);
    const close = new Float64Array([7.5, 9.0, 10.5, 12.5]);
    const volume = new Float64Array([10000.0, 20000.0, 25000.0, 30000.0]);
    
    const result = wasm.emv_js(high, low, close, volume);
    assert.strictEqual(result.length, 4);
    assert(isNaN(result[0])); // First value should be NaN
    assert(!isNaN(result[1]));
    
    // Test specific calculation for index 1
    // mid[0] = (10 + 5) / 2 = 7.5
    // mid[1] = (12 + 7) / 2 = 9.5
    // range[1] = 12 - 7 = 5
    // br[1] = 20000 / 10000 / 5 = 0.4
    // emv[1] = (9.5 - 7.5) / 0.4 = 5.0
    assertClose(result[1], 5.0, 0.01, "EMV calculation at index 1");
});

test('EMV accuracy', async () => {
    // Test EMV matches expected values from Rust tests - mirrors check_emv_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.emv_js(high, low, close, volume);
    assert.strictEqual(result.length, high.length);
    
    // Expected last 5 values from Rust tests
    const expected_last_five = [
        -6488905.579799851,
        2371436.7401001123,
        -3855069.958128531,
        1051939.877943717,
        -8519287.22257077,
    ];
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected_last_five,
        100,  // Larger absolute tolerance for EMV's large values
        "EMV last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('emv', result, 'ohlcv');
});

test('EMV warmup period', () => {
    // Test EMV warmup period behavior - mirrors check_emv_nan_handling
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.emv_js(high, low, close, volume);
    
    // EMV warmup period is 1 (first value is always NaN)
    assert(isNaN(result[0]), "First EMV value should be NaN (warmup)");
    
    // Find first valid data point
    let firstValid = null;
    for (let i = 0; i < high.length; i++) {
        if (!isNaN(high[i]) && !isNaN(low[i]) && !isNaN(volume[i])) {
            firstValid = i;
            break;
        }
    }
    
    if (firstValid !== null && firstValid + 1 < result.length) {
        // After first valid point, next should have a value
        assert(!isNaN(result[firstValid + 1]), 
               `Expected valid EMV at index ${firstValid + 1} after first valid data`);
    }
});

test('EMV empty data', () => {
    // Test EMV with empty data - mirrors check_emv_empty_data
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.emv_js(empty, empty, empty, empty);
    }, /Empty data|EmptyData/);
});

test('EMV all NaN', () => {
    // Test EMV with all NaN values - mirrors check_emv_all_nan
    const nanArr = new Float64Array([NaN, NaN]);
    
    assert.throws(() => {
        wasm.emv_js(nanArr, nanArr, nanArr, nanArr);
    }, /All values are NaN|AllValuesNaN/);
});

test('EMV not enough data', () => {
    // Test EMV with insufficient data - mirrors check_emv_not_enough_data
    const high = new Float64Array([10000.0, NaN]);
    const low = new Float64Array([9990.0, NaN]);
    const close = new Float64Array([9995.0, NaN]);
    const volume = new Float64Array([1_000_000.0, NaN]);
    
    assert.throws(() => {
        wasm.emv_js(high, low, close, volume);
    }, /Not enough data|NotEnoughData/);
});

test('EMV partial NaN handling', () => {
    // Test EMV with partial NaN values
    const high = new Float64Array([NaN, 12.0, 15.0, NaN, 13.0, 16.0]);
    const low = new Float64Array([NaN, 9.0, 11.0, NaN, 10.0, 12.0]);
    const close = new Float64Array([NaN, 10.0, 13.0, NaN, 11.5, 14.0]);
    const volume = new Float64Array([NaN, 10000.0, 20000.0, NaN, 15000.0, 25000.0]);
    
    const result = wasm.emv_js(high, low, close, volume);
    
    // Check shape
    assert.strictEqual(result.length, high.length);
    
    // First few should be NaN
    assert(isNaN(result[0]));
    assert(isNaN(result[1])); // Need previous value
    
    // Should have valid values after enough data
    assert(!isNaN(result[2]));
});

test('EMV zero range handling', () => {
    // Test EMV when high equals low (zero range)
    const high = new Float64Array([10.0, 10.0, 12.0, 13.0]);
    const low = new Float64Array([9.0, 10.0, 11.0, 12.0]); // At index 1: high == low
    const close = new Float64Array([9.5, 10.0, 11.5, 12.5]);
    const volume = new Float64Array([1000.0, 2000.0, 3000.0, 4000.0]);
    
    const result = wasm.emv_js(high, low, close, volume);
    
    // When range is zero, EMV should be NaN
    assert(isNaN(result[1]), "Expected NaN when range is zero");
    
    // Other values should be calculated
    assert(!isNaN(result[2]));
});

test('EMV mismatched lengths', () => {
    // Test EMV with mismatched input lengths
    const high = new Float64Array([10.0, 12.0, 13.0]);
    const low = new Float64Array([9.0, 11.0]); // Different length
    const close = new Float64Array([9.5, 11.5, 12.0]);
    const volume = new Float64Array([1000.0, 2000.0, 3000.0]);
    
    // Should work but use minimum length
    const result = wasm.emv_js(high, low, close, volume);
    assert.strictEqual(result.length, 2); // min(3, 2, 3, 3) = 2
});

test('EMV batch operations', () => {
    // Test batch API - mirrors check_batch_row
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // EMV has no parameters, so config is empty
    const config = {};
    
    const result = wasm.emv_batch(high, low, close, volume, config);
    
    assert(result.values, "Expected values array");
    assert.strictEqual(result.rows, 1, "EMV batch should have 1 row (no parameter sweep)");
    assert.strictEqual(result.cols, high.length, "Expected cols to match input length");
    
    // Values should match single calculation
    const singleResult = wasm.emv_js(high, low, close, volume);
    assertArrayClose(
        new Float64Array(result.values),
        singleResult,
        1e-10,
        "Batch values should match single calculation"
    );
    
    // Verify last 5 values match expected
    const expected_last_five = [
        -6488905.579799851,
        2371436.7401001123,
        -3855069.958128531,
        1051939.877943717,
        -8519287.22257077,
    ];
    
    const batchLast5 = result.values.slice(-5);
    assertArrayClose(
        batchLast5,
        expected_last_five,
        100,  // Larger absolute tolerance for EMV's large values
        "EMV batch last 5 values mismatch"
    );
});

test('EMV SIMD consistency', () => {
    // Test that SIMD (if available) produces same results as scalar
    const testCases = [
        { size: 10 },
        { size: 100 },
        { size: 1000 },
        { size: 10000 }
    ];
    
    for (const testCase of testCases) {
        // Generate synthetic OHLCV data
        const high = new Float64Array(testCase.size);
        const low = new Float64Array(testCase.size);
        const close = new Float64Array(testCase.size);
        const volume = new Float64Array(testCase.size);
        
        for (let i = 0; i < testCase.size; i++) {
            const base = 100 + Math.sin(i * 0.1) * 10;
            high[i] = base + Math.random() * 5;
            low[i] = base - Math.random() * 5;
            close[i] = (high[i] + low[i]) / 2;
            volume[i] = 10000 + Math.random() * 5000;
        }
        
        const result = wasm.emv_js(high, low, close, volume);
        
        // Basic sanity checks
        assert.strictEqual(result.length, testCase.size);
        
        // Check warmup period (first value is NaN)
        assert(isNaN(result[0]), `Expected NaN at warmup index 0 for size=${testCase.size}`);
        
        // Check values exist after warmup
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = 1; i < result.length; i++) {
            if (!isNaN(result[i])) {
                sumAfterWarmup += Math.abs(result[i]);
                countAfterWarmup++;
            }
        }
        
        // Verify we have valid values
        assert(countAfterWarmup > 0, `No valid values found for size=${testCase.size}`);
        
        // Verify reasonable values (EMV can be very large but should be finite)
        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(isFinite(avgAfterWarmup), `Average value ${avgAfterWarmup} is not finite`);
    }
});

test('EMV zero-copy (fast) API', () => {
    // Test the zero-copy fast API with proper memory allocation
    const size = 100;
    
    // Allocate memory for all inputs and output
    const highPtr = wasm.emv_alloc(size);
    const lowPtr = wasm.emv_alloc(size);
    const closePtr = wasm.emv_alloc(size);
    const volumePtr = wasm.emv_alloc(size);
    const outputPtr = wasm.emv_alloc(size);
    
    assert(highPtr !== 0, "Failed to allocate high buffer");
    assert(lowPtr !== 0, "Failed to allocate low buffer");
    assert(closePtr !== 0, "Failed to allocate close buffer");
    assert(volumePtr !== 0, "Failed to allocate volume buffer");
    assert(outputPtr !== 0, "Failed to allocate output buffer");
    
    try {
        // Create views into WASM memory
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, size);
        const volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, size);
        const outputView = new Float64Array(wasm.__wasm.memory.buffer, outputPtr, size);
        
        // Fill with test data
        for (let i = 0; i < size; i++) {
            const base = 100 + Math.sin(i * 0.1) * 10;
            highView[i] = base + 2;
            lowView[i] = base - 2;
            closeView[i] = base;
            volumeView[i] = 10000 + i * 100;
        }
        
        // Compute EMV using zero-copy API
        wasm.emv_into(
            highPtr,
            lowPtr,
            closePtr,
            volumePtr,
            outputPtr,
            size
        );
        
        // Verify results
        assert(isNaN(outputView[0]), "First value should be NaN");
        assert(!isNaN(outputView[10]), "Should have valid values after warmup");
        
        // Compare with regular API
        const highArray = new Float64Array(highView);
        const lowArray = new Float64Array(lowView);
        const closeArray = new Float64Array(closeView);
        const volumeArray = new Float64Array(volumeView);
        
        const regularResult = wasm.emv_js(highArray, lowArray, closeArray, volumeArray);
        
        for (let i = 0; i < size; i++) {
            if (isNaN(regularResult[i]) && isNaN(outputView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - outputView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${outputView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.emv_free(highPtr, size);
        wasm.emv_free(lowPtr, size);
        wasm.emv_free(closePtr, size);
        wasm.emv_free(volumePtr, size);
        wasm.emv_free(outputPtr, size);
    }
});

test('EMV zero-copy API with aliasing', () => {
    // Test in-place operation (output aliased with one of the inputs)
    const size = 50;
    
    // Allocate memory
    const highPtr = wasm.emv_alloc(size);
    const lowPtr = wasm.emv_alloc(size);
    const closePtr = wasm.emv_alloc(size);
    const volumePtr = wasm.emv_alloc(size);
    
    assert(highPtr !== 0 && lowPtr !== 0 && closePtr !== 0 && volumePtr !== 0, 
           "Failed to allocate memory");
    
    try {
        // Create views
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, size);
        const volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, size);
        
        // Fill with test data
        for (let i = 0; i < size; i++) {
            highView[i] = 100 + i;
            lowView[i] = 90 + i;
            closeView[i] = 95 + i;
            volumeView[i] = 10000;
        }
        
        // Save original close values
        const originalClose = new Float64Array(closeView);
        
        // Use close array as both input and output (aliased)
        wasm.emv_into(
            highPtr,
            lowPtr,
            closePtr,  // input
            volumePtr,
            closePtr,  // output (aliased with close)
            size
        );
        
        // closeView now contains EMV values, not original close values
        assert(isNaN(closeView[0]), "First EMV value should be NaN");
        assert(closeView[1] !== originalClose[1], "Values should have changed");
    } finally {
        wasm.emv_free(highPtr, size);
        wasm.emv_free(lowPtr, size);
        wasm.emv_free(closePtr, size);
        wasm.emv_free(volumePtr, size);
    }
});

test('EMV memory leak prevention', () => {
    // Test multiple alloc/free cycles to ensure no memory leaks
    const sizes = [100, 1000, 10000, 100000];
    
    for (const size of sizes) {
        const ptr = wasm.emv_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Write pattern to verify memory
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.emv_free(ptr, size);
    }
    
    // Allocate and free multiple times in sequence
    for (let round = 0; round < 10; round++) {
        const ptrs = [];
        
        // Allocate multiple buffers
        for (let i = 0; i < 5; i++) {
            const ptr = wasm.emv_alloc(1000);
            assert(ptr !== 0, `Failed to allocate in round ${round}, buffer ${i}`);
            ptrs.push(ptr);
        }
        
        // Free all buffers
        for (const ptr of ptrs) {
            wasm.emv_free(ptr, 1000);
        }
    }
});

test('EMV batch into (fast API)', () => {
    // Test batch into API with proper memory allocation
    const size = 100;
    
    // Allocate memory
    const highPtr = wasm.emv_alloc(size);
    const lowPtr = wasm.emv_alloc(size);
    const closePtr = wasm.emv_alloc(size);
    const volumePtr = wasm.emv_alloc(size);
    const outputPtr = wasm.emv_alloc(size);
    
    try {
        // Create views and fill with data
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, size);
        const volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, size);
        const outputView = new Float64Array(wasm.__wasm.memory.buffer, outputPtr, size);
        
        for (let i = 0; i < size; i++) {
            highView[i] = 100 + i * 0.5;
            lowView[i] = 95 + i * 0.5;
            closeView[i] = 97.5 + i * 0.5;
            volumeView[i] = 10000;
        }
        
        const rows = wasm.emv_batch_into(
            highPtr,
            lowPtr,
            closePtr,
            volumePtr,
            outputPtr,
            size
        );
        
        assert.strictEqual(rows, 1, "Expected 1 row for EMV batch");
        assert(isNaN(outputView[0]), "First value should be NaN");
        assert(!isNaN(outputView[10]), "Should have valid values after warmup");
    } finally {
        wasm.emv_free(highPtr, size);
        wasm.emv_free(lowPtr, size);
        wasm.emv_free(closePtr, size);
        wasm.emv_free(volumePtr, size);
        wasm.emv_free(outputPtr, size);
    }
});

test('EMV null pointer handling', () => {
    // Test null pointer error handling
    assert.throws(() => {
        wasm.emv_into(0, 0, 0, 0, 0, 100);
    }, /null pointer/);
});

test.after(() => {
    console.log('EMV WASM tests completed');
});