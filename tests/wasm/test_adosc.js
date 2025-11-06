/**
 * WASM binding tests for ADOSC indicator.
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

test('ADOSC with default parameters', () => {
    const { high, low, close, volume } = testData;
    const result = wasm.adosc_js(high, low, close, volume, 3, 10);
    
    // WASM returns Float64Array, not regular Array
    assert.ok(result instanceof Float64Array || Array.isArray(result));
    assert.strictEqual(result.length, close.length);
    
    // All values should be finite
    result.forEach((val, i) => {
        assert.ok(isFinite(val), `Value at index ${i} should be finite`);
    });
});

test('ADOSC matches expected values from Rust tests', () => {
    const { high, low, close, volume } = testData;
    
    const result = wasm.adosc_js(
        high, low, close, volume,
        3,  // Default short period
        10  // Default long period
    );
    
    // Check last 5 values match expected from Rust tests
    const expectedLastFive = [-166.2175, -148.9983, -144.9052, -128.5921, -142.0772];
    const last5 = Array.from(result.slice(-5));
    assertArrayClose(last5, expectedLastFive, 1e-1, 'ADOSC last 5 values mismatch');
});

test('ADOSC fails with zero period', () => {
    const high = new Float64Array([10.0, 10.0, 10.0]);
    const low = new Float64Array([5.0, 5.0, 5.0]);
    const close = new Float64Array([7.0, 7.0, 7.0]);
    const volume = new Float64Array([1000.0, 1000.0, 1000.0]);
    
    // Zero short period
    assert.throws(
        () => wasm.adosc_js(high, low, close, volume, 0, 10),
        /Invalid period/
    );
    
    // Zero long period
    assert.throws(
        () => wasm.adosc_js(high, low, close, volume, 3, 0),
        /Invalid period/
    );
});

test('ADOSC fails when short period >= long period', () => {
    const high = new Float64Array([10.0, 11.0, 12.0, 13.0, 14.0]);
    const low = new Float64Array([5.0, 5.5, 6.0, 6.5, 7.0]);
    const close = new Float64Array([7.0, 8.0, 9.0, 10.0, 11.0]);
    const volume = new Float64Array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0]);
    
    // short = long
    assert.throws(
        () => wasm.adosc_js(high, low, close, volume, 3, 3),
        /short_period must be less than long_period/
    );
    
    // short > long
    assert.throws(
        () => wasm.adosc_js(high, low, close, volume, 5, 3),
        /short_period must be less than long_period/
    );
});

test('ADOSC fails when period exceeds data length', () => {
    const high = new Float64Array([10.0, 11.0, 12.0]);
    const low = new Float64Array([5.0, 5.5, 6.0]);
    const close = new Float64Array([7.0, 8.0, 9.0]);
    const volume = new Float64Array([1000.0, 1000.0, 1000.0]);
    
    assert.throws(
        () => wasm.adosc_js(high, low, close, volume, 3, 10),
        /Invalid period/
    );
});

test('ADOSC fails with empty input', () => {
    const empty = new Float64Array([]);
    
    assert.throws(
        () => wasm.adosc_js(empty, empty, empty, empty, 3, 10),
        /empty/
    );
});

test('ADOSC handles zero volume correctly', () => {
    const high = new Float64Array([10.0, 11.0, 12.0, 13.0, 14.0]);
    const low = new Float64Array([5.0, 5.5, 6.0, 6.5, 7.0]);
    const close = new Float64Array([7.0, 8.0, 9.0, 10.0, 11.0]);
    const volume = new Float64Array([0.0, 0.0, 0.0, 0.0, 0.0]); // All zero volume
    
    const result = wasm.adosc_js(high, low, close, volume, 2, 3);
    assert.strictEqual(result.length, close.length);
    
    // With zero volume, ADOSC should be 0
    result.forEach((val, i) => {
        assert.strictEqual(val, 0, `Value at index ${i} should be 0`);
    });
});

test('ADOSC handles constant price correctly', () => {
    const price = 10.0;
    const high = new Float64Array(10).fill(price);
    const low = new Float64Array(10).fill(price);
    const close = new Float64Array(10).fill(price);
    const volume = new Float64Array(10).fill(1000.0);
    
    const result = wasm.adosc_js(high, low, close, volume, 3, 5);
    assert.strictEqual(result.length, close.length);
    
    // With constant price (high = low), MFM is 0, so ADOSC should be 0
    result.forEach((val, i) => {
        assert.strictEqual(val, 0, `Value at index ${i} should be 0`);
    });
});

test('ADOSC calculates from the first value (no warmup period)', () => {
    const { high, low, close, volume } = testData;
    const result = wasm.adosc_js(high, low, close, volume, 3, 10);
    
    // First value should not be NaN (ADOSC calculates from the start)
    assert.ok(!isNaN(result[0]), 'First ADOSC value should not be NaN');
});

test('ADOSC all NaN input', () => {
    // Test ADOSC with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.adosc_js(allNaN, allNaN, allNaN, allNaN, 3, 10);
    }, /All values are NaN/);
});

test('ADOSC batch calculation with default parameters', () => {
    const { high, low, close, volume } = testData;
    
    const result = wasm.adosc_batch_js(
        high, low, close, volume,
        3, 3, 0,   // short_period range (single value)
        10, 10, 0  // long_period range (single value)
    );
    
    // Batch returns flat array
    assert.ok(result instanceof Float64Array || Array.isArray(result));
    assert.strictEqual(result.length, close.length);
});

test('ADOSC batch calculation with multiple parameters', () => {
    const { high, low, close, volume } = testData;
    
    const result = wasm.adosc_batch_js(
        high, low, close, volume,
        2, 4, 1,   // short_period range: 2, 3, 4
        5, 7, 1    // long_period range: 5, 6, 7
    );
    
    // Should have 3 * 3 = 9 combinations
    const expected_rows = 3 * 3;
    assert.strictEqual(result.length, expected_rows * close.length);
});

test('ADOSC batch metadata', () => {
    // For short_period 2-4 step 1 and long_period 5-7 step 1
    const meta = wasm.adosc_batch_metadata_js(2, 4, 1, 5, 7, 1);
    
    assert.ok(meta instanceof Float64Array || Array.isArray(meta));
    // 3 short periods * 3 long periods = 9 combos, each has 2 values
    assert.strictEqual(meta.length, 3 * 3 * 2);
});

// Skip stream tests - WASM stream bindings not implemented for ADOSC
// test('ADOSC stream creation and update', () => {
//     const stream = wasm.adosc_stream_new(3, 10);
//     assert.ok(stream);
//     
//     // Test update with valid values
//     const val = wasm.adosc_stream_update(stream, 100.0, 90.0, 95.0, 1000.0);
//     assert.ok(isFinite(val));
//     
//     // Free the stream
//     wasm.adosc_stream_free(stream);
// });

test('ADOSC comparison with Rust', () => {
    const { high, low, close, volume } = testData;
    
    const result = wasm.adosc_js(
        high, low, close, volume,
        3,  // Default short period
        10  // Default long period
    );
    
    compareWithRust('adosc', Array.from(result), 'hlcv', { short_period: 3, long_period: 10 });
});

test('ADOSC fast API - allocation and deallocation', () => {
    const len = 100;
    const ptr = wasm.adosc_alloc(len);
    
    assert.notStrictEqual(ptr, 0, 'Allocated pointer should not be null');
    
    // Clean up
    wasm.adosc_free(ptr, len);
});

test('ADOSC fast API - basic computation', () => {
    const { high, low, close, volume } = testData;
    const len = high.length;
    
    // Allocate buffers
    const highPtr = wasm.adosc_alloc(len);
    const lowPtr = wasm.adosc_alloc(len);
    const closePtr = wasm.adosc_alloc(len);
    const volumePtr = wasm.adosc_alloc(len);
    const outPtr = wasm.adosc_alloc(len);
    
    try {
        // Create views into WASM memory (consistent with other tests)
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        
        // Copy data into WASM memory
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        volumeView.set(volume);
        
        // Compute ADOSC using fast API
        wasm.adosc_into(
            highPtr,
            lowPtr,
            closePtr,
            volumePtr,
            outPtr,
            len,
            3,  // short_period
            10  // long_period
        );
        
        // Read results from output buffer - recreate view after computation
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        // Verify results match safe API
        const safeResult = wasm.adosc_js(high, low, close, volume, 3, 10);
        
        for (let i = 0; i < len; i++) {
            assertClose(result[i], safeResult[i], 1e-10, `Fast API mismatch at index ${i}`);
        }
    } finally {
        // Always clean up
        wasm.adosc_free(highPtr, len);
        wasm.adosc_free(lowPtr, len);
        wasm.adosc_free(closePtr, len);
        wasm.adosc_free(volumePtr, len);
        wasm.adosc_free(outPtr, len);
    }
});

test('ADOSC fast API - in-place computation (aliasing)', () => {
    const { high, low, close, volume } = testData;
    const len = high.length;
    
    // Allocate buffers
    const highPtr = wasm.adosc_alloc(len);
    const lowPtr = wasm.adosc_alloc(len);
    const closePtr = wasm.adosc_alloc(len);
    const volumePtr = wasm.adosc_alloc(len);
    
    try {
        // Create views into WASM memory (consistent with other tests)
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        
        // Copy data into WASM memory
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        volumeView.set(volume);
        
        // Use close buffer as output (aliasing test)
        wasm.adosc_into(
            highPtr,
            lowPtr,
            closePtr,
            volumePtr,
            closePtr,  // Output to same buffer as close (aliasing)
            len,
            3,  // short_period
            10  // long_period
        );
        
        // Verify results match safe API
        const expected = wasm.adosc_js(high, low, close, volume, 3, 10);
        
        // closeView might be invalid after memory growth, so recreate
        const resultView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        
        for (let i = 0; i < len; i++) {
            assertClose(resultView[i], expected[i], 1e-10, `In-place computation mismatch at index ${i}`);
        }
    } finally {
        // Always clean up
        wasm.adosc_free(highPtr, len);
        wasm.adosc_free(lowPtr, len);
        wasm.adosc_free(closePtr, len);
        wasm.adosc_free(volumePtr, len);
    }
});

test('ADOSC fast API - null pointer handling', () => {
    const len = 100;
    
    // Test with null input pointers
    assert.throws(
        () => wasm.adosc_into(0, 0, 0, 0, wasm.adosc_alloc(len), len, 3, 10),
        /Null pointer/
    );
    
    // Test with null output pointer
    const dummyPtr = wasm.adosc_alloc(len);
    assert.throws(
        () => wasm.adosc_into(dummyPtr, dummyPtr, dummyPtr, dummyPtr, 0, len, 3, 10),
        /Null pointer/
    );
    wasm.adosc_free(dummyPtr, len);
});

// New ergonomic batch API tests
test('ADOSC batch - ergonomic API with single parameter', () => {
    const { high, low, close, volume } = testData;
    
    const result = wasm.adosc_batch(high, low, close, volume, {
        short_period_range: [3, 3, 0],
        long_period_range: [10, 10, 0]
    });
    
    // Verify structure
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');
    
    // Verify dimensions
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.values.length, close.length);
    
    // Verify parameters
    const combo = result.combos[0];
    assert.strictEqual(combo.short_period, 3);
    assert.strictEqual(combo.long_period, 10);
    
    // Compare with old API
    const oldResult = wasm.adosc_js(high, low, close, volume, 3, 10);
    for (let i = 0; i < oldResult.length; i++) {
        assertClose(oldResult[i], result.values[i], 1e-10,
                   `Value mismatch at index ${i}`);
    }
});

test('ADOSC batch - ergonomic API with multiple parameters', () => {
    const { high, low, close, volume } = testData;
    const testSlice = high.slice(0, 50); // Use smaller dataset
    
    const result = wasm.adosc_batch(
        high.slice(0, 50),
        low.slice(0, 50),
        close.slice(0, 50),
        volume.slice(0, 50),
        {
            short_period_range: [2, 4, 1],   // 2, 3, 4
            long_period_range: [5, 7, 1]     // 5, 6, 7
        }
    );
    
    // Should have 3 * 3 = 9 combinations
    assert.strictEqual(result.rows, 9);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 9);
    assert.strictEqual(result.values.length, 450);
    
    // Verify each combo
    const expectedCombos = [];
    for (let s = 2; s <= 4; s++) {
        for (let l = 5; l <= 7; l++) {
            expectedCombos.push({ short_period: s, long_period: l });
        }
    }
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].short_period, expectedCombos[i].short_period);
        assert.strictEqual(result.combos[i].long_period, expectedCombos[i].long_period);
    }
    
    // Compare first row with single calculation
    const oldResult = wasm.adosc_js(
        high.slice(0, 50),
        low.slice(0, 50),
        close.slice(0, 50),
        volume.slice(0, 50),
        2, 5
    );
    const firstRow = result.values.slice(0, result.cols);
    for (let i = 0; i < oldResult.length; i++) {
        assertClose(oldResult[i], firstRow[i], 1e-10,
                   `Value mismatch at index ${i}`);
    }
});

test('ADOSC batch - error handling', () => {
    const { high, low, close, volume } = testData;
    
    // Invalid config structure
    assert.throws(() => {
        wasm.adosc_batch(high, low, close, volume, {
            short_period_range: [3, 3], // Missing step
            long_period_range: [10, 10, 0]
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.adosc_batch(high, low, close, volume, {
            short_period_range: [3, 3, 0]
            // Missing long_period_range
        });
    }, /Invalid config/);
    
    // Invalid data type
    assert.throws(() => {
        wasm.adosc_batch(high, low, close, volume, {
            short_period_range: "invalid",
            long_period_range: [10, 10, 0]
        });
    }, /Invalid config/);
});

test('ADOSC batch - edge cases', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep
    const singleBatch = wasm.adosc_batch(data, data, data, data, {
        short_period_range: [2, 2, 1],
        long_period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.adosc_batch(data, data, data, data, {
        short_period_range: [2, 3, 10], // Step larger than range
        long_period_range: [5, 5, 0]
    });
    
    // Should only have short_period=2
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.adosc_batch(new Float64Array([]), new Float64Array([]), 
                        new Float64Array([]), new Float64Array([]), {
            short_period_range: [3, 3, 0],
            long_period_range: [10, 10, 0]
        });
    }, /empty/);
});

test('ADOSC batch into - fast API for batch computation', () => {
    const { high, low, close, volume } = testData;
    const len = high.length;
    
    // Calculate expected output size
    const shortPeriods = 3; // 2, 3, 4
    const longPeriods = 2;  // 5, 6
    const rows = shortPeriods * longPeriods;
    const totalSize = rows * len;
    
    // Allocate all buffers
    const highPtr = wasm.adosc_alloc(len);
    const lowPtr = wasm.adosc_alloc(len);
    const closePtr = wasm.adosc_alloc(len);
    const volumePtr = wasm.adosc_alloc(len);
    const outPtr = wasm.adosc_alloc(totalSize);
    
    try {
        // Copy data into WASM memory
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        volumeView.set(volume);
        
        // Compute batch
        const numRows = wasm.adosc_batch_into(
            highPtr, lowPtr, closePtr, volumePtr,
            outPtr, len,
            2, 4, 1,  // short_period range
            5, 6, 1   // long_period range
        );
        
        assert.strictEqual(numRows, rows);
        
        // Read results
        const results = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
        
        // Verify first row matches single calculation
        const expected = wasm.adosc_js(high, low, close, volume, 2, 5);
        for (let i = 0; i < len; i++) {
            assertClose(results[i], expected[i], 1e-10, `Batch into mismatch at index ${i}`);
        }
    } finally {
        wasm.adosc_free(highPtr, len);
        wasm.adosc_free(lowPtr, len);
        wasm.adosc_free(closePtr, len);
        wasm.adosc_free(volumePtr, len);
        wasm.adosc_free(outPtr, totalSize);
    }
});
