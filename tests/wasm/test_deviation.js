/**
 * WASM binding tests for DEVIATION indicator.
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
        // Prefer the pkg ESM wrapper first
        wasm = await import(importPath);
    } catch (error) {
        // Fallback: some Node/wasm-pack combos try to import `env` from the .wasm module
        // which fails under --experimental-wasm-modules. In that case, fall back to the
        // CommonJS test wrapper that wires imports explicitly.
        try {
            const { createRequire } = await import('node:module');
            const require = createRequire(import.meta.url);
            // Local CJS wrapper produced for the test harness (.cjs to avoid ESM coercion)
            // eslint-disable-next-line unicorn/prefer-module
            wasm = require(path.join(__dirname, 'my_project.cjs'));
        } catch (fallbackErr) {
            console.error('Failed to load WASM module via pkg and local wrapper.');
            console.error('Hint: run `wasm-pack build --features wasm --target nodejs` and ensure Node >=18.');
            throw fallbackErr;
        }
    }

    testData = loadTestData();
});

test('Deviation basic functionality', () => {
    const close = new Float64Array(testData.close);
    
    // Test with period=20, devtype=0 (simple standard deviation)
    const result = wasm.deviation_js(close, 20, 0);
    assert(result, 'Should have result');
    assert.strictEqual(result.length, close.length, 'Result length should match input');
    
    // First 19 values should be NaN (warmup period)
    for (let i = 0; i < 19; i++) {
        assert(isNaN(result[i]), `Value at index ${i} should be NaN during warmup`);
    }
    
    // After warmup, should have valid values
    for (let i = 19; i < result.length; i++) {
        assert(!isNaN(result[i]), `Value at index ${i} should not be NaN after warmup`);
        assert(result[i] >= 0, `Standard deviation at index ${i} should be non-negative`);
    }
});

test('Deviation accuracy (last 5 match references)', () => {
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.deviation;
    const { period, devtype } = expected.defaultParams;
    const result = wasm.deviation_js(close, period, devtype);
    assert.strictEqual(result.length, close.length);
    const last5 = Array.from(result.slice(-5));
    assertArrayClose(last5, expected.last5Values, 2e-8, 'Deviation last 5 values mismatch');
});

test('Deviation different types', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Test all deviation types
    const types = [
        { type: 0, name: 'Simple' },
        { type: 1, name: 'MeanAbsolute' },
        { type: 2, name: 'Median' },
        { type: 3, name: 'Mode' }
    ];
    
    for (const { type, name } of types) {
        const result = wasm.deviation_js(data, 5, type);
        assert(result, `Should have result for ${name} deviation`);
        assert.strictEqual(result.length, data.length, `Result length should match input for ${name}`);
        
        // Check warmup period
        for (let i = 0; i < 4; i++) {
            assert(isNaN(result[i]), `${name}: Value at index ${i} should be NaN during warmup`);
        }
        
        // Check valid values exist after warmup
        for (let i = 4; i < result.length; i++) {
            assert(!isNaN(result[i]), `${name}: Value at index ${i} should not be NaN after warmup`);
        }
    }
});

test('Deviation error handling', () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    
    // Period > data length
    assert.throws(() => {
        wasm.deviation_js(data, 10, 0);
    }, /NotEnoughData|period/);
    
    // Invalid period (0)
    assert.throws(() => {
        wasm.deviation_js(data, 0, 0);
    }, /InvalidPeriod|period/);
    
    // Empty data
    const emptyData = new Float64Array([]);
    assert.throws(() => {
        wasm.deviation_js(emptyData, 2, 0);
    }, /NotEnoughData|EmptyData/);
});

test('Deviation fast API (in-place)', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;
    
    // Allocate memory
    const inPtr = wasm.deviation_alloc(len);
    const outPtr = wasm.deviation_alloc(len);
    
    try {
        // Get memory views
        const memory = wasm.__wasm.memory.buffer;
        const inView = new Float64Array(memory, inPtr, len);
        const outView = new Float64Array(memory, outPtr, len);
        
        // Copy data to WASM memory
        inView.set(close);
        
        // Compute deviation
        wasm.deviation_into(inPtr, len, 20, 0, outPtr);
        
        // Compare with safe API
        const safeResult = wasm.deviation_js(close, 20, 0);
        assertArrayClose(Array.from(outView), safeResult, 1e-10, "Fast API should match safe API");
        
        // Test in-place (aliasing)
        inView.set(close); // Reset input
        wasm.deviation_into(inPtr, len, 20, 0, inPtr); // Output to same location as input
        assertArrayClose(Array.from(inView), safeResult, 1e-10, "In-place should match safe API");
    } finally {
        // Clean up
        wasm.deviation_free(inPtr, len);
        wasm.deviation_free(outPtr, len);
    }
});

test('Deviation batch single parameter set', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Single parameter set
    const batchResult = wasm.deviation_batch(close, {
        period_range: [20, 20, 0],
        devtype_range: [0, 0, 0]
    });
    
    assert(batchResult, 'Should have result');
    assert.strictEqual(batchResult.combos, 1, 'Should have 1 combination');
    assert.strictEqual(batchResult.cols, close.length, 'Cols should match input length');
    assert.strictEqual(batchResult.values.length, close.length, 'Values should match input length');
    
    // Should match single calculation
    const singleResult = wasm.deviation_js(close, 20, 0);
    // WASM batch uses a prefix-sum row kernel; tiny rounding deltas vs single O(1) path.
    // Keep a very tight tolerance while acknowledging ~1e-10 level differences on Node WASM.
    assertArrayClose(batchResult.values, singleResult, 3e-10, "Batch vs single mismatch");
});

test('Deviation batch multiple parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Multiple periods and deviation types
    const batchResult = wasm.deviation_batch(close, {
        period_range: [10, 30, 10],   // 10, 20, 30
        devtype_range: [0, 2, 1]       // 0, 1, 2
    });
    
    // Should have 3 * 3 = 9 combinations
    assert.strictEqual(batchResult.combos, 9);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.values.length, 9 * 100);
    
    // Verify first combination (period=10, devtype=0)
    const firstCombo = batchResult.values.slice(0, 100);
    const singleResult = wasm.deviation_js(close, 10, 0);
    assertArrayClose(firstCombo, singleResult, 3e-10, "First combination mismatch");
});

test('Deviation batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.deviation_batch_metadata(
        10, 30, 10,   // period: 10, 20, 30
        0, 2, 1       // devtype: 0, 1, 2
    );
    
    // Should have 9 combinations * 2 values each = 18 values
    assert.strictEqual(metadata.length, 18);
    
    // Check values (period, devtype pairs)
    const expected = [
        10, 0,  // combo 0
        10, 1,  // combo 1
        10, 2,  // combo 2
        20, 0,  // combo 3
        20, 1,  // combo 4
        20, 2,  // combo 5
        30, 0,  // combo 6
        30, 1,  // combo 7
        30, 2   // combo 8
    ];
    
    assertArrayClose(metadata, expected, 0, "Metadata mismatch");
});

test('Deviation streaming API', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const period = 5;
    
    // Create streaming instance
    const stream = new wasm.DeviationStream(period, 0);
    
    // Process values
    const results = [];
    for (const value of data) {
        const result = stream.update(value);
        results.push(result === undefined ? NaN : result);
    }
    
    // Compare with batch calculation
    const batchResult = wasm.deviation_js(new Float64Array(data), period, 0);
    assertArrayClose(results, batchResult, 1e-10, "Streaming vs batch mismatch");
});

test('Deviation NaN handling', () => {
    const dataWithNaN = new Float64Array([1, 2, NaN, 4, 5, 6, 7, 8, 9, 10]);
    
    const result = wasm.deviation_js(dataWithNaN, 5, 0);
    assert(result, 'Should handle NaN in input');
    assert.strictEqual(result.length, dataWithNaN.length);
    
    // NaN in input should propagate to output window
    assert(isNaN(result[2]), 'NaN should propagate');
    assert(isNaN(result[3]), 'NaN should affect window');
    assert(isNaN(result[4]), 'NaN should affect window');
    assert(isNaN(result[5]), 'NaN should affect window');
    assert(isNaN(result[6]), 'NaN should affect window');
    
    // After NaN leaves window, should recover
    assert(!isNaN(result[7]), 'Should recover after NaN leaves window');
});

test('Deviation edge cases', () => {
    // Minimum valid period
    const data = new Float64Array([1, 2, 3, 4, 5]);
    const result = wasm.deviation_js(data, 2, 0);
    assert.strictEqual(result.length, data.length);
    assert(isNaN(result[0]));
    assert(!isNaN(result[1]));
    
    // All same values (zero deviation)
    const sameData = new Float64Array(10).fill(42.0);
    const sameResult = wasm.deviation_js(sameData, 5, 0);
    for (let i = 4; i < sameResult.length; i++) {
        assertClose(sameResult[i], 0.0, 1e-10, `Deviation should be 0 for identical values at index ${i}`);
    }
});

test('Deviation fast batch API', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;
    
    // Allocate memory for input and output
    const inPtr = wasm.deviation_alloc(len);
    const combos = 3 * 3; // 3 periods × 3 devtypes
    const outPtr = wasm.deviation_alloc(len * combos);
    
    try {
        // Get memory views
        const memory = wasm.__wasm.memory.buffer;
        const inView = new Float64Array(memory, inPtr, len);
        
        // Copy data to WASM memory
        inView.set(close);
        
        // Compute batch deviation
        const rows = wasm.deviation_batch_into(
            inPtr, outPtr, len,
            10, 30, 10,  // period: 10, 20, 30
            0, 2, 1      // devtype: 0, 1, 2
        );
        
        assert.strictEqual(rows, 9, 'Should return 9 combinations');
        
        // Get output view
        const outView = new Float64Array(memory, outPtr, len * combos);
        
        // Verify first combination matches regular calculation
        const firstCombo = Array.from(outView.slice(0, len));
        const expected = wasm.deviation_js(close, 10, 0);
        assertArrayClose(firstCombo, expected, 3e-10, "First batch combo should match single calc");
    } finally {
        // Clean up
        wasm.deviation_free(inPtr, len);
        wasm.deviation_free(outPtr, len * combos);
    }
});

test('Deviation invalid devtype', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Invalid devtype (only 0-3 are valid)
    assert.throws(() => {
        wasm.deviation_js(data, 5, 4);
    }, /Invalid.*devtype|calculation error/i);
});

test('Deviation period edge validation', () => {
    const data = new Float64Array(100).fill(1.0).map((_, i) => i + 1);
    
    // Period = 1 should work
    const result1 = wasm.deviation_js(data, 1, 0);
    assert.strictEqual(result1.length, data.length);
    // With period=1, standard deviation is always 0
    for (let i = 0; i < result1.length; i++) {
        assertClose(result1[i], 0.0, 1e-10, `Period=1 should give 0 deviation at index ${i}`);
    }
    
    // Maximum valid period = data length
    const resultMax = wasm.deviation_js(data, data.length, 0);
    assert.strictEqual(resultMax.length, data.length);
    // All values before last should be NaN
    for (let i = 0; i < data.length - 1; i++) {
        assert(isNaN(resultMax[i]), `Should be NaN at index ${i}`);
    }
    assert(!isNaN(resultMax[data.length - 1]), 'Last value should be valid');
});

test('Deviation streaming vs batch consistency', () => {
    // Test that streaming produces same results as batch for all deviation types
    const data = [10, 15, 20, 18, 22, 25, 23, 27, 30, 28];
    const period = 4;
    
    for (let devtype = 0; devtype <= 3; devtype++) {
        const stream = new wasm.DeviationStream(period, devtype);
        const streamResults = [];
        
        for (const value of data) {
            const result = stream.update(value);
            streamResults.push(result === undefined ? NaN : result);
        }
        
        const batchResult = wasm.deviation_js(new Float64Array(data), period, devtype);
        assertArrayClose(
            streamResults, 
            batchResult, 
            1e-10, 
            `Streaming vs batch mismatch for devtype ${devtype}`
        );
    }
});

test('Deviation increasing/decreasing trends', () => {
    // Test deviation on known patterns
    const increasingData = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const decreasingData = new Float64Array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
    const period = 5;
    
    const incResult = wasm.deviation_js(increasingData, period, 0);
    const decResult = wasm.deviation_js(decreasingData, period, 0);
    
    // Both should have same standard deviation due to symmetry
    for (let i = period - 1; i < increasingData.length; i++) {
        assertClose(
            incResult[i], 
            decResult[i], 
            1e-10, 
            `Deviation should be same for symmetric data at index ${i}`
        );
    }
});

test.after(() => {
    console.log('DEVIATION WASM tests completed');
});
