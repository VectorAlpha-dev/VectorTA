/**
 * WASM binding tests for Kurtosis indicator.
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
    assertNoNaN
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
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('Kurtosis partial params', () => {
    // Test with default parameters - mirrors check_kurtosis_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.kurtosis_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('Kurtosis accuracy', async () => {
    // Test Kurtosis matches expected values from Rust tests - mirrors check_kurtosis_accuracy
    const hl2 = new Float64Array(testData.hl2);
    
    const result = wasm.kurtosis_js(hl2, 5);
    
    assert.strictEqual(result.length, hl2.length);
    
    // Expected values from Rust test
    const expectedLast5 = [
        -0.5438903789933454,
        -1.6848139264816433,
        -1.6331336745945797,
        -0.6130805596586351,
        -0.027802601135927585,
    ];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-10,
        "Kurtosis last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('kurtosis', result, 'hl2', { period: 5 });
});

test('Kurtosis default candles', () => {
    // Test Kurtosis with default parameters - mirrors check_kurtosis_default_candles
    const hl2 = new Float64Array(testData.hl2);
    
    const result = wasm.kurtosis_js(hl2, 5);
    assert.strictEqual(result.length, hl2.length);
});

test('Kurtosis zero period', () => {
    // Test Kurtosis fails with zero period - mirrors check_kurtosis_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.kurtosis_js(inputData, 0);
    }, /Invalid period/);
});

test('Kurtosis period exceeds length', () => {
    // Test Kurtosis fails when period exceeds data length - mirrors check_kurtosis_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.kurtosis_js(dataSmall, 10);
    }, /Invalid period/);
});

test('Kurtosis all nan', () => {
    // Test Kurtosis handles all NaN values - matches ALMA test pattern
    const nanData = new Float64Array(20).fill(NaN);
    
    assert.throws(() => {
        wasm.kurtosis_js(nanData, 5);
    }, /All values are NaN/);
});

test('Kurtosis empty input', () => {
    // Test Kurtosis with empty input - matches ALMA test pattern
    const emptyData = new Float64Array([]);
    
    assert.throws(() => {
        wasm.kurtosis_js(emptyData, 5);
    }, /Input data slice is empty/);
});

test('Kurtosis nan prefix', () => {
    // Test Kurtosis handles NaN prefix - mirrors Rust test pattern
    const nanPrefixData = new Float64Array(30);
    const period = 5;
    nanPrefixData.fill(NaN, 0, 10);
    for (let i = 10; i < 30; i++) {
        nanPrefixData[i] = 50.0 + i * 0.5;
    }
    
    const result = wasm.kurtosis_js(nanPrefixData, period);
    
    assert.strictEqual(result.length, nanPrefixData.length);
    
    // Check warmup NaN values (first 10 NaN + (period-1) warmup = 14)
    const expectedNaNCount = 10 + (period - 1);
    for (let i = 0; i < expectedNaNCount; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    // Valid values start after warmup
    for (let i = expectedNaNCount; i < result.length; i++) {
        assert(!isNaN(result[i]), `Expected valid value at index ${i}`);
    }
});

test('Kurtosis batch operation', () => {
    // Test batch kurtosis calculation - mirrors Rust batch tests
    const hl2 = new Float64Array(testData.hl2);
    
    // Test batch with default period
    const config = {
        period_range: [5, 5, 0]  // Default period only
    };
    
    const result = wasm.kurtosis_batch(hl2, config);
    
    assert(result.values);
    assert(result.periods);
    assert.strictEqual(result.rows, 1); // 1 period
    assert.strictEqual(result.cols, hl2.length);
    
    // Check last 5 values match expected from Rust test
    const expectedLast5 = [
        -0.5438903789933454,
        -1.6848139264816433,
        -1.6331336745945797,
        -0.6130805596586351,
        -0.027802601135927585,
    ];
    
    const last5 = result.values.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-6,
        "Kurtosis batch last 5 values mismatch"
    );
});

test('Kurtosis batch multiple periods', () => {
    // Test batch with multiple periods - matches ALMA pattern
    const close = new Float64Array(testData.close);
    
    const config = {
        period_range: [5, 20, 5]  // 5, 10, 15, 20
    };
    
    const result = wasm.kurtosis_batch(close, config);
    
    assert(result.values);
    assert(result.periods);
    assert.strictEqual(result.rows, 4); // 4 periods
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, 4 * close.length);
    
    // Verify periods array has correct values
    const expectedPeriods = [5, 10, 15, 20];
    assert.deepStrictEqual(result.periods, expectedPeriods);
});

test('Kurtosis fast API', () => {
    // Test fast/unsafe API
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate memory for both input and output
    const inPtr = wasm.kurtosis_alloc(len);
    const outPtr = wasm.kurtosis_alloc(len);
    
    try {
        // Create input view and copy data
        const inputArray = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inputArray.set(close);
        
        // Call fast API
        wasm.kurtosis_into(inPtr, outPtr, len, 5);
        
        // Read results (recreate view in case memory grew)
        const output = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        // Compare with safe API
        const expected = wasm.kurtosis_js(close, 5);
        assertArrayClose(output, expected, 1e-10, "Fast API mismatch");
        
    } finally {
        // Free memory
        wasm.kurtosis_free(inPtr, len);
        wasm.kurtosis_free(outPtr, len);
    }
});

test('Kurtosis fast API aliasing', () => {
    // Test fast API with aliasing (in-place operation)
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate memory
    const ptr = wasm.kurtosis_alloc(len);
    
    try {
        // Create memory view
        const dataArray = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        dataArray.set(close);
        
        // Call fast API with same pointer for input and output (aliasing)
        wasm.kurtosis_into(ptr, ptr, len, 5);
        
        // Recreate view in case memory grew
        const resultArray = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        
        // Compare with safe API
        const expected = wasm.kurtosis_js(close, 5);
        assertArrayClose(resultArray, expected, 1e-10, "Fast API aliasing mismatch");
        
    } finally {
        // Free memory
        wasm.kurtosis_free(ptr, len);
    }
});


test('Kurtosis very small dataset', () => {
    // Test Kurtosis with very small dataset - mirrors Rust test
    const smallData = new Float64Array([5.0, 10.0, 15.0, 20.0, 25.0]);
    const period = 5;
    
    const result = wasm.kurtosis_js(smallData, period);
    assert.strictEqual(result.length, smallData.length);
    
    // First (period-1) values should be NaN (warmup)
    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    // Last value should be valid
    assert(!isNaN(result[period - 1]), `Expected valid value at index ${period - 1}`);
});

test('Kurtosis batch metadata from result', () => {
    // Test batch metadata is correctly returned
    const close = new Float64Array(testData.close);
    
    const config = {
        period_range: [5, 10, 1]  // 6 values: 5,6,7,8,9,10
    };
    
    const result = wasm.kurtosis_batch(close, config);
    
    assert(result.periods);
    assert.strictEqual(result.periods.length, 6);
    
    // Check each period value is correct
    const expectedPeriods = [5, 6, 7, 8, 9, 10];
    assert.deepStrictEqual(result.periods, expectedPeriods);
});

test('Kurtosis batch edge cases', () => {
    // Test batch with edge case parameters - matches ALMA pattern
    const close = new Float64Array(testData.close);
    
    // Single parameter (step = 0)
    const config1 = {
        period_range: [5, 5, 0]
    };
    const result1 = wasm.kurtosis_batch(close, config1);
    assert.strictEqual(result1.rows, 1, "Single parameter should give 1 row");
    
    // Large step (only 2 values: 5, 50)
    if (close.length > 50) {
        const config2 = {
            period_range: [5, 50, 45]
        };
        const result2 = wasm.kurtosis_batch(close.slice(0, 100), config2);
        assert.strictEqual(result2.rows, 2, "Large step should give 2 rows");
        assert.deepStrictEqual(result2.periods, [5, 50]);
    }
});

test('Kurtosis batch full parameter sweep', () => {
    // Test full parameter sweep - matches ALMA pattern
    const close = new Float64Array(testData.close.slice(0, 50)); // Use smaller dataset
    
    // Multiple period values: 5, 7, 9
    const config = {
        period_range: [5, 9, 2]
    };
    
    const result = wasm.kurtosis_batch(close, config);
    
    assert(result.values);
    assert(result.periods);
    assert.strictEqual(result.rows, 3); // 3 periods
    assert.strictEqual(result.cols, 50);
    
    // Verify each row matches individual calculation
    const periods = [5, 7, 9];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 50;
        const rowEnd = rowStart + 50;
        const rowData = result.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.kurtosis_js(close, periods[i]);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Period ${periods[i]} mismatch`
        );
    }
});

