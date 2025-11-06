/**
 * WASM binding tests for EMA indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
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
        wasm = await import(importPath);
        // Initialize the wasm module (required for generated wrapper)
        try {
            if (typeof wasm.default === 'function') {
                await wasm.default();
            }
        } catch (initErr) {
            // Fallback for environments where fetch(file://) is not implemented
            const wasmBinPath = path.join(__dirname, '../../pkg/my_project_bg.wasm');
            const bytes = fs.readFileSync(wasmBinPath);
            if (typeof wasm.initSync === 'function') {
                wasm.initSync(bytes);
            } else {
                throw initErr;
            }
        }
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('EMA partial params', () => {
    // Test with default parameters - mirrors check_ema_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.ema_js(close, 9);
    assert.strictEqual(result.length, close.length);
});

test('EMA accuracy', () => {
    // Test EMA matches expected values from Rust tests - mirrors check_ema_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.ema;
    
    const result = wasm.ema_js(
        close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.lastFive,
        1e-1,
        "EMA last 5 values mismatch"
    );
});

test('EMA default candles', () => {
    // Test EMA with default candle data - mirrors check_ema_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.ema_js(close, 9);
    assert.strictEqual(result.length, close.length);
});

test('EMA zero period', () => {
    // Test EMA fails with zero period - mirrors check_ema_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.ema_js(inputData, 0);
    }, "Expected error for zero period");
});

test('EMA period exceeds length', () => {
    // Test EMA fails when period exceeds data length - mirrors check_ema_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.ema_js(dataSmall, 10);
    }, "Expected error for period exceeding length");
});

test('EMA very small dataset', () => {
    // Test EMA with single data point - mirrors check_ema_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.ema_js(singlePoint, 9);
    }, "Expected error for insufficient data");
});

test('EMA empty input', () => {
    // Test EMA fails with empty input - mirrors check_ema_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.ema_js(empty, 9);
    }, /Input data slice is empty|Empty input/);
});

test('EMA all NaN input', () => {
    // Test EMA fails with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.ema_js(allNaN, 9);
    }, /All values are NaN/);
});

test('EMA NaN handling', () => {
    // Test EMA handles NaN values correctly - mirrors check_ema_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.ema_js(close, 9);
    assert.strictEqual(result.length, close.length);
    
    // Check that values after warm-up period are not NaN
    if (result.length > 30) {
        for (let i = 30; i < result.length; i++) {
            assert.ok(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('EMA warmup period', () => {
    // Test EMA warmup period behavior
    const simpleData = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    
    const result = wasm.ema_js(simpleData, period);
    
    // EMA starts from first value, so first value should NOT be NaN
    assert.ok(!isNaN(result[0]), "EMA should start from first value");
    
    // All values should be finite for EMA (unlike SMA which has NaN warmup)
    for (let i = 0; i < result.length; i++) {
        assert.ok(isFinite(result[i]), `EMA value at index ${i} should be finite`);
    }
});

test('EMA batch', () => {
    // Test EMA batch computation
    const close = new Float64Array(testData.close);
    
    // Test batch with period range using the new API
    const config = {
        period_range: [5, 20, 5]
    };
    
    const result = wasm.ema_batch(close, config);
    
    // New API returns an object with structured output
    const expectedPeriods = [5, 10, 15, 20];
    assert.strictEqual(result.rows, expectedPeriods.length);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, result.rows * result.cols);
    
    // Check combos to verify periods
    assert.strictEqual(result.combos.length, expectedPeriods.length);
    for (let i = 0; i < expectedPeriods.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedPeriods[i]);
    }
    
    // Test that we can also still use the metadata function
    const metadata = wasm.ema_batch_metadata_js(5, 20, 5);
    assert.strictEqual(metadata.length, expectedPeriods.length);
    for (let i = 0; i < expectedPeriods.length; i++) {
        assert.strictEqual(metadata[i], expectedPeriods[i]);
    }
});

test.skip('EMA fast API (ema_into)', () => {
    // Test the zero-copy fast API
    const close = new Float64Array(testData.close);
    const period = 9;
    
    // Allocate memory using ema_alloc
    const len = close.length;
    const inPtr = wasm.ema_alloc(len);
    const outPtr = wasm.ema_alloc(len);
    
    try {
        // Create a view of WASM memory
        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        
        // Copy input data to WASM memory
        const inOffset = inPtr / 8; // Convert byte offset to f64 offset
        for (let i = 0; i < len; i++) {
            memory[inOffset + i] = close[i];
        }
        
        // Call the fast API
        wasm.ema_into(inPtr, outPtr, len, period);
        
        // Read results from WASM memory
        const outOffset = outPtr / 8;
        const result = new Float64Array(len);
        for (let i = 0; i < len; i++) {
            result[i] = memory[outOffset + i];
        }
        
        // Compare with safe API results
        const expected = wasm.ema_js(close, period);
        assertArrayClose(result, expected, 1e-10, "Fast API results should match safe API");
        
    } finally {
        // Always free allocated memory
        wasm.ema_free(inPtr, len);
        wasm.ema_free(outPtr, len);
    }
});

test.skip('EMA fast API with aliasing', () => {
    // Test in-place operation (input and output use same buffer)
    const close = new Float64Array(testData.close);
    const period = 9;
    
    // Allocate single buffer for both input and output
    const len = close.length;
    const ptr = wasm.ema_alloc(len);
    
    try {
        // Create a view of WASM memory
        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        
        // Copy input data to WASM memory
        const offset = ptr / 8;
        for (let i = 0; i < len; i++) {
            memory[offset + i] = close[i];
        }
        
        // Call fast API with same pointer for input and output (aliasing)
        wasm.ema_into(ptr, ptr, len, period);
        
        // Read results from same buffer
        const result = new Float64Array(len);
        for (let i = 0; i < len; i++) {
            result[i] = memory[offset + i];
        }
        
        // Compare with safe API results
        const expected = wasm.ema_js(close, period);
        assertArrayClose(result, expected, 1e-10, "In-place operation should produce correct results");
        
    } finally {
        wasm.ema_free(ptr, len);
    }
});

test.skip('EMA batch fast API', () => {
    // Test batch fast API
    const close = new Float64Array(testData.close);
    
    // Batch parameters
    const periodStart = 5;
    const periodEnd = 20;
    const periodStep = 5;
    const expectedPeriods = [5, 10, 15, 20];
    const rows = expectedPeriods.length;
    const cols = close.length;
    
    // Allocate memory
    const inPtr = wasm.ema_alloc(cols);
    const outPtr = wasm.ema_alloc(rows * cols);
    
    try {
        // Create a view of WASM memory
        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        
        // Copy input data to WASM memory
        const inOffset = inPtr / 8;
        for (let i = 0; i < cols; i++) {
            memory[inOffset + i] = close[i];
        }
        
        // Call batch fast API
        const resultRows = wasm.ema_batch_into(inPtr, outPtr, cols, periodStart, periodEnd, periodStep);
        assert.strictEqual(resultRows, rows);
        
        // Read results from WASM memory
        const outOffset = outPtr / 8;
        const result = new Float64Array(rows * cols);
        for (let i = 0; i < result.length; i++) {
            result[i] = memory[outOffset + i];
        }
        
        // Verify some values are computed (not all NaN)
        let hasValidValues = false;
        for (let i = 0; i < result.length; i++) {
            if (!isNaN(result[i])) {
                hasValidValues = true;
                break;
            }
        }
        assert.ok(hasValidValues, "Batch results should contain valid values");
        
    } finally {
        wasm.ema_free(inPtr, cols);
        wasm.ema_free(outPtr, rows * cols);
    }
});

test('EMA batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep
    const singleBatch = wasm.ema_batch(close, {
        period_range: [5, 5, 1]
    });
    assert.strictEqual(singleBatch.rows, 1);
    assert.strictEqual(singleBatch.combos.length, 1);
    assert.strictEqual(singleBatch.combos[0].period, 5);
    
    // Edge case: step larger than range
    const largeSweep = wasm.ema_batch(close, {
        period_range: [3, 5, 10]
    });
    assert.strictEqual(largeSweep.rows, 1); // Should only have start value
    assert.strictEqual(largeSweep.combos[0].period, 3);
});

test('EMA batch consistency', () => {
    // Test that batch results match individual calculations
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset
    
    const batchResult = wasm.ema_batch(close, {
        period_range: [10, 30, 10]  // periods: 10, 20, 30
    });
    
    const expectedPeriods = [10, 20, 30];
    assert.strictEqual(batchResult.rows, expectedPeriods.length);
    
    // Verify each batch row matches individual calculation
    for (let i = 0; i < expectedPeriods.length; i++) {
        const period = expectedPeriods[i];
        const singleResult = wasm.ema_js(close, period);
        
        // Extract row from batch
        const rowStart = i * close.length;
        const rowEnd = rowStart + close.length;
        const batchRow = batchResult.values.slice(rowStart, rowEnd);
        
        // Compare values
        assertArrayClose(batchRow, singleResult, 1e-10, 
                        `Batch row for period ${period} doesn't match single calculation`);
    }
});

test('EMA batch multiple periods', () => {
    // Test comprehensive batch with multiple periods
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.ema_batch(close, {
        period_range: [5, 15, 2]  // periods: 5, 7, 9, 11, 13, 15
    });
    
    const expectedPeriods = [5, 7, 9, 11, 13, 15];
    assert.strictEqual(batchResult.rows, expectedPeriods.length);
    assert.strictEqual(batchResult.cols, close.length);
    
    // Verify all expected periods are in combos
    for (let i = 0; i < expectedPeriods.length; i++) {
        assert.strictEqual(batchResult.combos[i].period, expectedPeriods[i]);
    }
    
    // Verify dimensions
    assert.strictEqual(batchResult.values.length, expectedPeriods.length * close.length);
});

test('EMA zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.ema_into(0, 0, 10, 9);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.ema_alloc(10);
    try {
        // Invalid period (0)
        assert.throws(() => {
            wasm.ema_into(ptr, ptr, 10, 0);
        }, /Invalid period/i);
        
        // Invalid period (exceeds length)
        assert.throws(() => {
            wasm.ema_into(ptr, ptr, 10, 20);
        }, /Invalid period/i);
    } finally {
        wasm.ema_free(ptr, 10);
    }
});

test.skip('EMA memory allocation edge cases', () => {
    // Test allocation and free of various sizes
    const sizes = [0, 1, 10, 100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.ema_alloc(size);
        assert(ptr !== 0 || size === 0, `Failed to allocate ${size} elements`);
        
        if (size > 0) {
            // Write pattern to verify memory
            const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
            for (let i = 0; i < Math.min(10, size); i++) {
                memView[i] = i * 1.5;
            }
            
            // Verify pattern
            for (let i = 0; i < Math.min(10, size); i++) {
                assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
            }
        }
        
        wasm.ema_free(ptr, size);
    }
});

test('EMA batch fast API error handling', () => {
    // Test null pointers
    assert.throws(() => {
        wasm.ema_batch_into(0, 0, 10, 5, 20, 5);
    }, /null pointer/i);
    
    // Test with valid pointers but invalid parameters
    const inPtr = wasm.ema_alloc(10);
    const outPtr = wasm.ema_alloc(100); // Enough for multiple results
    
    try {
        // This should succeed
        const rows = wasm.ema_batch_into(inPtr, outPtr, 10, 5, 10, 5);
        assert.strictEqual(rows, 2); // periods 5 and 10
    } finally {
        wasm.ema_free(inPtr, 10);
        wasm.ema_free(outPtr, 100);
    }
});
