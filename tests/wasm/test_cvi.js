/**
 * WASM binding tests for CVI indicator.
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

test('CVI partial params', () => {
    // Test with default parameters - mirrors check_cvi_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.cvi_js(high, low, 10);
    assert.strictEqual(result.length, high.length);
});

test('CVI accuracy', async () => {
    // Test CVI matches expected values from Rust tests - mirrors check_cvi_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const expected = EXPECTED_OUTPUTS.cvi;
    const period = expected.accuracyParams.period; // Use period=5 for accuracy test
    
    const result = wasm.cvi_js(
        high,
        low,
        period
    );
    
    assert.strictEqual(result.length, high.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "CVI last 5 values mismatch"
    );
    
    // Note: We don't compare with generate_references here because it uses default period=10
    // await compareWithRust('cvi', result, 'hl', { period });
});

test('CVI default candles', () => {
    // Test CVI with default parameters - mirrors check_cvi_input_with_default_candles
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.cvi_js(high, low, 10);
    assert.strictEqual(result.length, high.length);
});

test('CVI zero period', () => {
    // Test CVI fails with zero period - mirrors check_cvi_with_zero_period
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.cvi_js(high, low, 0);
    }, /Invalid period/);
});

test('CVI period exceeds length', () => {
    // Test CVI fails when period exceeds data length - mirrors check_cvi_with_period_exceeding_data_length
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.cvi_js(high, low, 10);
    }, /Invalid period/);
});

test('CVI very small dataset', () => {
    // Test CVI fails with insufficient data - mirrors check_cvi_very_small_data_set
    const high = new Float64Array([42.0]);
    const low = new Float64Array([40.0]);
    
    assert.throws(() => {
        wasm.cvi_js(high, low, 10);
    }, /Invalid period|Not enough valid data/);
});

test('CVI empty input', () => {
    // Test CVI fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.cvi_js(empty, empty, 10);
    }, /Empty data/);
});

test('CVI with NaN data', () => {
    // Test CVI handles NaN data - mirrors check_cvi_with_nan_data
    const high = new Float64Array([NaN, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, NaN]);
    
    assert.throws(() => {
        wasm.cvi_js(high, low, 2);
    }, /Not enough valid data|All.*values are NaN/);
});

test('CVI slice reinput', () => {
    // Test CVI with output as input - mirrors check_cvi_slice_reinput
    const high = new Float64Array([10.0, 12.0, 12.5, 12.2, 13.0, 14.0, 15.0, 16.0, 16.5, 17.0, 17.5, 18.0]);
    const low = new Float64Array([9.0, 10.0, 11.5, 11.0, 12.0, 13.5, 14.0, 14.5, 15.5, 16.0, 16.5, 17.0]);
    
    const first_result = wasm.cvi_js(high, low, 3);
    assert.strictEqual(first_result.length, high.length);
    
    const second_result = wasm.cvi_js(first_result, low, 3);
    assert.strictEqual(second_result.length, low.length);
});

// Batch processing tests
test('CVI batch single parameter', () => {
    // Test batch processing with single parameter
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.cvi_batch(high, low, {
        period_range: [10, 10, 0]
    });
    
    // Single result matches regular API
    const singleResult = wasm.cvi_js(high, low, 10);
    assertArrayClose(
        result.values,
        singleResult,
        1e-10,
        "Batch single parameter mismatch"
    );
});

test('CVI batch metadata from result', () => {
    // Test that batch result includes correct parameter combinations
    const high = new Float64Array(50);  // Need at least 2*15-1=29 values
    const low = new Float64Array(50);
    high.fill(100);
    low.fill(90);
    
    const result = wasm.cvi_batch(high, low, {
        period_range: [5, 15, 5]  // periods: 5, 10, 15
    });
    
    // Should have 3 combinations
    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    
    // Check combinations
    assert.strictEqual(result.combos[0].period, 5);
    assert.strictEqual(result.combos[1].period, 10);
    assert.strictEqual(result.combos[2].period, 15);
});

test('CVI batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    
    const batchResult = wasm.cvi_batch(high, low, {
        period_range: [10, 20, 10]  // 2 periods
    });
    
    // Should have 2 combinations
    assert.strictEqual(batchResult.combos.length, 2);
    assert.strictEqual(batchResult.rows, 2);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 2 * 50);
    
    // Verify structure
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const period = batchResult.combos[combo].period;
        
        const rowStart = combo * 50;
        const rowData = batchResult.values.slice(rowStart, rowStart + 50);
        
        // First 2*period-1 values should be NaN
        const warmup = 2 * period - 1;
        for (let i = 0; i < warmup && i < 50; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        // After warmup should have values
        for (let i = warmup; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

// Memory management tests
test('CVI zero-copy API', () => {
    // Test zero-copy API with allocated memory
    const data = new Float64Array([100, 110, 105, 120, 115, 125, 130, 135, 140, 145]);
    const period = 3;
    
    // Allocate buffers
    const highPtr = wasm.cvi_alloc(data.length);
    const lowPtr = wasm.cvi_alloc(data.length);
    assert(highPtr !== 0, 'Failed to allocate high buffer');
    assert(lowPtr !== 0, 'Failed to allocate low buffer');
    
    // Create views into WASM memory
    const highView = new Float64Array(
        wasm.__wasm.memory.buffer,
        highPtr,
        data.length
    );
    const lowView = new Float64Array(
        wasm.__wasm.memory.buffer,
        lowPtr,
        data.length
    );
    
    // Fill with test data
    highView.set(data);
    for (let i = 0; i < data.length; i++) {
        lowView[i] = data[i] - 5; // Low is 5 less than high
    }
    
    // Compute CVI in-place (using high buffer as output)
    try {
        wasm.cvi_into(highPtr, lowPtr, highPtr, data.length, period);
        
        // Verify results match regular API
        const regularResult = wasm.cvi_js(data, lowView, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(highView[i])) {
                continue;
            }
            assertClose(highView[i], regularResult[i], 1e-10, `Mismatch at index ${i}`);
        }
    } finally {
        // Always free memory
        wasm.cvi_free(highPtr, data.length);
        wasm.cvi_free(lowPtr, data.length);
    }
});

test('CVI zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = 100 + Math.sin(i * 0.1) * 10;
    }
    
    const highPtr = wasm.cvi_alloc(size);
    const lowPtr = wasm.cvi_alloc(size);
    assert(highPtr !== 0, 'Failed to allocate large high buffer');
    assert(lowPtr !== 0, 'Failed to allocate large low buffer');
    
    try {
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        highView.set(data);
        for (let i = 0; i < size; i++) {
            lowView[i] = data[i] - 5;
        }
        
        wasm.cvi_into(highPtr, lowPtr, highPtr, size, 10);
        
        // Recreate view in case memory grew
        const highView2 = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        
        // Check warmup period has NaN
        for (let i = 0; i < 19; i++) { // 2*period-1 = 19
            assert(isNaN(highView2[i]), `Expected NaN at index ${i}`);
        }
        
        // Check we have values after warmup
        for (let i = 19; i < Math.min(100, size); i++) {
            assert(!isNaN(highView2[i]), `Expected value at index ${i}`);
        }
    } finally {
        wasm.cvi_free(highPtr, size);
        wasm.cvi_free(lowPtr, size);
    }
});

test('CVI zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.cvi_into(0, 0, 0, 10, 5);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.cvi_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.cvi_into(ptr, ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        // Period exceeds length
        assert.throws(() => {
            wasm.cvi_into(ptr, ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.cvi_free(ptr, 10);
    }
});

test('CVI all NaN input', () => {
    // Test CVI with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.cvi_js(allNaN, allNaN, 10);
    }, /All.*NaN|Not enough valid/);
});

test('CVI NaN handling', () => {
    // Test CVI handles NaN values correctly in warmup period
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const expected = EXPECTED_OUTPUTS.cvi;
    
    const result = wasm.cvi_js(high, low, 10);
    assert.strictEqual(result.length, high.length);
    
    // First 2*period-1 values should be NaN (warmup period)
    const warmupPeriod = expected.warmupPeriod;  // 19 for period=10
    assertAllNaN(result.slice(0, warmupPeriod), "Expected NaN in warmup period");
    
    // After warmup period, no NaN values should exist
    if (result.length > warmupPeriod + 20) {
        for (let i = warmupPeriod; i < warmupPeriod + 20; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('CVI batch edge cases', () => {
    // Test edge cases for batch processing
    const high = new Float64Array([100, 102, 105, 108, 112, 117, 123, 130, 138, 147]);
    const low = new Float64Array([99, 100, 102, 105, 108, 112, 117, 123, 130, 138]);
    
    // Single value sweep
    const singleBatch = wasm.cvi_batch(high, low, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.cvi_batch(high, low, {
        period_range: [5, 7, 10]  // Step larger than range
    });
    
    // Should only have period=5
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    assert.strictEqual(largeBatch.combos[0].period, 5);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.cvi_batch(new Float64Array([]), new Float64Array([]), {
            period_range: [10, 10, 0]
        });
    }, /Empty data|All.*NaN/);
});

// Memory leak prevention test
test('CVI zero-copy memory management', () => {
    // Allocate and free multiple times to check for leaks
    const sizes = [10, 100, 1000, 10000];
    
    for (const size of sizes) {
        const highPtr = wasm.cvi_alloc(size);
        const lowPtr = wasm.cvi_alloc(size);
        assert(highPtr !== 0, `Failed to allocate ${size} elements for high`);
        assert(lowPtr !== 0, `Failed to allocate ${size} elements for low`);
        
        // Write pattern to verify memory
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            highView[i] = i * 1.5;
            lowView[i] = i * 1.5 - 2;
        }
        
        // Free memory
        wasm.cvi_free(highPtr, size);
        wasm.cvi_free(lowPtr, size);
    }
});

test.after(() => {
    console.log('CVI WASM tests completed');
});
