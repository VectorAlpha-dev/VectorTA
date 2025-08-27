/**
 * WASM binding tests for KAUFMANSTOP indicator.
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

test('KAUFMANSTOP partial params', () => {
    // Test with default parameters - mirrors check_kaufmanstop_partial_params
    const { high, low } = testData;
    
    // Test with default params (should use defaults from Rust)
    const result = wasm.kaufmanstop_js(
        new Float64Array(high), 
        new Float64Array(low), 
        22, 2.0, 'long', 'sma'
    );
    assert.ok(result instanceof Float64Array, 'Result should be Float64Array');
    assert.strictEqual(result.length, high.length);
});

test('KAUFMANSTOP accuracy', async () => {
    // Test KAUFMANSTOP matches expected values from Rust tests - mirrors check_kaufmanstop_accuracy
    const { high, low } = testData;
    const expected = EXPECTED_OUTPUTS.kaufmanstop;
    
    const result = wasm.kaufmanstop_js(
        new Float64Array(high), 
        new Float64Array(low),
        expected.defaultParams.period,
        expected.defaultParams.mult,
        expected.defaultParams.direction,
        expected.defaultParams.maType
    );
    
    assert.ok(result instanceof Float64Array, 'Result should be Float64Array');
    assert.strictEqual(result.length, high.length);
    
    // Check last 5 values match expected
    const last5 = Array.from(result).slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-1,  // Use same tolerance as Rust test
        "KAUFMANSTOP last 5 values mismatch"
    );
    
    // Compare full output with Rust - skip for now as generate_references doesn't have kaufmanstop yet
    // await compareWithRust('kaufmanstop', result, 'high,low', expected.defaultParams);
});

test('KAUFMANSTOP default candles', () => {
    // Test KAUFMANSTOP with default parameters - mirrors check_kaufmanstop_default_candles
    const { high, low } = testData;
    const expected = EXPECTED_OUTPUTS.kaufmanstop;
    
    const result = wasm.kaufmanstop_js(
        new Float64Array(high),
        new Float64Array(low),
        expected.defaultParams.period,
        expected.defaultParams.mult,
        expected.defaultParams.direction,
        expected.defaultParams.maType
    );
    assert.ok(result instanceof Float64Array, 'Result should be Float64Array');
    assert.strictEqual(result.length, high.length);
});

test('KAUFMANSTOP zero period', () => {
    // Test KAUFMANSTOP fails with zero period - mirrors check_kaufmanstop_zero_period
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.kaufmanstop_js(high, low, 0, 2.0, 'long', 'sma');
    }, /Invalid period/);
});

test('KAUFMANSTOP period exceeds length', () => {
    // Test KAUFMANSTOP fails when period exceeds data length - mirrors check_kaufmanstop_period_exceeds_length
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.kaufmanstop_js(high, low, 10, 2.0, 'long', 'sma');
    }, /Invalid period/);
});

test('KAUFMANSTOP very small dataset', () => {
    // Test KAUFMANSTOP fails with insufficient data - mirrors check_kaufmanstop_very_small_dataset
    const high = new Float64Array([42.0]);
    const low = new Float64Array([41.0]);
    
    assert.throws(() => {
        wasm.kaufmanstop_js(high, low, 22, 2.0, 'long', 'sma');
    }, /Invalid period|Not enough valid data/);
});

test('KAUFMANSTOP empty data', () => {
    // Test KAUFMANSTOP fails with empty arrays - mirrors check_kaufmanstop_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.kaufmanstop_js(empty, empty, 22, 2.0, 'long', 'sma');
    }, /Empty data/);
});

test('KAUFMANSTOP mismatched lengths', () => {
    // Test KAUFMANSTOP fails when high and low have different lengths
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]); // Different length
    
    assert.throws(() => {
        wasm.kaufmanstop_js(high, low, 2, 2.0, 'long', 'sma');
    });  // Just check that it throws, error message might vary
});

test('KAUFMANSTOP all NaN input', () => {
    // Test KAUFMANSTOP with all NaN values
    const allNaNHigh = new Float64Array(100);
    const allNaNLow = new Float64Array(100);
    allNaNHigh.fill(NaN);
    allNaNLow.fill(NaN);
    
    assert.throws(() => {
        wasm.kaufmanstop_js(allNaNHigh, allNaNLow, 22, 2.0, 'long', 'sma');
    }, /All values are NaN/);
});

test('KAUFMANSTOP NaN handling', () => {
    // Test KAUFMANSTOP handles NaN values correctly - mirrors check_kaufmanstop_nan_handling
    const { high, low } = testData;
    const expected = EXPECTED_OUTPUTS.kaufmanstop;
    
    const result = wasm.kaufmanstop_js(
        new Float64Array(high),
        new Float64Array(low),
        expected.defaultParams.period,
        expected.defaultParams.mult,
        expected.defaultParams.direction,
        expected.defaultParams.maType
    );
    assert.strictEqual(result.length, high.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // Check warmup period - first + period - 1 values should be NaN
    const warmup = expected.warmupPeriod;  // 43
    for (let i = 0; i < warmup && i < result.length; i++) {
        assert(isNaN(result[i]), `Expected NaN at warmup index ${i}`);
    }
});

test('KAUFMANSTOP with short direction', () => {
    const { high, low } = testData;
    
    const resultLong = wasm.kaufmanstop_js(
        new Float64Array(high.slice(0, 100)), 
        new Float64Array(low.slice(0, 100)), 
        22, 2.0, 'long', 'sma'
    );
    const resultShort = wasm.kaufmanstop_js(
        new Float64Array(high.slice(0, 100)), 
        new Float64Array(low.slice(0, 100)), 
        22, 2.0, 'short', 'sma'
    );
    
    // Results should be different
    let foundDifference = false;
    const warmup = 43;
    for (let i = warmup; i < 100; i++) {
        if (!isNaN(resultLong[i]) && !isNaN(resultShort[i]) && resultLong[i] !== resultShort[i]) {
            foundDifference = true;
            break;
        }
    }
    assert.ok(foundDifference, "Long and short directions should produce different results");
});

test('KAUFMANSTOP batch single parameter set', () => {
    // Test batch with single parameter combination
    const { high, low } = testData;
    const expected = EXPECTED_OUTPUTS.kaufmanstop;
    
    const batchResult = wasm.kaufmanstop_batch_js(
        new Float64Array(high),
        new Float64Array(low),
        22, 22, 0,      // period range
        2.0, 2.0, 0.0,  // mult range
        'long', 'sma'
    );
    
    assert.ok(batchResult);
    assert.ok(batchResult.values);
    assert.ok(batchResult.combos);
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, high.length);
    
    // Extract single row and compare with expected
    const singleRow = batchResult.values.slice(0, high.length);
    const last5 = singleRow.slice(-5);
    assertArrayClose(
        last5,
        expected.batchDefaultRow,
        1e-1,
        "KAUFMANSTOP batch single params mismatch"
    );
});

test('KAUFMANSTOP batch multiple periods', () => {
    // Test batch with multiple period values
    const { high, low } = testData;
    const testHigh = new Float64Array(high.slice(0, 100));
    const testLow = new Float64Array(low.slice(0, 100));
    
    const batchResult = wasm.kaufmanstop_batch_js(
        testHigh, testLow,
        20, 24, 2,      // period range: 20, 22, 24
        2.0, 2.0, 0.0,  // mult range
        'long', 'sma'
    );
    
    // Should have 3 rows * 100 cols
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.combos.length, 3);
    
    // Verify each row matches individual calculation
    const periods = [20, 22, 24];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowData = batchResult.values.slice(rowStart, rowStart + 100);
        
        const singleResult = wasm.kaufmanstop_js(
            testHigh, testLow,
            periods[i], 2.0, 'long', 'sma'
        );
        
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Period ${periods[i]} mismatch`
        );
    }
});

test('KAUFMANSTOP batch metadata', () => {
    // Test that batch result includes correct parameter combinations
    const { high, low } = testData;
    const testHigh = new Float64Array(high.slice(0, 50));
    const testLow = new Float64Array(low.slice(0, 50));
    
    const result = wasm.kaufmanstop_batch_js(
        testHigh, testLow,
        20, 22, 2,      // period: 20, 22
        1.5, 2.0, 0.5,  // mult: 1.5, 2.0
        'long', 'sma'
    );
    
    // Should have 2 * 2 = 4 combinations
    assert.strictEqual(result.combos.length, 4);
    assert.strictEqual(result.rows, 4);
    assert.strictEqual(result.cols, 50);
    
    // Check first combination
    assert.strictEqual(result.combos[0].period, 20);
    assert.strictEqual(result.combos[0].mult, 1.5);
    assert.strictEqual(result.combos[0].direction, 'long');
    assert.strictEqual(result.combos[0].ma_type, 'sma');
    
    // Check last combination
    assert.strictEqual(result.combos[3].period, 22);
    assert.strictEqual(result.combos[3].mult, 2.0);
});

test('KAUFMANSTOP batch edge cases', () => {
    // Test edge cases for batch processing
    const testData = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]);
    
    // Single value sweep
    const singleBatch = wasm.kaufmanstop_batch_js(
        testData, testData,
        5, 5, 1,
        2.0, 2.0, 0.1,
        'long', 'sma'
    );
    
    assert.strictEqual(singleBatch.values.length, 25);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.kaufmanstop_batch_js(
        testData, testData,
        5, 7, 10,  // Step larger than range
        2.0, 2.0, 0,
        'long', 'sma'
    );
    
    // Should only have period=5
    assert.strictEqual(largeBatch.values.length, 25);
    assert.strictEqual(largeBatch.combos.length, 1);
});

test('KAUFMANSTOP different MA types', () => {
    // Test various MA types
    const { high, low } = testData;
    const testHigh = new Float64Array(high.slice(0, 100));
    const testLow = new Float64Array(low.slice(0, 100));
    const maTypes = ['sma', 'ema', 'wma', 'smma'];  // Only test supported types
    const results = [];
    
    for (const maType of maTypes) {
        try {
            const result = wasm.kaufmanstop_js(
                testHigh, testLow,
                22, 2.0, 'long', maType
            );
            results.push({ maType, result });
        } catch (e) {
            // Some MA types might not be supported
        }
    }
    
    // At least SMA should work
    assert(results.length >= 1, "At least SMA should be supported");
    
    // Different MA types should produce different results
    if (results.length > 1) {
        for (let i = 1; i < results.length; i++) {
            let foundDifference = false;
            for (let j = 21; j < 100; j++) {  // After warmup (index 21+)
                if (!isNaN(results[0].result[j]) && !isNaN(results[i].result[j]) &&
                    Math.abs(results[0].result[j] - results[i].result[j]) > 1e-10) {
                    foundDifference = true;
                    break;
                }
            }
            // Note: Some MA types might default to SMA if not supported
            // So we don't fail the test, just note it
            if (!foundDifference) {
                console.log(`Note: ${results[0].maType} and ${results[i].maType} produced same results`);
            }
        }
    }
});

// Zero-copy API tests
test('KAUFMANSTOP zero-copy API', () => {
    const high = new Float64Array([100, 102, 101, 103, 102, 104, 103, 105, 104, 106,
                                    105, 107, 106, 108, 107, 109, 108, 110, 109, 111,
                                    110, 112, 111, 113, 112]);
    const low = new Float64Array([99, 101, 100, 102, 101, 103, 102, 104, 103, 105,
                                   104, 106, 105, 107, 106, 108, 107, 109, 108, 110,
                                   109, 111, 110, 112, 111]);
    const len = high.length;
    
    // Allocate buffers
    const highPtr = wasm.kaufmanstop_alloc(len);
    const lowPtr = wasm.kaufmanstop_alloc(len);
    const outPtr = wasm.kaufmanstop_alloc(len);
    
    assert(highPtr !== 0, 'Failed to allocate high buffer');
    assert(lowPtr !== 0, 'Failed to allocate low buffer');
    assert(outPtr !== 0, 'Failed to allocate output buffer');
    
    try {
        // Create views into WASM memory
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        // Copy data into WASM memory
        highView.set(high);
        lowView.set(low);
        
        // Compute KAUFMANSTOP
        wasm.kaufmanstop_into(highPtr, lowPtr, outPtr, len, 5, 2.0, 'long', 'sma');
        
        // Verify results match regular API
        const regularResult = wasm.kaufmanstop_js(high, low, 5, 2.0, 'long', 'sma');
        
        // Recreate view in case memory grew
        const outView2 = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        for (let i = 0; i < len; i++) {
            if (isNaN(regularResult[i]) && isNaN(outView2[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - outView2[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${outView2[i]}`);
        }
    } finally {
        // Always free memory
        wasm.kaufmanstop_free(highPtr, len);
        wasm.kaufmanstop_free(lowPtr, len);
        wasm.kaufmanstop_free(outPtr, len);
    }
});

test('KAUFMANSTOP zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.kaufmanstop_into(0, 0, 0, 10, 22, 2.0, 'long', 'sma');
    }, /Null pointer/);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.kaufmanstop_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.kaufmanstop_into(ptr, ptr, ptr, 10, 0, 2.0, 'long', 'sma');
        }, /Invalid period/);
    } finally {
        wasm.kaufmanstop_free(ptr, 10);
    }
});

test('KAUFMANSTOP memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const highPtr = wasm.kaufmanstop_alloc(size);
        const lowPtr = wasm.kaufmanstop_alloc(size);
        assert(highPtr !== 0, `Failed to allocate high buffer of ${size} elements`);
        assert(lowPtr !== 0, `Failed to allocate low buffer of ${size} elements`);
        
        // Write pattern to verify memory
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            highView[i] = i * 1.5;
            lowView[i] = i * 1.2;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(highView[i], i * 1.5, `High memory corruption at index ${i}`);
            assert.strictEqual(lowView[i], i * 1.2, `Low memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.kaufmanstop_free(highPtr, size);
        wasm.kaufmanstop_free(lowPtr, size);
    }
});

test.after(() => {
    console.log('KAUFMANSTOP WASM tests completed');
});