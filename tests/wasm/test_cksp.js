/**
 * WASM binding tests for CKSP indicator.
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

test('CKSP partial params', () => {
    // Test with default parameters - mirrors check_cksp_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.cksp_js(high, low, close, 10, 1.0, 9);
    assert.strictEqual(result.length, close.length * 2);
});

test('CKSP accuracy', async () => {
    // Test CKSP matches expected values from Rust tests - mirrors check_cksp_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const expected = EXPECTED_OUTPUTS['cksp'];
    const params = expected['default_params'];
    
    // Run CKSP with default parameters
    const result = wasm.cksp_js(high, low, close, params.p, params.x, params.q);
    
    // Result should have 2x the length (long_values, short_values)
    assert.strictEqual(result.length, close.length * 2);
    
    // Split the result into long and short arrays
    const len = close.length;
    const longResult = result.slice(0, len);
    const shortResult = result.slice(len);
    
    // Check last 5 values match expected for long
    const expectedLongLast5 = [
        60306.66197802568,
        60306.66197802568,
        60306.66197802568,
        60203.29578022311,
        60201.57958198072,
    ];
    const longLast5 = longResult.slice(-5);
    expectedLongLast5.forEach((expectedVal, i) => {
        assertClose(longLast5[i], expectedVal, 1e-5, `CKSP long value mismatch at index ${i}`);
    });
    
    // Check last 5 values match expected for short
    const expectedShortLast5 = [
        58757.826484736055,
        58701.74383626245,
        58656.36945263621,
        58611.03250737258,
        58611.03250737258,
    ];
    const shortLast5 = shortResult.slice(-5);
    expectedShortLast5.forEach((expectedVal, i) => {
        assertClose(shortLast5[i], expectedVal, 1e-5, `CKSP short value mismatch at index ${i}`);
    });
    
    // Compare full output with Rust
    // NOTE: CKSP not yet added to generate_references binary
    // await compareWithRust('cksp_long', longResult, 'hlc', params);
    // await compareWithRust('cksp_short', shortResult, 'hlc', params);
});

test('CKSP default candles', () => {
    // Test CKSP with default parameters - mirrors check_cksp_default_candles
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.cksp_js(high, low, close, 10, 1.0, 9);
    assert.strictEqual(result.length, close.length * 2);
});

test('CKSP zero period', () => {
    // Test CKSP fails with zero period - mirrors check_cksp_zero_period
    const high = new Float64Array([10, 11, 12]);
    const low = new Float64Array([9, 10, 10.5]);
    const close = new Float64Array([9.5, 10.5, 11]);
    
    assert.throws(() => {
        wasm.cksp_js(high, low, close, 0, 1.0, 9);
    }, /Invalid param/);
});

test('CKSP invalid x', () => {
    // Test CKSP fails with invalid x multiplier
    const high = new Float64Array([10, 11, 12]);
    const low = new Float64Array([9, 10, 10.5]);
    const close = new Float64Array([9.5, 10.5, 11]);
    
    // Test with NaN x
    assert.throws(() => {
        wasm.cksp_js(high, low, close, 10, NaN, 9);
    }, /Invalid param/);
    
    // Test with infinite x
    assert.throws(() => {
        wasm.cksp_js(high, low, close, 10, Infinity, 9);
    }, /Invalid param/);
});

test('CKSP invalid q', () => {
    // Test CKSP fails with invalid q parameter
    const high = new Float64Array([10, 11, 12]);
    const low = new Float64Array([9, 10, 10.5]);
    const close = new Float64Array([9.5, 10.5, 11]);
    
    assert.throws(() => {
        wasm.cksp_js(high, low, close, 10, 1.0, 0);
    }, /Invalid param/);
});

test('CKSP period exceeds length', () => {
    // Test CKSP fails when period exceeds data length - mirrors check_cksp_period_exceeds_length
    const high = new Float64Array([10, 11, 12]);
    const low = new Float64Array([9, 10, 10.5]);
    const close = new Float64Array([9.5, 10.5, 11]);
    
    assert.throws(() => {
        wasm.cksp_js(high, low, close, 10, 1.0, 9);
    }, /Not enough data/);
});

test('CKSP very small dataset', () => {
    // Test CKSP with single data point - mirrors check_cksp_very_small_dataset
    const high = new Float64Array([42]);
    const low = new Float64Array([41]);
    const close = new Float64Array([41.5]);
    
    assert.throws(() => {
        wasm.cksp_js(high, low, close, 10, 1.0, 9);
    }, /Not enough data/);
});

test('CKSP all NaN input', () => {
    // Test CKSP with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.cksp_js(allNaN, allNaN, allNaN, 10, 1.0, 9);
    }, /Data is empty|No data/);
});

test('CKSP error handling', () => {
    // Test error handling - additional edge cases
    // Test with empty data
    assert.throws(() => {
        wasm.cksp_js(new Float64Array(), new Float64Array(), new Float64Array(), 10, 1.0, 9);
    }, /Data is empty/);
    
    // Test with inconsistent lengths
    assert.throws(() => {
        wasm.cksp_js(
            new Float64Array([1, 2, 3]),
            new Float64Array([1, 2]),  // Wrong length
            new Float64Array([1, 2, 3]),
            10, 1.0, 9
        );
    }, /Inconsistent/);
    
    // Test with zero period
    const high = new Float64Array([10, 11, 12]);
    const low = new Float64Array([9, 10, 10.5]);
    const close = new Float64Array([9.5, 10.5, 11]);
    assert.throws(() => {
        wasm.cksp_js(high, low, close, 0, 1.0, 9);
    }, /Invalid param/);
    
});

test('CKSP nan handling', () => {
    // Test CKSP handles NaN values correctly - mirrors check_cksp_nan_handling
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const expected = EXPECTED_OUTPUTS['cksp'];
    const params = expected['default_params'];
    
    const result = wasm.cksp_js(high, low, close, params.p, params.x, params.q);
    
    // Split the result
    const len = close.length;
    const longResult = result.slice(0, len);
    const shortResult = result.slice(len);
    
    // Check warmup period has NaN values (p + q - 1 = 10 + 9 - 1 = 18)
    const warmupPeriod = params.p + params.q - 1;
    assert.strictEqual(warmupPeriod, 18, 'Expected warmup period of 18');
    for (let i = 0; i < warmupPeriod && i < longResult.length; i++) {
        assert(isNaN(longResult[i]), `Expected NaN in long warmup at index ${i}`);
        assert(isNaN(shortResult[i]), `Expected NaN in short warmup at index ${i}`);
    }
    
    // After warmup, no NaN values should exist (if we have enough data)
    if (longResult.length > 240) {
        for (let i = 240; i < longResult.length; i++) {
            assert(!isNaN(longResult[i]), `Found unexpected NaN in long at index ${i}`);
            assert(!isNaN(shortResult[i]), `Found unexpected NaN in short at index ${i}`);
        }
    }
});

test('CKSP fast API (cksp_into)', () => {
    // Test fast API with pre-allocated memory
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const len = close.length;
    const expected = EXPECTED_OUTPUTS['cksp'];
    const params = expected['default_params'];
    
    // Allocate output buffers
    const longPtr = wasm.cksp_alloc(len);
    const shortPtr = wasm.cksp_alloc(len);
    
    try {
        // Call the fast API with TypedArrays (not ArrayBuffers)
        wasm.cksp_into(
            high,
            low,
            close,
            longPtr,
            shortPtr,
            len,
            params.p,
            params.x,
            params.q
        );
        
        // Read results from memory and copy them
        const longView = new Float64Array(wasm.__wasm.memory.buffer, longPtr, len);
        const shortView = new Float64Array(wasm.__wasm.memory.buffer, shortPtr, len);
        
        // Copy the data before any other operations that might invalidate the views
        const longResult = Array.from(longView);
        const shortResult = Array.from(shortView);
        
        // Verify results match the safe API
        const safeResult = wasm.cksp_js(high, low, close, params.p, params.x, params.q);
        const safeLong = safeResult.slice(0, len);
        const safeShort = safeResult.slice(len);
        
        assertArrayClose(
            longResult,
            Array.from(safeLong),
            1e-10,
            'Fast API long values should match safe API'
        );
        
        assertArrayClose(
            shortResult,
            Array.from(safeShort),
            1e-10,
            'Fast API short values should match safe API'
        );
        
    } finally {
        // Clean up allocated memory
        wasm.cksp_free(longPtr, len);
        wasm.cksp_free(shortPtr, len);
    }
});

test('CKSP batch processing', async () => {
    // Test batch processing - mirrors batch tests
    const high = new Float64Array(testData.high.slice(0, 100)); // Use smaller dataset
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Test batch with single parameter set
    const config = {
        p_range: [10, 10, 0],
        x_range: [1.0, 1.0, 0.0],
        q_range: [9, 9, 0]
    };
    
    const result = wasm.cksp_batch(high, low, close, config);
    
    // Check result structure
    assert(result.long_values, 'Missing long_values in batch result');
    assert(result.short_values, 'Missing short_values in batch result');
    assert(result.combos, 'Missing combos in batch result');
    assert.strictEqual(result.rows, 1, 'Expected 1 row for single parameter set');
    assert.strictEqual(result.cols, 100, 'Expected 100 columns');
    
    // Test with multiple parameter sets
    const multiConfig = {
        p_range: [5, 15, 5],  // 5, 10, 15
        x_range: [0.5, 1.5, 0.5],  // 0.5, 1.0, 1.5
        q_range: [5, 10, 5]  // 5, 10
    };
    
    const multiResult = wasm.cksp_batch(high, low, close, multiConfig);
    
    // Should have 3 * 3 * 2 = 18 combinations
    assert.strictEqual(multiResult.rows, 18, 'Expected 18 parameter combinations');
    assert.strictEqual(multiResult.combos.length, 18, 'Expected 18 combo entries');
    assert.strictEqual(multiResult.long_values.length, 18 * 100, 'Expected correct long_values length');
    assert.strictEqual(multiResult.short_values.length, 18 * 100, 'Expected correct short_values length');
    
    // Verify first combination
    assert.strictEqual(multiResult.combos[0].p, 5);
    assert.strictEqual(multiResult.combos[0].x, 0.5);
    assert.strictEqual(multiResult.combos[0].q, 5);
    
    // Verify last combination
    assert.strictEqual(multiResult.combos[17].p, 15);
    assert.strictEqual(multiResult.combos[17].x, 1.5);
    assert.strictEqual(multiResult.combos[17].q, 10);
    
    // Verify warmup periods for different combinations
    for (let i = 0; i < multiResult.combos.length; i++) {
        const combo = multiResult.combos[i];
        const expectedWarmup = combo.p + combo.q - 1;
        const rowStart = i * 100;
        const rowData = multiResult.long_values.slice(rowStart, rowStart + 100);
        
        // Count NaN values at the start
        let nanCount = 0;
        for (const val of rowData) {
            if (isNaN(val)) {
                nanCount++;
            } else {
                break;
            }
        }
        
        // Warmup period should match expected
        assert(nanCount >= Math.min(expectedWarmup, 100), 
               `Row ${i}: Expected at least ${expectedWarmup} NaN values, got ${nanCount}`);
    }
});

test('CKSP batch metadata', () => {
    // Test batch metadata and edge cases
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    // Test with single value sweep
    const singleConfig = {
        p_range: [10, 10, 1],
        x_range: [1.0, 1.0, 0.1],
        q_range: [9, 9, 1]
    };
    
    const singleResult = wasm.cksp_batch(high, low, close, singleConfig);
    assert.strictEqual(singleResult.rows, 1, 'Single sweep should have 1 row');
    assert.strictEqual(singleResult.combos.length, 1, 'Single sweep should have 1 combo');
    
    // Test with step larger than range
    const largeStepConfig = {
        p_range: [10, 12, 10],  // Step larger than range
        x_range: [1.0, 1.0, 0],
        q_range: [9, 9, 0]
    };
    
    const largeStepResult = wasm.cksp_batch(high, low, close, largeStepConfig);
    assert.strictEqual(largeStepResult.rows, 1, 'Large step should only have 1 row');
    assert.strictEqual(largeStepResult.combos[0].p, 10, 'Should only have p=10');
});

test.after(() => {
    console.log('CKSP WASM tests completed');
});
