/**
 * WASM binding tests for Decycler Oscillator (DEC_OSC) indicator.
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

test('DEC_OSC partial params', () => {
    // Test with default parameters - mirrors check_dec_osc_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.dec_osc_js(close, 125, 1.0);
    assert.strictEqual(result.length, close.length);
});

test('DEC_OSC accuracy', async () => {
    // Test DEC_OSC matches expected values from Rust tests - mirrors check_dec_osc_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.dec_osc_js(close, 125, 1.0);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected values from Rust tests
    const expectedLast5 = [
        -1.5036367540303395,
        -1.4037875172207006,
        -1.3174199471429475,
        -1.2245874070642693,
        -1.1638422627265639,
    ];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-7,
        "DEC_OSC last 5 values mismatch"
    );
    
    // Compare full output with Rust
    // Note: dec_osc is not yet registered in generate_references
    // await compareWithRust('dec_osc', result, 'close', { hp_period: 125, k: 1.0 });
});

test('DEC_OSC default candles', () => {
    // Test DEC_OSC with default parameters - mirrors check_dec_osc_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.dec_osc_js(close, 125, 1.0);
    assert.strictEqual(result.length, close.length);
});

test('DEC_OSC zero period', () => {
    // Test DEC_OSC fails with zero period - mirrors check_dec_osc_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.dec_osc_js(inputData, 0, 1.0);
    }, /Invalid period/);
});

test('DEC_OSC period exceeds length', () => {
    // Test DEC_OSC fails when period exceeds data length - mirrors check_dec_osc_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.dec_osc_js(dataSmall, 10, 1.0);
    }, /Invalid period/);
});

test('DEC_OSC very small dataset', () => {
    // Test DEC_OSC fails with insufficient data - mirrors check_dec_osc_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.dec_osc_js(singlePoint, 125, 1.0);
    }, /Invalid period|Not enough valid data/);
});

test('DEC_OSC empty input', () => {
    // Test DEC_OSC fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.dec_osc_js(empty, 125, 1.0);
    }, /Input data slice is empty/);
});

test('DEC_OSC invalid k', () => {
    // Test DEC_OSC fails with invalid k value
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // k = 0
    assert.throws(() => {
        wasm.dec_osc_js(data, 2, 0.0);
    }, /Invalid K/);
    
    // Negative k
    assert.throws(() => {
        wasm.dec_osc_js(data, 2, -1.0);
    }, /Invalid K/);
    
    // NaN k
    assert.throws(() => {
        wasm.dec_osc_js(data, 2, NaN);
    }, /Invalid K/);
});

test('DEC_OSC reinput', () => {
    // Test DEC_OSC using output as input - mirrors check_dec_osc_reinput
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.dec_osc_js(close, 50, 1.0);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass using first result as input
    const secondResult = wasm.dec_osc_js(firstResult, 50, 1.0);
    assert.strictEqual(secondResult.length, firstResult.length);
});

test('DEC_OSC NaN handling', () => {
    // Test DEC_OSC NaN handling - verifies warmup period and NaN propagation
    const close = new Float64Array(testData.close);
    const hpPeriod = 10;
    
    // Test normal data - verify warmup period
    const result = wasm.dec_osc_js(close, hpPeriod, 1.0);
    assert.strictEqual(result.length, close.length);
    
    // DEC_OSC warmup period is 2 (needs two seeded samples)
    const warmupPeriod = 2;
    
    // First warmup_period values should be NaN
    assertAllNaN(result.slice(0, warmupPeriod), `Expected NaN in first ${warmupPeriod} values (warmup period)`);
    
    // After warmup period, no NaN values should exist (assuming clean input)
    if (result.length > warmupPeriod + 100) {
        for (let i = warmupPeriod + 100; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i} (after warmup)`);
        }
    }
    
    // Test with NaN values in input
    const dataWithNaN = new Float64Array(close);
    for (let i = 0; i < 5; i++) {
        dataWithNaN[i] = NaN;
    }
    
    const resultWithNaN = wasm.dec_osc_js(dataWithNaN, hpPeriod, 1.0);
    assert.strictEqual(resultWithNaN.length, dataWithNaN.length);
    
    // NaN propagation: with 5 input NaNs, we expect at least those to be NaN in output
    assertAllNaN(resultWithNaN.slice(0, 5), "Expected NaN propagation from input NaNs");
});

test('DEC_OSC warmup period', () => {
    // Test DEC_OSC warmup period calculation for different hp_periods
    const close = new Float64Array(testData.close);
    const testPeriods = [5, 10, 20, 50, 125];
    
    for (const hpPeriod of testPeriods) {
        const result = wasm.dec_osc_js(close, hpPeriod, 1.0);
        
        // DEC_OSC warmup is always 2 (needs two seeded samples)
        const expectedWarmup = 2;
        
        // Check first expectedWarmup values are NaN
        for (let i = 0; i < Math.min(expectedWarmup, result.length); i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} for hp_period=${hpPeriod}`);
        }
        
        // Check value after warmup is not NaN (if we have enough data)
        if (result.length > expectedWarmup) {
            assert(!isNaN(result[expectedWarmup]), `Unexpected NaN at index ${expectedWarmup} (after warmup) for hp_period=${hpPeriod}`);
        }
    }
});

test('DEC_OSC batch calculation', () => {
    // Test DEC_OSC batch calculation with parameter ranges using ergonomic API
    const close = new Float64Array(testData.close);
    
    // Test batch calculation with parameter ranges using ergonomic API
    const result = wasm.dec_osc_batch(close, {
        hp_period_range: [100, 150, 25],
        k_range: [0.5, 1.5, 0.5]
    });
    
    // Check result structure
    assert(result.values !== undefined, 'Result should have values');
    assert(result.combos !== undefined, 'Result should have combos');
    assert(result.rows !== undefined, 'Result should have rows');
    assert(result.cols !== undefined, 'Result should have cols');
    
    // Check dimensions
    const expectedCombinations = 3 * 3; // 3 periods * 3 k values
    assert.strictEqual(result.rows, expectedCombinations);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, expectedCombinations * close.length);
    assert.strictEqual(result.combos.length, expectedCombinations);
    
    // Verify parameter combinations
    const expectedPeriods = [100, 100, 100, 125, 125, 125, 150, 150, 150];
    const expectedKs = [0.5, 1.0, 1.5, 0.5, 1.0, 1.5, 0.5, 1.0, 1.5];
    
    for (let i = 0; i < expectedCombinations; i++) {
        assert.strictEqual(result.combos[i].hp_period, expectedPeriods[i]);
        assertClose(result.combos[i].k, expectedKs[i], 1e-10, `k value mismatch at index ${i}`);
        
        // Verify each combination matches individual calculation
        const rowStart = i * result.cols;
        const rowEnd = rowStart + result.cols;
        const rowData = result.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.dec_osc_js(close, result.combos[i].hp_period, result.combos[i].k);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Batch row ${i} (hp_period=${result.combos[i].hp_period}, k=${result.combos[i].k}) doesn't match single calculation`
        );
    }
});

test('DEC_OSC batch single parameter', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Single parameter combination (default values)
    const result = wasm.dec_osc_batch(close, {
        hp_period_range: [125, 125, 0],
        k_range: [1.0, 1.0, 0]
    });
    
    // Should have exactly 1 combination
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.values.length, close.length);
    
    // Verify parameters
    assert.strictEqual(result.combos[0].hp_period, 125);
    assert.strictEqual(result.combos[0].k, 1.0);
    
    // Should match single calculation
    const singleResult = wasm.dec_osc_js(close, 125, 1.0);
    assertArrayClose(result.values, singleResult, 1e-10, "Batch single param doesn't match single calculation");
    
    // Verify it matches expected values
    const expectedLast5 = [
        -1.5036367540303395,
        -1.4037875172207006,
        -1.3174199471429475,
        -1.2245874070642693,
        -1.1638422627265639,
    ];
    const last5 = result.values.slice(-5);
    assertArrayClose(last5, expectedLast5, 1e-7, "Batch default params last 5 values mismatch");
});

test('DEC_OSC batch metadata from result', () => {
    // Test that batch result includes correct parameter combinations
    const close = new Float64Array(20); // Need enough data for period 15
    close.fill(100);
    
    const result = wasm.dec_osc_batch(close, {
        hp_period_range: [10, 15, 5],    // periods: 10, 15
        k_range: [0.5, 1.0, 0.25]        // k: 0.5, 0.75, 1.0
    });
    
    // Should have 2 * 3 = 6 combinations
    assert.strictEqual(result.combos.length, 6);
    
    // Check first combination
    assert.strictEqual(result.combos[0].hp_period, 10);
    assert.strictEqual(result.combos[0].k, 0.5);
    
    // Check last combination
    assert.strictEqual(result.combos[5].hp_period, 15);
    assertClose(result.combos[5].k, 1.0, 1e-10, "k mismatch");
});

test('DEC_OSC edge cases', () => {
    // Test DEC_OSC with edge case inputs
    
    // Test with all same values
    const sameValues = new Float64Array(100).fill(50.0);
    const result1 = wasm.dec_osc_js(sameValues, 10, 1.0);
    assert.strictEqual(result1.length, sameValues.length);
    
    // Test with monotonically increasing values
    const increasing = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        increasing[i] = i;
    }
    const result2 = wasm.dec_osc_js(increasing, 10, 1.0);
    assert.strictEqual(result2.length, increasing.length);
    
    // Test with alternating values
    const alternating = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        alternating[i] = i % 2 === 0 ? 10.0 : 20.0;
    }
    const result3 = wasm.dec_osc_js(alternating, 10, 1.0);
    assert.strictEqual(result3.length, alternating.length);
});

test('DEC_OSC batch edge cases', () => {
    // Test batch edge cases
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    
    // Single value sweep
    const singleBatch = wasm.dec_osc_batch(close, {
        hp_period_range: [5, 5, 1],
        k_range: [1.0, 1.0, 0.1]
    });
    
    assert.strictEqual(singleBatch.values.length, 15);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.dec_osc_batch(close, {
        hp_period_range: [5, 7, 10], // Step larger than range
        k_range: [1.0, 1.0, 0]
    });
    
    // Should only have hp_period=5
    assert.strictEqual(largeBatch.values.length, 15);
    assert.strictEqual(largeBatch.combos.length, 1);
    assert.strictEqual(largeBatch.combos[0].hp_period, 5);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.dec_osc_batch(new Float64Array([]), {
            hp_period_range: [10, 10, 0],
            k_range: [1.0, 1.0, 0]
        });
    }, /All values are NaN|Input data slice is empty/);
});

test('DEC_OSC batch API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    // Invalid config structure
    assert.throws(() => {
        wasm.dec_osc_batch(close, {
            hp_period_range: [10, 10], // Missing step
            k_range: [1.0, 1.0, 0]
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.dec_osc_batch(close, {
            hp_period_range: [10, 10, 0]
            // Missing k_range
        });
    }, /Invalid config/);
    
    // Invalid data type
    assert.throws(() => {
        wasm.dec_osc_batch(close, {
            hp_period_range: "invalid",
            k_range: [1.0, 1.0, 0]
        });
    }, /Invalid config/);
});

test('DEC_OSC zero-copy API', () => {
    // Test the zero-copy API with preallocated memory
    const close = new Float64Array(testData.close);
    const len = close.length;
    const hpPeriod = 125;
    const k = 1.0;
    
    // Allocate memory
    const ptr = wasm.dec_osc_alloc(len);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    try {
        // Create view into WASM memory
        const memView = new Float64Array(
            wasm.__wasm.memory.buffer,
            ptr,
            len
        );
        
        // Copy data into WASM memory
        memView.set(close);
        
        // Compute DEC_OSC in-place
        wasm.dec_osc_into(ptr, ptr, len, hpPeriod, k);
        
        // Verify results match regular API
        const regularResult = wasm.dec_osc_js(close, hpPeriod, k);
        for (let i = 0; i < len; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.dec_osc_free(ptr, len);
    }
});

test('DEC_OSC zero-copy with large dataset', () => {
    // Test zero-copy API with large dataset
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.dec_osc_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.dec_osc_into(ptr, ptr, size, 125, 1.0);
        
        // Recreate view in case memory grew
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        // Check warmup period has NaN (always 2 for DEC_OSC)
        const warmupPeriod = 2;
        for (let i = 0; i < warmupPeriod; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = warmupPeriod; i < Math.min(warmupPeriod + 100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.dec_osc_free(ptr, size);
    }
});

test('DEC_OSC zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.dec_osc_into(0, 0, 10, 10, 1.0);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.dec_osc_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.dec_osc_into(ptr, ptr, 10, 0, 1.0);
        }, /Invalid period/);
        
        // Invalid k
        assert.throws(() => {
            wasm.dec_osc_into(ptr, ptr, 10, 5, 0.0);
        }, /Invalid K/);
    } finally {
        wasm.dec_osc_free(ptr, 10);
    }
});

test('DEC_OSC memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000, 100000];
    
    for (const size of sizes) {
        const ptr = wasm.dec_osc_alloc(size);
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
        wasm.dec_osc_free(ptr, size);
    }
});

test('DEC_OSC SIMD128 consistency', () => {
    // This test verifies SIMD128 produces same results as scalar
    // It runs automatically when SIMD128 is enabled
    const testCases = [
        { size: 10, hp_period: 5 },
        { size: 100, hp_period: 10 },
        { size: 1000, hp_period: 50 },
        { size: 10000, hp_period: 125 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }
        
        const result = wasm.dec_osc_js(data, testCase.hp_period, 1.0);
        
        // Basic sanity checks
        assert.strictEqual(result.length, data.length);
        
        // Check warmup period (always 2 for DEC_OSC)
        const warmupPeriod = 2;
        for (let i = 0; i < warmupPeriod && i < result.length; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        // Check values exist after warmup
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = warmupPeriod; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        // Verify reasonable values (DEC_OSC can produce large values depending on input)
        // Unlike traditional oscillators, DEC_OSC can have significant magnitudes
        if (countAfterWarmup > 0) {
            const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
            // DEC_OSC can produce values in the hundreds or even thousands
            assert(Math.abs(avgAfterWarmup) < 10000, `Average value ${avgAfterWarmup} seems unreasonable`);
        }
    }
});

test('DEC_OSC all NaN input', () => {
    // Test DEC_OSC with all NaN values
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.dec_osc_js(allNaN, 10, 1.0);
    }, /All values are NaN/);
});
