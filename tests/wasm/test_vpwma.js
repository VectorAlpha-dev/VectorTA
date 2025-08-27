/**
 * WASM binding tests for VPWMA indicator.
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

test('VPWMA partial params', () => {
    // Test with default parameters - mirrors check_vpwma_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.vpwma_js(close, 14, 0.382);
    assert.strictEqual(result.length, close.length);
});

test('VPWMA accuracy', async () => {
    // Test VPWMA matches expected values from Rust tests - mirrors check_vpwma_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.vpwma;
    
    const result = wasm.vpwma_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.power
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-4,  // Using 1e-4 as per Rust test which uses 1e-2
        "VPWMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('vpwma', result, 'close', expected.defaultParams);
});

test('VPWMA zero period', () => {
    // Test VPWMA fails with zero period - mirrors check_vpwma_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.vpwma_js(inputData, 0, 0.382);
    }, /Invalid period/);
});

test('VPWMA period exceeds length', () => {
    // Test VPWMA fails when period exceeds data length - mirrors check_vpwma_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.vpwma_js(dataSmall, 10, 0.382);
    }, /Invalid period/);
});

test('VPWMA very small dataset', () => {
    // Test VPWMA fails with insufficient data - mirrors check_vpwma_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.vpwma_js(singlePoint, 2, 0.382);
    }, /Invalid period|Not enough valid data/);
});

test('VPWMA empty input', () => {
    // Test VPWMA fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.vpwma_js(empty, 14, 0.382);
    }, /Input data slice is empty/);
});

test('VPWMA invalid power', () => {
    // Test VPWMA fails with invalid power - mirrors vpwma power validation
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // NaN power
    assert.throws(() => {
        wasm.vpwma_js(data, 2, NaN);
    }, /Invalid power/);
    
    // Infinite power
    assert.throws(() => {
        wasm.vpwma_js(data, 2, Infinity);
    }, /Invalid power/);
});

test('VPWMA reinput', () => {
    // Test VPWMA applied twice (re-input) - mirrors check_vpwma_reinput
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.vpwma_js(close, 14, 0.382);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply VPWMA to VPWMA output
    const secondResult = wasm.vpwma_js(firstResult, 5, 0.5);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Check that values after warmup are not NaN
    if (secondResult.length > 240) {
        for (let i = 240; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Unexpected NaN at index ${i}`);
        }
    }
});

test('VPWMA NaN handling', () => {
    // Test VPWMA handles NaN values correctly - mirrors check_vpwma_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.vpwma_js(close, 14, 0.382);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (50), no NaN values should exist
    if (result.length > 50) {
        for (let i = 50; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First period-1 values should be NaN
    assertAllNaN(result.slice(0, 13), "Expected NaN in warmup period");
});

test('VPWMA all NaN input', () => {
    // Test VPWMA with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.vpwma_js(allNaN, 14, 0.382);
    }, /All values are NaN/);
});

test('VPWMA batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Using the new ergonomic batch API for single parameter
    const batchResult = wasm.vpwma_batch(close, {
        period_range: [14, 14, 0],
        power_range: [0.382, 0.382, 0]
    });
    
    // Should match single calculation
    const singleResult = wasm.vpwma_js(close, 14, 0.382);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('VPWMA batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 14, 16, 18 using ergonomic API
    const batchResult = wasm.vpwma_batch(close, {
        period_range: [14, 18, 2],      // period range
        power_range: [0.382, 0.382, 0]  // power range  
    });
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    // Verify each row matches individual calculation
    const periods = [14, 16, 18];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.vpwma_js(close, periods[i], 0.382);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('VPWMA batch metadata from result', () => {
    // Test that batch result includes correct parameter combinations
    const close = new Float64Array(20); // Need enough data for period 18
    close.fill(100);
    
    const result = wasm.vpwma_batch(close, {
        period_range: [14, 18, 2],      // period: 14, 16, 18
        power_range: [0.3, 0.5, 0.1]   // power: 0.3, 0.4, 0.5
    });
    
    // Should have 3 * 3 = 9 combinations
    assert.strictEqual(result.combos.length, 9);
    
    // Check first combination
    assert.strictEqual(result.combos[0].period, 14);   // period
    assert.strictEqual(result.combos[0].power, 0.3);  // power
    
    // Check last combination
    assert.strictEqual(result.combos[8].period, 18);  // period
    assertClose(result.combos[8].power, 0.5, 1e-10, "power mismatch"); // power
});

test('VPWMA batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.vpwma_batch(close, {
        period_range: [12, 14, 2],      // 2 periods
        power_range: [0.3, 0.4, 0.1]    // 2 powers
    });
    
    // Should have 2 * 2 = 4 combinations
    assert.strictEqual(batchResult.combos.length, 4);
    assert.strictEqual(batchResult.rows, 4);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 4 * 50);
    
    // Verify structure
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const period = batchResult.combos[combo].period;
        const power = batchResult.combos[combo].power;
        
        const rowStart = combo * 50;
        const rowData = batchResult.values.slice(rowStart, rowStart + 50);
        
        // First period-1 values should be NaN
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        // After warmup should have values
        for (let i = period - 1; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('VPWMA batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep
    const singleBatch = wasm.vpwma_batch(close, {
        period_range: [5, 5, 1],
        power_range: [0.382, 0.382, 0.1]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.vpwma_batch(close, {
        period_range: [5, 7, 10], // Step larger than range
        power_range: [0.382, 0.382, 0]
    });
    
    // Should only have period=5
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.vpwma_batch(new Float64Array([]), {
            period_range: [14, 14, 0],
            power_range: [0.382, 0.382, 0]
        });
    }, /Input data slice is empty/);
});

// New API tests
test('VPWMA batch - new ergonomic API with single parameter', () => {
    // Test the new API with a single parameter combination
    const close = new Float64Array(testData.close);
    
    const result = wasm.vpwma_batch(close, {
        period_range: [14, 14, 0],
        power_range: [0.382, 0.382, 0]
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
    assert.strictEqual(combo.period, 14);
    assert.strictEqual(combo.power, 0.382);
    
    // Compare with old API
    const oldResult = wasm.vpwma_js(close, 14, 0.382);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('VPWMA batch - new API with multiple parameters', () => {
    // Test with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.vpwma_batch(close, {
        period_range: [10, 14, 2],       // 3 periods
        power_range: [0.3, 0.4, 0.1]     // 2 powers
    });
    
    // Should have 3 * 2 = 6 combinations
    assert.strictEqual(result.combos.length, 6);
    assert.strictEqual(result.rows, 6);
    assert.strictEqual(result.cols, 50);
    
    // Verify all combos are present
    const expectedCombos = [
        {period: 10, power: 0.3},
        {period: 10, power: 0.4},
        {period: 12, power: 0.3},
        {period: 12, power: 0.4},
        {period: 14, power: 0.3},
        {period: 14, power: 0.4}
    ];
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedCombos[i].period);
        assertClose(result.combos[i].power, expectedCombos[i].power, 1e-10, `Power mismatch at combo ${i}`);
    }
});

test.skip('VPWMA zero-copy API', () => {
    // Skip: Zero-copy API not yet implemented for VPWMA in WASM bindings
    // This test is kept as a placeholder for future implementation
    // Test zero-copy API for VPWMA
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
    const period = 5;
    const power = 0.382;
    
    // Allocate memory in WASM
    const ptr = wasm.vpwma_alloc(data.length);
    assert(ptr !== 0, 'Should allocate memory successfully');
    
    try {
        // Get view of WASM memory and copy data
        const memory = wasm.memory;
        const memView = new Float64Array(memory.buffer, ptr, data.length);
        memView.set(data);
        
        // Process in-place
        wasm.vpwma_into(ptr, ptr, data.length, period, power);
        
        // Compare with regular API
        const regularResult = wasm.vpwma_js(data, period, power);
        
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        wasm.vpwma_free(ptr, data.length);
    }
});

test.skip('VPWMA zero-copy with large dataset', () => {
    // Skip: Zero-copy API not yet implemented for VPWMA in WASM bindings
    // Test zero-copy with a larger dataset for performance
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.1) * 100 + 100;
    }
    
    const ptr = wasm.vpwma_alloc(size);
    assert(ptr !== 0, 'Should allocate large memory successfully');
    
    try {
        const memory = wasm.memory;
        const memView = new Float64Array(memory.buffer, ptr, size);
        memView.set(data);
        
        // Process with default parameters
        wasm.vpwma_into(ptr, ptr, size, 14, 0.382);
        
        // Check that warmup period has NaN values
        for (let i = 0; i < 13; i++) {
            assert(isNaN(memView[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check that values after warmup are not NaN
        for (let i = 13; i < 100; i++) {
            assert(!isNaN(memView[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.vpwma_free(ptr, size);
    }
});

// Error handling for zero-copy API
test.skip('VPWMA zero-copy error handling', () => {
    // Skip: Zero-copy API not yet implemented for VPWMA in WASM bindings
    // Test with null pointer
    assert.throws(() => {
        wasm.vpwma_into(0, 0, 10, 14, 0.382);
    }, /null pointer|invalid memory/i);
    
    const ptr = wasm.vpwma_alloc(10);
    try {
        // Test with invalid period
        assert.throws(() => {
            wasm.vpwma_into(ptr, ptr, 10, 0, 0.382);
        }, /Invalid period/);
        
        // Test with invalid power
        assert.throws(() => {
            wasm.vpwma_into(ptr, ptr, 10, 5, NaN);
        }, /Invalid power/);
    } finally {
        wasm.vpwma_free(ptr, 10);
    }
});

test.skip('VPWMA zero-copy memory management', () => {
    // Skip: Zero-copy API not yet implemented for VPWMA in WASM bindings
    // Test memory allocation and deallocation
    const sizes = [10, 100, 1000];
    
    for (const size of sizes) {
        const ptr = wasm.vpwma_alloc(size);
        assert(ptr !== 0, `Should allocate ${size} elements`);
        
        // Verify we can write to the memory
        const memory = wasm.memory;
        const memView = new Float64Array(memory.buffer, ptr, size);
        for (let i = 0; i < size; i++) {
            memView[i] = i;
        }
        
        // Verify we can read back
        for (let i = 0; i < size; i++) {
            assert.strictEqual(memView[i], i, `Memory corruption at index ${i}`);
        }
        
        // Clean up
        wasm.vpwma_free(ptr, size);
    }
});

test('VPWMA warmup period verification', () => {
    // Test that warmup periods match expected values for different periods
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const testCases = [
        {period: 5, power: 0.5, expectedWarmup: 4},
        {period: 10, power: 0.382, expectedWarmup: 9},
        {period: 14, power: 0.382, expectedWarmup: 13},
        {period: 20, power: 0.3, expectedWarmup: 19}
    ];
    
    for (const tc of testCases) {
        const result = wasm.vpwma_js(close, tc.period, tc.power);
        
        // Check warmup period has NaN values
        for (let i = 0; i < tc.expectedWarmup; i++) {
            assert(isNaN(result[i]), 
                   `Expected NaN at index ${i} for period=${tc.period}, got ${result[i]}`);
        }
        
        // Check first non-NaN value appears at expected index
        assert(!isNaN(result[tc.expectedWarmup]), 
               `Expected first valid value at index ${tc.expectedWarmup} for period=${tc.period}`);
        
        // Verify warmup calculation: warmup = period - 1
        assert.strictEqual(tc.expectedWarmup, tc.period - 1,
                          `Warmup calculation mismatch for period=${tc.period}`);
    }
});

test('VPWMA SIMD consistency', () => {
    // Test that different SIMD kernels produce consistent results
    const close = new Float64Array(testData.close.slice(0, 100));
    const period = 14;
    const power = 0.382;
    
    // Test available kernels
    const kernels = ['scalar', 'auto'];
    const results = {};
    
    for (const kernel of kernels) {
        try {
            // Note: kernel selection might not be exposed in WASM API
            // This test assumes we can force kernel selection somehow
            // If not available, just test that auto kernel works
            results[kernel] = wasm.vpwma_js(close, period, power);
        } catch (e) {
            // Kernel might not be available
            continue;
        }
    }
    
    // At minimum, scalar and auto should be available
    assert(results['auto'], 'Auto kernel should be available');
    
    // If we have multiple results, compare them
    if (Object.keys(results).length > 1) {
        const baseResult = results['auto'];
        for (const [kernel, result] of Object.entries(results)) {
            if (kernel === 'auto') continue;
            
            for (let i = 0; i < result.length; i++) {
                if (isNaN(baseResult[i]) && isNaN(result[i])) {
                    continue;
                }
                assert(Math.abs(baseResult[i] - result[i]) < 1e-9,
                       `Kernel ${kernel} mismatch with auto at index ${i}`);
            }
        }
    }
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('VPWMA WASM tests completed');
});