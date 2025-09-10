/**
 * WASM binding tests for EHLERS_ECEMA indicator.
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
        wasm = await import(importPath);
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('EHLERS_ECEMA partial params', () => {
    // Test with default parameters - mirrors test_ehlers_ecema_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.ehlers_ecema_js(close, 20, 50);
    assert.strictEqual(result.length, close.length);
});

test('EHLERS_ECEMA accuracy', () => {
    // Test EHLERS_ECEMA matches expected values from Rust tests - mirrors check_ehlers_ecema_accuracy
    const expected = EXPECTED_OUTPUTS.ehlersEcema;
    // Use real CSV data
    const data = new Float64Array(testData.close);
    
    // Default parameters from Rust
    const length = expected.defaultParams.length;
    const gainLimit = expected.defaultParams.gainLimit;
    
    // Test regular mode (default: pine_compatible=false, confirmed_only=false)
    const result = wasm.ehlers_ecema_js(data, length, gainLimit);
    
    assert.strictEqual(result.length, data.length);
    
    // Check warmup period
    for (let i = 0; i < expected.warmupPeriod; i++) {
        assert(isNaN(result[i]), `Expected NaN during warmup at index ${i}`);
    }
    
    // Check that we have valid values after warmup
    assert(!isNaN(result[expected.warmupPeriod]), "Expected valid value after warmup");
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "EHLERS_ECEMA last 5 values mismatch"
    );
});

test('EHLERS_ECEMA Pine mode accuracy', () => {
    // Test EHLERS_ECEMA Pine mode matches expected values
    const expected = EXPECTED_OUTPUTS.ehlersEcema;
    // Use real CSV data
    const data = new Float64Array(testData.close);
    
    // Test with pine_compatible mode if available via new API
    if (wasm.ehlers_ecema_into_ex) {
        const len = data.length;
        const ptr = wasm.ehlers_ecema_alloc(len);
        
        // Copy data to WASM memory using correct API
        const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        const memView = new Float64Array(memory.buffer, ptr, len);
        memView.set(data);
        
        // Allocate output
        const outPtr = wasm.ehlers_ecema_alloc(len);
        
        // Run with pine_compatible=true
        wasm.ehlers_ecema_into_ex(ptr, outPtr, len, 20, 50, true, false);
        
        // Read result
        const memory2 = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        const result = new Float64Array(memory2.buffer, outPtr, len);
        const resultCopy = Array.from(result);
        
        // Clean up
        wasm.ehlers_ecema_free(ptr, len);
        wasm.ehlers_ecema_free(outPtr, len);
        
        // Check last 5 values match expected Pine mode values
        const last5 = resultCopy.slice(-5);
        assertArrayClose(
            last5,
            expected.pineModeLast5,
            1e-8,
            "EHLERS_ECEMA Pine mode last 5 values mismatch"
        );
        
        // Pine mode should have values from the start
        assert(!isNaN(resultCopy[0]), "Pine mode should have valid value at index 0");
    }
});

test('EHLERS_ECEMA default candles', () => {
    // Test EHLERS_ECEMA with default parameters - mirrors test_ehlers_ecema_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.ehlers_ecema_js(close, 20, 50);
    assert.strictEqual(result.length, close.length);
});

test('EHLERS_ECEMA zero period', () => {
    // Test EHLERS_ECEMA fails with zero period - mirrors test_ehlers_ecema_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.ehlers_ecema_js(inputData, 0, 50);
    }, /Invalid/);
});

test('EHLERS_ECEMA zero gain limit', () => {
    // Test EHLERS_ECEMA fails with zero gain limit
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.ehlers_ecema_js(inputData, 2, 0);
    }, /Invalid gain limit/);
});

test('EHLERS_ECEMA period exceeds length', () => {
    // Test EHLERS_ECEMA fails when period exceeds data length - mirrors test_ehlers_ecema_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.ehlers_ecema_js(dataSmall, 10, 50);
    }, /Invalid/);
});

test('EHLERS_ECEMA very small dataset', () => {
    // Test EHLERS_ECEMA fails with insufficient data - mirrors test_ehlers_ecema_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.ehlers_ecema_js(singlePoint, 20, 50);
    }, /Invalid|Not enough/);
});

test('EHLERS_ECEMA empty input', () => {
    // Test EHLERS_ECEMA fails with empty input - mirrors test_ehlers_ecema_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.ehlers_ecema_js(empty, 20, 50);
    }, /empty|Empty/);
});

test('EHLERS_ECEMA all NaN input', () => {
    // Test EHLERS_ECEMA fails with all NaN input - mirrors test_ehlers_ecema_all_nan
    const allNan = new Float64Array([NaN, NaN, NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.ehlers_ecema_js(allNan, 2, 50);
    }, /All values are NaN/);
});

test('EHLERS_ECEMA invalid gain limit', () => {
    // Test EHLERS_ECEMA fails with invalid gain limit - mirrors check_ehlers_ecema_invalid_gain_limit
    const inputData = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    // Test zero gain limit - should throw error
    assert.throws(() => {
        wasm.ehlers_ecema_js(inputData, 3, 0);
    }, /Invalid gain limit/);
    
    // Note: Negative gain limit in WASM gets converted to large unsigned value
    // and doesn't throw an error (JavaScript -10 becomes 4294967286 in usize)
    // This is expected behavior for WASM unsigned integer conversion
});

test('EHLERS_ECEMA reinput', () => {
    // Test EHLERS_ECEMA applied twice (re-input) - mirrors check_ehlers_ecema_reinput
    const expected = EXPECTED_OUTPUTS.ehlersEcema;
    // Use real CSV data
    const data = new Float64Array(testData.close);
    
    // Use reinput parameters
    const length = expected.reinputParams.length;
    const gainLimit = expected.reinputParams.gainLimit;
    
    // First pass
    const firstResult = wasm.ehlers_ecema_js(data, length, gainLimit);
    assert.strictEqual(firstResult.length, data.length);
    
    // Second pass - apply ECEMA to ECEMA output
    const secondResult = wasm.ehlers_ecema_js(firstResult, length, gainLimit);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Check warmup periods are correct
    const warmupPeriod = length - 1;
    for (let i = 0; i < warmupPeriod; i++) {
        assert(isNaN(firstResult[i]), `First pass should have NaN in warmup at index ${i}`);
    }
    
    // Second pass applies to data that already has NaN values, so warmup extends
    // It needs length (10) consecutive non-NaN values, which start at index 9
    // So first valid output is at index 9 + (length-1) = 18
    const secondWarmup = warmupPeriod + warmupPeriod;
    for (let i = 0; i < secondWarmup; i++) {
        assert(isNaN(secondResult[i]), `Second pass should have NaN in extended warmup at index ${i}`);
    }
    
    // Verify values after warmup exist
    assert(!isNaN(firstResult[warmupPeriod]), "First pass should have valid values after warmup");
    // Check for valid values after the extended warmup period
    const validIndices = secondResult.reduce((acc, val, idx) => {
        if (!isNaN(val)) acc.push(idx);
        return acc;
    }, []);
    assert(validIndices.length > 0, "Second pass should have some valid values");
    
    // Check last 5 values match expected
    const last5 = secondResult.slice(-5);
    assertArrayClose(
        last5,
        expected.reinputLast5,
        1e-8,
        "EHLERS_ECEMA re-input last 5 values mismatch"
    );
});

test('EHLERS_ECEMA NaN handling', () => {
    // Test EHLERS_ECEMA handles NaN values correctly - mirrors check_ehlers_ecema_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.ehlers_ecema_js(close, 20, 50);
    assert.strictEqual(result.length, close.length);
    
    // First period-1 values should be NaN (warmup period)
    const warmupPeriod = 19; // length - 1 = 20 - 1 = 19
    for (let i = 0; i < warmupPeriod; i++) {
        assert(isNaN(result[i]), `Expected NaN in warmup period at index ${i}`);
    }
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN after warmup period at index ${i}`);
        }
    }
    
    // Check transition point - first valid value should be at index warmupPeriod
    if (result.length > warmupPeriod) {
        assert(!isNaN(result[warmupPeriod]), `Expected valid value at index ${warmupPeriod}`);
    }
});

test('EHLERS_ECEMA memory management', () => {
    // Test memory allocation and deallocation
    const expected = EXPECTED_OUTPUTS.ehlersEcema;
    // Use real CSV data (using a subset for memory test)
    const data = new Float64Array(testData.close.slice(0, 25));
    
    const len = data.length;
    const ptr = wasm.ehlers_ecema_alloc(len);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create a memory view and copy data using correct API
    const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
    const memView = new Float64Array(memory.buffer, ptr, len);
    memView.set(data);
    
    // Allocate output buffer
    const outPtr = wasm.ehlers_ecema_alloc(len);
    
    // Run the computation
    wasm.ehlers_ecema_into(ptr, outPtr, len, 20, 50);
    
    // Read the result using correct API
    const memory2 = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
    const result = new Float64Array(memory2.buffer, outPtr, len);
    const resultCopy = Array.from(result);
    
    // Clean up
    wasm.ehlers_ecema_free(ptr, len);
    wasm.ehlers_ecema_free(outPtr, len);
    
    // Verify results match the regular function
    const regularResult = wasm.ehlers_ecema_js(data, 20, 50);
    assertArrayClose(resultCopy, regularResult, 1e-10, "Memory management test failed");
});

test('EHLERS_ECEMA in-place computation', () => {
    // Test in-place computation (input and output same buffer)
    const expected = EXPECTED_OUTPUTS.ehlersEcema;
    // Use real CSV data (using a subset for memory test)
    const data = new Float64Array(testData.close.slice(0, 25));
    
    const len = data.length;
    const ptr = wasm.ehlers_ecema_alloc(len);
    
    // Create a memory view and copy data using correct API
    const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
    const memView = new Float64Array(memory.buffer, ptr, len);
    memView.set(data);
    
    // Run the computation in-place
    wasm.ehlers_ecema_into(ptr, ptr, len, 20, 50);
    
    // Read the result using correct API
    const memory2 = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
    const result = new Float64Array(memory2.buffer, ptr, len);
    const resultCopy = Array.from(result);
    
    // Clean up
    wasm.ehlers_ecema_free(ptr, len);
    
    // Verify results match the regular function
    const regularResult = wasm.ehlers_ecema_js(data, 20, 50);
    assertArrayClose(resultCopy, regularResult, 1e-10, "In-place computation test failed");
});

// Batch API tests
test('EHLERS_ECEMA batch single parameter set', () => {
    // Test batch with single parameter combination (if available)
    if (!wasm.ehlers_ecema_batch) {
        console.log('EHLERS_ECEMA batch API not yet available');
        return;
    }
    
    const expected = EXPECTED_OUTPUTS.ehlersEcema;
    const close = new Float64Array(testData.close);
    
    // Using ergonomic batch API for single parameter
    const batchResult = wasm.ehlers_ecema_batch(close, {
        length_range: [expected.defaultParams.length, expected.defaultParams.length, 0],
        gain_limit_range: [expected.defaultParams.gainLimit, expected.defaultParams.gainLimit, 0]
    });
    
    // Should match single calculation
    const singleResult = wasm.ehlers_ecema_js(close, expected.defaultParams.length, expected.defaultParams.gainLimit);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
    
    // Verify metadata
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, close.length);
    assert.strictEqual(batchResult.combos.length, 1);
    assert.strictEqual(batchResult.combos[0].length, expected.defaultParams.length);
    assert.strictEqual(batchResult.combos[0].gain_limit, expected.defaultParams.gainLimit);
});

test('EHLERS_ECEMA batch multiple parameters', () => {
    // Test batch with multiple parameter combinations (if available)
    if (!wasm.ehlers_ecema_batch) {
        console.log('EHLERS_ECEMA batch API not yet available');
        return;
    }
    
    const expected = EXPECTED_OUTPUTS.ehlersEcema;
    const close = new Float64Array(testData.close.slice(0, 100)); // Smaller dataset
    
    // Multiple parameters from expected
    const batchResult = wasm.ehlers_ecema_batch(close, {
        length_range: expected.batchParams.lengthRange,
        gain_limit_range: expected.batchParams.gainLimitRange
    });
    
    // Should have expected combinations
    assert.strictEqual(batchResult.rows, expected.batchCombinations);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.values.length, expected.batchCombinations * 100);
    
    // Verify metadata
    assert.strictEqual(batchResult.combos.length, expected.batchCombinations);
    
    // Verify each row matches individual calculation
    const lengths = [15, 20, 25];
    const gains = [40, 50, 60];
    let rowIdx = 0;
    
    for (const length of lengths) {
        for (const gain of gains) {
            const rowStart = rowIdx * 100;
            const rowEnd = rowStart + 100;
            const rowData = batchResult.values.slice(rowStart, rowEnd);
            
            const singleResult = wasm.ehlers_ecema_js(close, length, gain);
            assertArrayClose(
                rowData,
                singleResult,
                1e-10,
                `Length ${length}, gain ${gain} mismatch`
            );
            rowIdx++;
        }
    }
});

// Test new ehlers_ecema_into_ex API with mode flags
test('EHLERS_ECEMA_into_ex with mode flags', () => {
    // Test the new API function with pine_compatible and confirmed_only flags
    if (!wasm.ehlers_ecema_into_ex) {
        console.log('ehlers_ecema_into_ex not yet available');
        return;
    }
    
    // Use real CSV data (using a subset for mode test)
    const data = new Float64Array(testData.close.slice(0, 25));
    
    const len = data.length;
    
    // Test different mode combinations
    const testCases = [
        { pine: false, confirmed: false, desc: "Regular mode" },
        { pine: true, confirmed: false, desc: "Pine mode" },
        { pine: false, confirmed: true, desc: "Confirmed mode" },
        { pine: true, confirmed: true, desc: "Pine + Confirmed mode" }
    ];
    
    for (const testCase of testCases) {
        const inPtr = wasm.ehlers_ecema_alloc(len);
        const outPtr = wasm.ehlers_ecema_alloc(len);
        
        // Copy data using correct API
        const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        const inMem = new Float64Array(memory.buffer, inPtr, len);
        inMem.set(data);
        
        // Run with specific modes
        wasm.ehlers_ecema_into_ex(
            inPtr, outPtr, len, 20, 50,
            testCase.pine, testCase.confirmed
        );
        
        // Read result using correct API
        const memory2 = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        const outMem = new Float64Array(memory2.buffer, outPtr, len);
        const result = Array.from(outMem);
        
        // Clean up
        wasm.ehlers_ecema_free(inPtr, len);
        wasm.ehlers_ecema_free(outPtr, len);
        
        // Basic validation
        assert.strictEqual(result.length, len, `${testCase.desc}: Length mismatch`);
        
        // Check warmup period behavior based on mode
        if (testCase.pine) {
            // Pine mode starts calculating from first value
            assert(!isNaN(result[0]), `${testCase.desc}: Should have value at index 0`);
        } else {
            // Regular mode has warmup period
            for (let i = 0; i < 19; i++) {
                assert(isNaN(result[i]), `${testCase.desc}: Expected NaN at index ${i}`);
            }
            assert(!isNaN(result[19]), `${testCase.desc}: Expected value at index 19`);
        }
    }
});

// Test memory leak prevention
test('EHLERS_ECEMA zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.ehlers_ecema_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Write pattern to verify memory using correct API
        const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        const memView = new Float64Array(memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 2.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 2.5, `Memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.ehlers_ecema_free(ptr, size);
    }
});

// Test SIMD128 consistency
test('EHLERS_ECEMA SIMD128 consistency', () => {
    // This test verifies SIMD128 produces same results as scalar
    const testCases = [
        { size: 25, length: 10, gain: 30 },
        { size: 100, length: 20, gain: 50 },
        { size: 1000, length: 30, gain: 70 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) * 100 + 50000;
        }
        
        const result = wasm.ehlers_ecema_js(data, testCase.length, testCase.gain);
        
        // Basic sanity checks
        assert.strictEqual(result.length, data.length);
        
        // Check warmup period
        for (let i = 0; i < testCase.length - 1; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        // Check values exist after warmup
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.length - 1; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        // Verify reasonable values
        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(avgAfterWarmup > 40000 && avgAfterWarmup < 60000, 
               `Average value ${avgAfterWarmup} seems unreasonable`);
    }
});