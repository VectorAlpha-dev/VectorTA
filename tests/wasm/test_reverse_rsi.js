/**
 * WASM binding tests for REVERSE_RSI indicator.
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

test('REVERSE_RSI partial params', () => {
    // Test with default parameters - mirrors check_reverse_rsi_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.reverse_rsi_js(close, 14, 50.0);
    assert.strictEqual(result.length, close.length);
});

test('REVERSE_RSI accuracy', () => {
    // Test REVERSE_RSI accuracy - mirrors check_reverse_rsi_accuracy
    // Use the same CSV data as Rust test
    const close = new Float64Array(testData.close);
    
    // Default parameters matching Rust test
    const rsiLength = 14;
    const rsiLevel = 50.0;
    
    const result = wasm.reverse_rsi_js(close, rsiLength, rsiLevel);
    
    // Verify the calculation produces valid results
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected reference values
    // Note: We check positions -6 to -2 (5 values before the last one)
    const expectedLast5 = [
        60124.655535277416, 60064.68013990046, 60001.56012990757, 59932.80583491417, 59877.248275277445
    ];
    
    const last5 = result.slice(-6, -1);  // Get values at positions -6 to -2
    for (let i = 0; i < 5; i++) {
        assertClose(
            last5[i],
            expectedLast5[i],
            1e-6,
            `REVERSE_RSI last 5 values mismatch at index ${i}`
        );
    }
});

test('REVERSE_RSI default params', () => {
    // Test REVERSE_RSI with default parameters - mirrors check_reverse_rsi_default_candles
    const close = new Float64Array(testData.close);
    
    // Default params: rsi_length=14, rsi_level=50.0
    const result = wasm.reverse_rsi_js(close, 14, 50.0);
    assert.strictEqual(result.length, close.length);
});

test('REVERSE_RSI zero period', () => {
    // Test REVERSE_RSI fails with zero period - mirrors check_reverse_rsi_zero_period
    const input_data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.reverse_rsi_js(input_data, 0, 50.0);
    }, /Invalid period/, 'Should throw error for zero period');
});

test('REVERSE_RSI period exceeds length', () => {
    // Test REVERSE_RSI fails when period exceeds data length - mirrors check_reverse_rsi_period_exceeds_length
    const data_small = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.reverse_rsi_js(data_small, 10, 50.0);
    }, /Invalid period/, 'Should throw error when period exceeds data length');
});

test('REVERSE_RSI invalid level', () => {
    // Test REVERSE_RSI fails with invalid RSI level - mirrors check_reverse_rsi_invalid_level
    const input_data = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0, 10.0, 20.0, 30.0, 40.0, 50.0,
                                          10.0, 20.0, 30.0, 40.0, 50.0, 10.0, 20.0, 30.0, 40.0, 50.0,
                                          10.0, 20.0, 30.0, 40.0, 50.0]);
    
    // Test level > 100
    assert.throws(() => {
        wasm.reverse_rsi_js(input_data, 14, 150.0);
    }, /Invalid RSI level/, 'Should throw error for RSI level > 100');
    
    // Test negative level
    assert.throws(() => {
        wasm.reverse_rsi_js(input_data, 14, -10.0);
    }, /Invalid RSI level/, 'Should throw error for negative RSI level');
});

test('REVERSE_RSI edge levels', () => {
    // Test REVERSE_RSI with edge RSI levels (near 0 and 100) - not in Rust tests but validates edge behavior
    const input_data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        input_data[i] = 10.0 + (i % 5) * 10.0;
    }
    
    // Test with RSI level = 0.01 (near extreme oversold, but not exactly 0)
    const result01 = wasm.reverse_rsi_js(input_data, 14, 0.01);
    assert.strictEqual(result01.length, input_data.length);
    // Should produce valid values (though extreme)
    assert(!result01.every(isNaN), 'Should have some valid values for RSI level = 0.01');
    
    // Also test with RSI level = 1.0
    const result1 = wasm.reverse_rsi_js(input_data, 14, 1.0);
    assert.strictEqual(result1.length, input_data.length);
    // Should produce valid values (though extreme)
    assert(!result1.every(isNaN), 'Should have some valid values for RSI level = 1.0');
    
    // Test with RSI level = 99.0 (near extreme overbought, but not exactly 100)
    const result99 = wasm.reverse_rsi_js(input_data, 14, 99.0);
    assert.strictEqual(result99.length, input_data.length);
    // Should produce valid values (though extreme)
    assert(!result99.every(isNaN), 'Should have some valid values for RSI level = 99.0');
});

test('REVERSE_RSI various levels', () => {
    // Test REVERSE_RSI with various RSI levels - validates different levels produce different results
    const input_data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        input_data[i] = 10.0 + i;
    }
    
    const levels = [20.0, 30.0, 50.0, 70.0, 80.0];
    const results = [];
    
    for (const level of levels) {
        const result = wasm.reverse_rsi_js(input_data, 14, level);
        assert.strictEqual(result.length, input_data.length);
        results.push(result);
    }
    
    // Different levels should produce different results
    for (let i = 0; i < results.length - 1; i++) {
        // Compare non-NaN values
        let foundDifference = false;
        for (let j = 0; j < results[i].length; j++) {
            if (!isNaN(results[i][j]) && !isNaN(results[i+1][j])) {
                if (Math.abs(results[i][j] - results[i+1][j]) > 1e-10) {
                    foundDifference = true;
                    break;
                }
            }
        }
        assert(foundDifference, `Different RSI levels (${levels[i]} vs ${levels[i+1]}) should produce different results`);
    }
});

test('REVERSE_RSI empty input', () => {
    // Test REVERSE_RSI with empty input - mirrors check_reverse_rsi_empty_input
    const input_data = new Float64Array([]);
    
    assert.throws(() => {
        wasm.reverse_rsi_js(input_data, 14, 50.0);
    }, /Input data slice is empty/, 'Should throw error for empty input');
});

test('REVERSE_RSI all NaN', () => {
    // Test REVERSE_RSI with all NaN values - mirrors check_reverse_rsi_all_nan
    const input_data = new Float64Array(30);
    input_data.fill(NaN);
    
    assert.throws(() => {
        wasm.reverse_rsi_js(input_data, 14, 50.0);
    }, /All values are NaN/, 'Should throw error for all NaN input');
});

test('REVERSE_RSI insufficient data', () => {
    // Test REVERSE_RSI with insufficient data - mirrors check_reverse_rsi_very_small_dataset
    // Need at least ema_length + 1 values
    // ema_length = (2 * rsi_length) - 1 = 27 for rsi_length=14
    const input_data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]); // 5 values
    
    assert.throws(() => {
        wasm.reverse_rsi_js(input_data, 14, 50.0); // Needs 28 values
    }, /(Invalid period|Not enough valid data)/, 'Should throw error for insufficient data');
});

test('REVERSE_RSI nan handling', () => {
    // Test REVERSE_RSI handles NaN in middle of data - mirrors check_reverse_rsi_nan_handling
    // Create data with 50 values
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1.0;
    }
    
    const rsiLength = 14;
    const rsiLevel = 50.0;
    
    // Test 1: NaN in the middle of data
    const dataWithNaN = new Float64Array(data);
    dataWithNaN[25] = NaN;
    
    const result = wasm.reverse_rsi_js(dataWithNaN, rsiLength, rsiLevel);
    assert.strictEqual(result.length, dataWithNaN.length);
    
    // Check that we have valid values after warmup (position 26+)
    // With rsiLength=14, warmup is 26 positions (indices 0-25)
    let hasValidAfter = false;
    for (let i = 26; i < result.length; i++) {
        if (!isNaN(result[i])) {
            hasValidAfter = true;
            break;
        }
    }
    assert(hasValidAfter, 'Should have valid values after warmup/NaN');
    
    // Test 2: Verify the indicator handles multiple NaNs gracefully
    const dataMultiNaN = new Float64Array(data);
    dataMultiNaN[20] = NaN;
    dataMultiNaN[30] = NaN;
    
    // Should not crash with multiple NaNs
    const result2 = wasm.reverse_rsi_js(dataMultiNaN, rsiLength, rsiLevel);
    assert.strictEqual(result2.length, dataMultiNaN.length);
    assert(result2 instanceof Float64Array, 'Should return Float64Array');
});

test('REVERSE_RSI warmup nans', () => {
    // Test REVERSE_RSI preserves warmup NaNs - mirrors check_reverse_rsi_warmup_nans
    const close = new Float64Array(testData.close);
    const rsiLength = 14;
    const rsiLevel = 50.0;
    
    // Find first non-NaN value
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    
    const result = wasm.reverse_rsi_js(close, rsiLength, rsiLevel);
    
    // All values before firstValid should be NaN
    for (let i = 0; i < firstValid; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} (before first valid data)`);
    }
});

test('REVERSE_RSI memory management with into', () => {
    // Test REVERSE_RSI memory management functions - validates WASM pointer API
    const close = new Float64Array(testData.close.slice(0, 50)); // Use smaller dataset for speed
    const rsiLength = 14;
    const rsiLevel = 50.0;
    
    // Allocate memory for output
    const outPtr = wasm.reverse_rsi_alloc(close.length);
    
    // Create input array in WASM memory
    const inPtr = wasm.reverse_rsi_alloc(close.length);
    const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer);
    const inOffset = inPtr / 8; // Convert byte offset to Float64 offset
    
    // Copy input data to WASM memory
    for (let i = 0; i < close.length; i++) {
        wasmMemory[inOffset + i] = close[i];
    }
    
    // Process data (doesn't return a value on success, throws on error)
    wasm.reverse_rsi_into(inPtr, outPtr, close.length, rsiLength, rsiLevel);
    
    // Read output from WASM memory
    const outOffset = outPtr / 8;
    const output = new Float64Array(close.length);
    for (let i = 0; i < close.length; i++) {
        output[i] = wasmMemory[outOffset + i];
    }
    
    // Clean up
    wasm.reverse_rsi_free(inPtr, close.length);
    wasm.reverse_rsi_free(outPtr, close.length);
    
    // Compare with normal function
    const expected = wasm.reverse_rsi_js(close, rsiLength, rsiLevel);
    assertArrayClose(output, expected, 1e-10, 'Memory-managed version should match normal version');
});

test('REVERSE_RSI batch processing', () => {
    // Test REVERSE_RSI batch processing - mirrors check_batch_sweep
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Test with the new unified batch API
    const config = {
        rsi_length_range: [10, 20, 5],  // Start, end, step (10, 15, 20)
        rsi_level_range: [50.0, 50.0, 0.0]  // Just 50.0
    };
    
    const result = wasm.reverse_rsi_batch(close, config);
    
    // Should have values, combos, rows, cols
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert.strictEqual(result.rows, 3, 'Should have 3 parameter combinations');
    assert.strictEqual(result.cols, close.length, 'Should have same columns as input');
    
    // Verify we have the expected parameter combinations
    assert.strictEqual(result.combos.length, 3, 'Should have 3 combinations');
    const expectedLengths = [10, 15, 20];
    for (let i = 0; i < 3; i++) {
        assert.strictEqual(result.combos[i].rsi_length, expectedLengths[i], 
            `Combo ${i} should have rsi_length ${expectedLengths[i]}`);
        assert.strictEqual(result.combos[i].rsi_level, 50.0, 
            `Combo ${i} should have rsi_level 50.0`);
    }
    
    // Check each row has appropriate values
    for (let row = 0; row < result.rows; row++) {
        const rowStart = row * result.cols;
        const rowData = result.values.slice(rowStart, rowStart + result.cols);
        
        // Should have some valid values after warmup
        let hasValid = false;
        for (let i = 0; i < rowData.length; i++) {
            if (!isNaN(rowData[i])) {
                hasValid = true;
                break;
            }
        }
        assert(hasValid, `Row ${row} should have some valid values`);
    }
    
    // Also test the range-based batch_into API if available
    if (wasm.reverse_rsi_batch_into) {
        const len = close.length;
        
        // Allocate memory for input and output
        const inPtr = wasm.reverse_rsi_alloc(len);
        const outPtr = wasm.reverse_rsi_alloc(len * 3); // 3 parameter combinations
        
        // Copy input data to WASM memory
        const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer);
        const inOffset = inPtr / 8;
        for (let i = 0; i < len; i++) {
            wasmMemory[inOffset + i] = close[i];
        }
        
        // Process batch with range parameters
        const numRows = wasm.reverse_rsi_batch_into(
            inPtr, outPtr, len,
            10, 20, 5,  // rsi_length range
            50.0, 50.0, 0.0  // rsi_level range
        );
        
        assert.strictEqual(numRows, 3, 'Should return 3 rows');
        
        // Read output
        const outOffset = outPtr / 8;
        const output = new Float64Array(len * 3);
        for (let i = 0; i < output.length; i++) {
            output[i] = wasmMemory[outOffset + i];
        }
        
        // Clean up
        wasm.reverse_rsi_free(inPtr, len);
        wasm.reverse_rsi_free(outPtr, len * 3);
        
        // Verify output has valid values
        for (let row = 0; row < 3; row++) {
            const rowStart = row * len;
            const rowData = output.slice(rowStart, rowStart + len);
            
            let hasValid = false;
            for (let i = 0; i < rowData.length; i++) {
                if (!isNaN(rowData[i])) {
                    hasValid = true;
                    break;
                }
            }
            assert(hasValid, `Batch_into row ${row} should have some valid values`);
        }
    }
});

test('REVERSE_RSI numerical precision', () => {
    // Test REVERSE_RSI numerical precision and edge cases
    
    // Test with extreme values
    const extremeData = new Float64Array(40);
    for (let i = 0; i < 10; i++) {
        extremeData[i * 4] = 1e-10;
        extremeData[i * 4 + 1] = 1e10;
        extremeData[i * 4 + 2] = 1e-10;
        extremeData[i * 4 + 3] = 1e10;
    }
    
    const result1 = wasm.reverse_rsi_js(extremeData, 5, 50.0);
    assert.strictEqual(result1.length, extremeData.length);
    // Should handle extreme values without overflow/underflow
    const validValues1 = result1.filter(v => !isNaN(v));
    assert(!validValues1.some(v => !isFinite(v)), 'Should not produce infinity');
    
    // Test with very small differences
    const smallDiffData = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        smallDiffData[i] = 100.0 + i * 1e-10;
    }
    
    const result2 = wasm.reverse_rsi_js(smallDiffData, 10, 50.0);
    assert.strictEqual(result2.length, smallDiffData.length);
    // Should handle tiny price movements without numerical issues
    const validValues2 = result2.filter(v => !isNaN(v));
    if (validValues2.length > 0) {
        assert(!validValues2.some(v => !isFinite(v)), 'Should not produce infinity');
    }
    
    // Test with constant values
    const constantData = new Float64Array(30);
    constantData.fill(100.0);
    
    const result3 = wasm.reverse_rsi_js(constantData, 10, 50.0);
    assert.strictEqual(result3.length, constantData.length);
    // With constant prices, reverse RSI should also be constant after warmup
    const validValues3 = result3.filter(v => !isNaN(v));
    if (validValues3.length > 0) {
        assert(!validValues3.some(v => !isFinite(v)), 'Should not produce infinity with constant values');
    }
});