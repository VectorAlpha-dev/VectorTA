/**
 * WASM binding tests for DMA (Dickson Moving Average) indicator.
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
    } catch (error) {
        console.error('Failed to load WASM module:', error);
        throw error;
    }
    
    // Load test data
    testData = loadTestData();
});

test('DMA partial params', () => {
    // Test with default parameters - mirrors check_dma_partial_params
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.dma;
    
    const result = wasm.dma_js(
        close,
        expected.defaultParams.hull_length,
        expected.defaultParams.ema_length,
        expected.defaultParams.ema_gain_limit,
        expected.defaultParams.hull_ma_type
    );
    assert.strictEqual(result.length, close.length);
    
    // Calculate correct warmup period: max(hull, ema) - 1
    const hullLen = expected.defaultParams.hull_length;
    const emaLen = expected.defaultParams.ema_length;
    const warmup = Math.max(hullLen, emaLen) - 1;
    
    assertAllNaN(result.slice(0, warmup), `Expected NaN during warmup period (first ${warmup} values)`);
    
    // Check we have values after warmup
    let nonNanCount = 0;
    for (let i = warmup; i < result.length; i++) {
        if (!isNaN(result[i])) nonNanCount++;
    }
    assert(nonNanCount > close.length - warmup - 10, "Should have values after warmup");
});

test('DMA accuracy', () => {
    // Test DMA matches expected values from Rust tests - mirrors check_dma_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.dma;
    
    const result = wasm.dma_js(
        close,
        expected.defaultParams.hull_length,
        expected.defaultParams.ema_length,
        expected.defaultParams.ema_gain_limit,
        expected.defaultParams.hull_ma_type
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected with proper tolerance for DMA
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        0.001,  // Use same tolerance as Rust tests
        "DMA last 5 values mismatch"
    );
});

test('DMA default candles', () => {
    // Test DMA with default parameters - mirrors check_dma_default_candles  
    const close = new Float64Array(testData.close);
    
    // Default params from Rust
    const result = wasm.dma_js(close, 7, 20, 50, "WMA");
    assert.strictEqual(result.length, close.length);
    
    // Calculate correct warmup: max(7, 20) - 1 = 19
    const warmup = Math.max(7, 20) - 1;
    
    assertAllNaN(result.slice(0, warmup), `Expected NaN in first ${warmup} values`);
    
    // Check we have values after warmup
    let hasValues = false;
    for (let i = warmup; i < result.length; i++) {
        if (!isNaN(result[i])) {
            hasValues = true;
            break;
        }
    }
    assert(hasValues, "Should have values after warmup");
});

test('DMA zero hull period', () => {
    // Test DMA fails with zero hull period - mirrors check_dma_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.dma_js(inputData, 0, 20, 50, "WMA");
    }, /Invalid period/);
});

test('DMA zero ema period', () => {
    // Test DMA fails with zero EMA period - mirrors check_dma_zero_ema_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.dma_js(inputData, 7, 0, 50, "WMA");
    }, /Invalid period/);
});

test('DMA empty input', () => {
    // Test DMA fails with empty input - mirrors check_dma_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.dma_js(empty, 7, 20, 50, "WMA");
    }, /empty/i);
});

test('DMA all NaN', () => {
    // Test DMA fails with all NaN values - mirrors check_dma_all_nan
    const nanData = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.dma_js(nanData, 7, 20, 50, "WMA");
    }, /NaN/i);
});

test('DMA invalid hull type', () => {
    // Test DMA fails with invalid hull_ma_type - mirrors check_dma_invalid_hull_type
    const inputData = new Float64Array(Array(50).fill(10.0));
    
    assert.throws(() => {
        wasm.dma_js(inputData, 7, 20, 50, "INVALID");
    }, /Invalid Hull MA type/);
});

test('DMA EMA hull type', () => {
    // Test DMA works with EMA hull type - mirrors check_dma_ema_hull_type
    const inputData = new Float64Array(Array.from({length: 100}, (_, i) => i));
    
    const result = wasm.dma_js(
        inputData,
        7,    // hull_length
        20,   // ema_length
        50,   // ema_gain_limit
        "EMA" // hull_ma_type
    );
    
    assert.strictEqual(result.length, inputData.length);
    // Check that some non-NaN values are produced after warmup
    const nonNanCount = result.filter(x => !isNaN(x)).length;
    assert(nonNanCount > 0, 'Should produce non-NaN values after warmup');
});

test('DMA period exceeds length', () => {
    // Test DMA fails when period exceeds data length - mirrors check_dma_period_exceeds_length
    const smallData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.dma_js(smallData, 10, 20, 50, "WMA");
    }, /Invalid period/);
});

test('DMA insufficient data', () => {
    // Test DMA fails with insufficient data - mirrors check_dma_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.dma_js(singlePoint, 7, 20, 50, "WMA");
    });
});

test('DMA NaN handling', () => {
    // Test DMA handles NaN values correctly - mirrors check_dma_nan_handling
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.dma;
    
    const result = wasm.dma_js(
        close,
        expected.defaultParams.hull_length,
        expected.defaultParams.ema_length,
        expected.defaultParams.ema_gain_limit,
        expected.defaultParams.hull_ma_type
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Calculate correct warmup period
    const hullLen = expected.defaultParams.hull_length;
    const emaLen = expected.defaultParams.ema_length;
    const warmup = Math.max(hullLen, emaLen) - 1;
    
    // After warmup period, no NaN values should exist (unless input has NaN)
    if (result.length > warmup + 100) {
        // Check a safe region well after warmup
        for (let i = warmup + 100; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First warmup values should be NaN
    assertAllNaN(result.slice(0, warmup), `Expected NaN in warmup period (first ${warmup} values)`);
});

// NOTE: DMA batch processing is not available in WASM bindings
// Only Python bindings have batch support

test('DMA constant input', () => {
    // Test DMA with constant input values
    const constantVal = EXPECTED_OUTPUTS.dma.constantValue;
    const inputData = new Float64Array(100).fill(constantVal);
    
    const result = wasm.dma_js(inputData, 7, 20, 50, "WMA");
    
    assert.strictEqual(result.length, inputData.length);
    
    // After sufficient warmup, DMA should converge close to the constant
    // Check last 10 values are close to constant
    const last10 = result.slice(-10);
    const validValues = last10.filter(x => !isNaN(x));
    
    if (validValues.length > 0) {
        // DMA of constant should be close to the constant
        validValues.forEach((val, i) => {
            assertClose(
                val, 
                constantVal,
                0.01,  // Allow 1% tolerance for convergence
                `DMA should converge to constant input value at index ${i}`
            );
        });
    }
});

test('DMA mixed hull types', () => {
    // Test DMA with different Hull MA types
    const inputData = new Float64Array(Array.from({length: 100}, (_, i) => i * 10 + Math.sin(i)));
    const expected = EXPECTED_OUTPUTS.dma;
    
    const results = {};
    for (const hullType of expected.hullMaTypes) {
        const result = wasm.dma_js(
            inputData,
            7,    // hull_length
            20,   // ema_length
            50,   // ema_gain_limit
            hullType
        );
        
        assert.strictEqual(result.length, inputData.length, `Result length mismatch for hull_type=${hullType}`);
        
        // Check warmup and valid values
        const nonNanCount = result.filter(x => !isNaN(x)).length;
        assert(nonNanCount > 0, `Should produce values for hull_type=${hullType}`);
        
        results[hullType] = result;
    }
    
    // Verify EMA and WMA produce different results
    if (results['EMA'] && results['WMA']) {
        let hasDifference = false;
        for (let i = 0; i < results['EMA'].length; i++) {
            if (!isNaN(results['EMA'][i]) && !isNaN(results['WMA'][i])) {
                if (Math.abs(results['EMA'][i] - results['WMA'][i]) > 1e-10) {
                    hasDifference = true;
                    break;
                }
            }
        }
        assert(hasDifference, "EMA and WMA hull types should produce different results");
    }
});

test('DMA gain limit edge cases', () => {
    // Test DMA with edge case gain limit values
    const inputData = new Float64Array(Array.from({length: 50}, (_, i) => i));
    
    // Test with zero gain limit
    const resultZero = wasm.dma_js(
        inputData,
        7,    // hull_length
        20,   // ema_length
        0,    // ema_gain_limit = 0
        "WMA"
    );
    assert.strictEqual(resultZero.length, inputData.length, "Zero gain limit should produce output");
    
    // Test with very large gain limit
    const resultLarge = wasm.dma_js(
        inputData,
        7,     // hull_length
        20,    // ema_length
        1000,  // ema_gain_limit = 1000
        "WMA"
    );
    assert.strictEqual(resultLarge.length, inputData.length, "Large gain limit should produce output");
});

test('DMA trending data', () => {
    // Test DMA with perfectly trending data
    // Linear uptrend
    const uptrend = new Float64Array(Array.from({length: 100}, (_, i) => i));
    const resultUp = wasm.dma_js(uptrend, 7, 20, 50, "WMA");
    assert.strictEqual(resultUp.length, uptrend.length);
    
    // Linear downtrend
    const downtrend = new Float64Array(Array.from({length: 100}, (_, i) => 100 - i));
    const resultDown = wasm.dma_js(downtrend, 7, 20, 50, "WMA");
    assert.strictEqual(resultDown.length, downtrend.length);
    
    // After warmup, DMA should follow the trend
    const warmup = 19; // max(7, 20) - 1
    
    // Check uptrend follows direction
    if (resultUp.length > warmup + 10) {
        const last10 = resultUp.slice(-10);
        const valid = last10.filter(x => !isNaN(x));
        if (valid.length > 1) {
            // Should be generally increasing
            let increasingCount = 0;
            for (let i = 1; i < valid.length; i++) {
                if (valid[i] > valid[i-1]) increasingCount++;
            }
            assert(increasingCount > valid.length / 2, "DMA should follow uptrend");
        }
    }
});

test('DMA oscillating data', () => {
    // Test DMA with oscillating/choppy data
    // Create sine wave data
    const points = 100;
    const oscillating = new Float64Array(Array.from({length: points}, (_, i) => {
        const x = (i / points) * 4 * Math.PI;
        return 50.0 + 10.0 * Math.sin(x);
    }));
    
    const result = wasm.dma_js(
        oscillating,
        7,    // hull_length
        20,   // ema_length
        50,   // ema_gain_limit
        "WMA"
    );
    
    assert.strictEqual(result.length, oscillating.length);
    
    // DMA should smooth the oscillations
    const warmup = 19;
    if (result.length > warmup) {
        const validResult = result.slice(warmup).filter(x => !isNaN(x));
        const validInput = oscillating.slice(warmup);
        
        // Calculate variance
        const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
        const variance = arr => {
            const m = mean(arr);
            return arr.reduce((a, b) => a + Math.pow(b - m, 2), 0) / arr.length;
        };
        
        if (validResult.length > 0) {
            const resultVar = variance(validResult);
            const inputVar = variance(validInput);
            
            // DMA should reduce variance (smooth the data)
            assert(resultVar < inputVar, "DMA should smooth oscillating data");
        }
    }
});

test('DMA extreme ratios', () => {
    // Test DMA with extreme hull/ema length ratios
    const inputData = new Float64Array(Array.from({length: 100}, (_, i) => i * 2 + Math.random()));
    
    // Very small hull, large ema
    const result1 = wasm.dma_js(
        inputData,
        3,    // hull_length
        50,   // ema_length
        50,   // ema_gain_limit
        "WMA"
    );
    assert.strictEqual(result1.length, inputData.length);
    
    // Very large hull, small ema
    const result2 = wasm.dma_js(
        inputData,
        50,   // hull_length
        3,    // ema_length
        50,   // ema_gain_limit
        "WMA"
    );
    assert.strictEqual(result2.length, inputData.length);
    
    // Results should be different
    let hasDifference = false;
    for (let i = 0; i < result1.length; i++) {
        if (!isNaN(result1[i]) && !isNaN(result2[i])) {
            if (Math.abs(result1[i] - result2[i]) > 0.01) {
                hasDifference = true;
                break;
            }
        }
    }
    assert(hasDifference, "Different hull/ema ratios should produce different results");
});

test('DMA zero-copy API', () => {
    // Test zero-copy API for DMA
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
    const hullLength = 5;
    const emaLength = 10;
    const emaGainLimit = 50;
    const hullMaType = "WMA";
    
    // Allocate buffer
    const ptr = wasm.dma_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memory = wasm.__wbindgen_memory();
    const memView = new Float64Array(
        memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    try {
        // Compute DMA in-place
        wasm.dma_into(
            ptr,
            ptr,
            data.length,
            hullLength,
            emaLength,
            emaGainLimit,
            hullMaType
        );
        
        // Verify results match regular API
        const regularResult = wasm.dma_js(data, hullLength, emaLength, emaGainLimit, hullMaType);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.dma_free(ptr, data.length);
    }
});

test('DMA zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.dma_into(0, 0, 10, 7, 20, 50, "WMA");
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.dma_alloc(10);
    try {
        // Invalid hull period
        assert.throws(() => {
            wasm.dma_into(ptr, ptr, 10, 0, 20, 50, "WMA");
        }, /Invalid period/);
        
        // Invalid ema period
        assert.throws(() => {
            wasm.dma_into(ptr, ptr, 10, 7, 0, 50, "WMA");
        }, /Invalid period/);
        
        // Invalid hull type - but with small buffer, it fails on period check first
        // so we just check that it throws an error
        assert.throws(() => {
            wasm.dma_into(ptr, ptr, 10, 7, 20, 50, "INVALID");
        });
    } finally {
        wasm.dma_free(ptr, 10);
    }
});

test('DMA memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.dma_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Write pattern to verify memory
        const memory = wasm.__wbindgen_memory();
        const memView = new Float64Array(memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.dma_free(ptr, size);
    }
});

test.after(() => {
    console.log('DMA WASM tests completed');
});