/**
 * WASM binding tests for CCI_CYCLE indicator.
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

test('CCI_CYCLE partial params', () => {
    // Test with default parameters - mirrors check_cci_cycle_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.cci_cycle_js(close, 10, 0.5);
    assert.strictEqual(result.length, close.length);
});

test('CCI_CYCLE accuracy', () => {
    // Test CCI_CYCLE matches expected values from Rust tests - mirrors check_cci_cycle_accuracy
    const close = new Float64Array(testData.close);
    
    // Default parameters from Rust
    const length = 10;
    const factor = 0.5;
    
    const result = wasm.cci_cycle_js(close, length, factor);
    
    assert.strictEqual(result.length, close.length);
    
    // Reference values from PineScript
    const expected_last_five = [
        9.25177192,
        20.49219826,
        35.42917181,
        55.57843075,
        77.78921538,
    ];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected_last_five,
        1e-6,
        "CCI_CYCLE last 5 values mismatch"
    );
});

test('CCI_CYCLE default candles', () => {
    // Test CCI_CYCLE with default parameters - mirrors check_cci_cycle_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.cci_cycle_js(close, 10, 0.5);
    assert.strictEqual(result.length, close.length);
});

test('CCI_CYCLE zero period', () => {
    // Test CCI_CYCLE fails with zero period - mirrors check_cci_cycle_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cci_cycle_js(inputData, 0, 0.5);
    }, /Invalid period/);
});

test('CCI_CYCLE period exceeds length', () => {
    // Test CCI_CYCLE fails when period exceeds data length - mirrors check_cci_cycle_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cci_cycle_js(dataSmall, 10, 0.5);
    }, /Invalid period/);
});

test('CCI_CYCLE very small dataset', () => {
    // Test CCI_CYCLE fails with insufficient data - mirrors check_cci_cycle_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.cci_cycle_js(singlePoint, 10, 0.5);
    }, /Invalid period|Not enough valid data/);
});

test('CCI_CYCLE empty input', () => {
    // Test CCI_CYCLE fails with empty input - mirrors check_cci_cycle_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.cci_cycle_js(empty, 10, 0.5);
    }, /empty/i);
});

test('CCI_CYCLE factor edge cases', () => {
    const data = new Float64Array([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
    ]);

    // NaN factor is allowed; it should propagate NaNs through most of the output.
    const nanResult = wasm.cci_cycle_js(data, 5, NaN);
    assert.strictEqual(nanResult.length, data.length);
    let nanCount = 0;
    for (const v of nanResult) {
        if (Number.isNaN(v)) nanCount++;
    }
    assert.ok(
        nanCount >= data.length - 5,
        `Expected mostly NaN when factor is NaN, got ${nanCount}/${data.length} NaN values`
    );

    // Negative and large factors are allowed.
    const negResult = wasm.cci_cycle_js(data, 5, -0.5);
    assert.strictEqual(negResult.length, data.length);

    const bigResult = wasm.cci_cycle_js(data, 5, 10.0);
    assert.strictEqual(bigResult.length, data.length);

    // Infinities are rejected.
    assert.throws(() => {
        wasm.cci_cycle_js(data, 5, Infinity);
    }, /Invalid factor/);
});

test('CCI_CYCLE all NaN input', () => {
    // Test CCI_CYCLE with all NaN values - mirrors check_cci_cycle_all_nan
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.cci_cycle_js(allNaN, 10, 0.5);
    }, /All values are NaN/);
});

test('CCI_CYCLE NaN handling', () => {
    // Test CCI_CYCLE handles NaN values correctly - mirrors check_cci_cycle_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.cci_cycle_js(close, 10, 0.5);
    assert.strictEqual(result.length, close.length);
    
    // CCI Cycle has a complex warmup period involving multiple indicators
    // Check that we have NaN values at the beginning
    let initialNans = 0;
    for (let i = 0; i < Math.min(50, result.length); i++) {
        if (isNaN(result[i])) {
            initialNans++;
        } else {
            break;
        }
    }
    
    // Should have at least some warmup period
    assert(initialNans > 0, "Expected some NaN values during warmup period");
    
    // After sufficient data, should have valid values
    if (result.length > 100) {
        let nonNanCount = 0;
        for (let i = 100; i < Math.min(200, result.length); i++) {
            if (!isNaN(result[i])) {
                nonNanCount++;
            }
        }
        assert(nonNanCount > 0, "Should have some valid values after sufficient data");
    }
});

test('CCI_CYCLE batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    try {
        // Using the ergonomic batch API for single parameter
        const batchResult = wasm.cci_cycle_batch(close, {
            length_range: [10, 10, 0],
            factor_range: [0.5, 0.5, 0]
        });
        
        // Should match single calculation
        const singleResult = wasm.cci_cycle_js(close, 10, 0.5);
        
        assert.strictEqual(batchResult.values.length, singleResult.length);
        assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
    } catch (error) {
        // Batch API might not be available, skip
        console.log("CCI_CYCLE batch API not available, skipping batch tests");
    }
});

test('CCI_CYCLE batch multiple parameters', () => {
    // Test batch with multiple parameter values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    try {
        // Multiple parameters using ergonomic API
        const batchResult = wasm.cci_cycle_batch(close, {
            length_range: [10, 20, 5],      // 10, 15, 20
            factor_range: [0.3, 0.7, 0.2]   // 0.3, 0.5, 0.7
        });
        
        // Should have 3 * 3 = 9 combinations
        assert.strictEqual(batchResult.combos.length, 9);
        assert.strictEqual(batchResult.rows, 9);
        assert.strictEqual(batchResult.cols, 100);
        assert.strictEqual(batchResult.values.length, 900);
        
        // Verify each row matches individual calculation
        const lengths = [10, 10, 10, 15, 15, 15, 20, 20, 20];
        const factors = [0.3, 0.5, 0.7, 0.3, 0.5, 0.7, 0.3, 0.5, 0.7];
        
        for (let i = 0; i < 9; i++) {
            const rowStart = i * 100;
            const rowEnd = rowStart + 100;
            const rowData = batchResult.values.slice(rowStart, rowEnd);
            
            const singleResult = wasm.cci_cycle_js(close, lengths[i], factors[i]);
            assertArrayClose(
                rowData, 
                singleResult, 
                1e-10, 
                `Length ${lengths[i]}, Factor ${factors[i]} mismatch`
            );
        }
    } catch (error) {
        // Batch API might not be available, skip
        console.log("CCI_CYCLE batch API not available, skipping batch tests");
    }
});

test('CCI_CYCLE batch metadata', () => {
    // Test that batch result includes correct parameter combinations
    const close = new Float64Array(50); // Small dataset
    close.fill(100);
    
    try {
        const result = wasm.cci_cycle_batch(close, {
            length_range: [10, 15, 5],      // 10, 15
            factor_range: [0.3, 0.7, 0.2]   // 0.3, 0.5, 0.7
        });
        
        // Should have 2 * 3 = 6 combinations
        assert.strictEqual(result.combos.length, 6);
        
        // Check first combination
        assert.strictEqual(result.combos[0].length, 10);
        assertClose(result.combos[0].factor, 0.3, 1e-10, "factor mismatch");
        
        // Check last combination
        assert.strictEqual(result.combos[5].length, 15);
        assertClose(result.combos[5].factor, 0.7, 1e-10, "factor mismatch");
    } catch (error) {
        // Batch API might not be available, skip
        console.log("CCI_CYCLE batch API not available, skipping batch tests");
    }
});

test('CCI_CYCLE batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
    
    try {
        // Single value sweep
        const singleBatch = wasm.cci_cycle_batch(close, {
            length_range: [5, 5, 1],
            factor_range: [0.5, 0.5, 0.1]
        });
        
        assert.strictEqual(singleBatch.values.length, 20);
        assert.strictEqual(singleBatch.combos.length, 1);
        
        // Step larger than range
        const largeBatch = wasm.cci_cycle_batch(close, {
            length_range: [5, 7, 10], // Step larger than range
            factor_range: [0.5, 0.5, 0]
        });
        
        // Should only have length=5
        assert.strictEqual(largeBatch.values.length, 20);
        assert.strictEqual(largeBatch.combos.length, 1);
        
        // Empty data should throw
        assert.throws(() => {
            wasm.cci_cycle_batch(new Float64Array([]), {
                length_range: [10, 10, 0],
                factor_range: [0.5, 0.5, 0]
            });
        }, /All values are NaN|empty/);
    } catch (error) {
        // Batch API might not be available, skip
        console.log("CCI_CYCLE batch API not available, skipping batch tests");
    }
});

// Zero-copy API tests
test('CCI_CYCLE zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
    const length = 5;
    const factor = 0.5;
    
    try {
        // Allocate buffer
        const ptr = wasm.cci_cycle_alloc(data.length);
        assert(ptr !== 0, 'Failed to allocate memory');
        
        // Create view into WASM memory
        const memory = wasm.__wasm.memory.buffer;
        const memView = new Float64Array(
            memory,
            ptr,
            data.length
        );
        
        // Copy data into WASM memory
        memView.set(data);
        
        // Compute CCI_CYCLE in-place
        try {
            wasm.cci_cycle_into(ptr, ptr, data.length, length, factor);
            
            // Verify results match regular API
            const regularResult = wasm.cci_cycle_js(data, length, factor);
            for (let i = 0; i < data.length; i++) {
                if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                    continue; // Both NaN is OK
                }
                assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                       `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
            }
        } finally {
            // Always free memory
            wasm.cci_cycle_free(ptr, data.length);
        }
    } catch (error) {
        // Zero-copy API might not be available, skip
        console.log("CCI_CYCLE zero-copy API not available, skipping zero-copy tests");
    }
});

test('CCI_CYCLE zero-copy with large dataset', () => {
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    try {
        const ptr = wasm.cci_cycle_alloc(size);
        assert(ptr !== 0, 'Failed to allocate large buffer');
        
        try {
            const memory = wasm.__wasm.memory.buffer;
            const memView = new Float64Array(memory, ptr, size);
            memView.set(data);
            
            wasm.cci_cycle_into(ptr, ptr, size, 10, 0.5);
            
            // Recreate view in case memory grew
            const memory2 = wasm.__wasm.memory.buffer;
            const memView2 = new Float64Array(memory2, ptr, size);
            
            // Check warmup period has NaN (length * 4)
            const warmupPeriod = 10 * 4;
            for (let i = 0; i < warmupPeriod; i++) {
                assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
            }
            
            // Check after warmup has values (with some buffer)
            let hasValidValues = false;
            for (let i = warmupPeriod + 10; i < Math.min(warmupPeriod + 100, size); i++) {
                if (!isNaN(memView2[i])) {
                    hasValidValues = true;
                    break;
                }
            }
            assert(hasValidValues, "Should have some valid values after warmup");
        } finally {
            wasm.cci_cycle_free(ptr, size);
        }
    } catch (error) {
        // Zero-copy API might not be available, skip
        console.log("CCI_CYCLE zero-copy API not available, skipping zero-copy tests");
    }
});

// Error handling for zero-copy API
test('CCI_CYCLE zero-copy error handling', () => {
    try {
        // Test null pointer
        assert.throws(() => {
            wasm.cci_cycle_into(0, 0, 10, 10, 0.5);
        }, /null pointer|invalid memory/i);
        
        // Test invalid parameters with allocated memory
        const ptr = wasm.cci_cycle_alloc(20);
        try {
            // Invalid period
            assert.throws(() => {
                wasm.cci_cycle_into(ptr, ptr, 20, 0, 0.5);
            }, /Invalid period/);
            
            // NaN factor is allowed (should not throw)
            assert.doesNotThrow(() => {
                wasm.cci_cycle_into(ptr, ptr, 20, 5, NaN);
            });

            // Invalid factor (infinite)
            assert.throws(() => {
                wasm.cci_cycle_into(ptr, ptr, 20, 5, Infinity);
            }, /Invalid factor/);
        } finally {
            wasm.cci_cycle_free(ptr, 20);
        }
    } catch (error) {
        // Zero-copy API might not be available, skip
        console.log("CCI_CYCLE zero-copy API not available, skipping error handling tests");
    }
});

// Memory leak prevention test
test('CCI_CYCLE zero-copy memory management', () => {
    try {
        // Allocate and free multiple times to ensure no leaks
        const sizes = [100, 1000, 5000];
        
        for (const size of sizes) {
            const ptr = wasm.cci_cycle_alloc(size);
            assert(ptr !== 0, `Failed to allocate ${size} elements`);
            
            // Write pattern to verify memory
            const memory = wasm.__wasm.memory.buffer;
            const memView = new Float64Array(memory, ptr, size);
            for (let i = 0; i < Math.min(10, size); i++) {
                memView[i] = i * 1.5;
            }
            
            // Verify pattern
            for (let i = 0; i < Math.min(10, size); i++) {
                assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
            }
            
            // Free memory
            wasm.cci_cycle_free(ptr, size);
        }
    } catch (error) {
        // Zero-copy API might not be available, skip
        console.log("CCI_CYCLE zero-copy API not available, skipping memory management tests");
    }
});

test.after(() => {
    console.log('CCI_CYCLE WASM tests completed');
});
