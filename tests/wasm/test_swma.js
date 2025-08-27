/**
 * WASM binding tests for SWMA indicator.
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
    
    testData = await loadTestData();
});

test.describe('SWMA (Symmetric Weighted Moving Average)', () => {
    test('SWMA empty input', () => {
        // Test SWMA fails with empty input - mirrors check_swma_empty_input
        const empty = new Float64Array([]);
        
        assert.throws(
            () => wasm.swma_js(empty, 5),
            /Input data slice is empty/,
            'Should throw error for empty input'
        );
    });
    
    test('SWMA accuracy', async () => {
        // Test SWMA matches expected values from Rust tests - mirrors check_swma_accuracy
        const close = new Float64Array(testData.close);
        const expected = EXPECTED_OUTPUTS.swma;
        
        const result = wasm.swma_js(
            close,
            expected.defaultParams.period
        );
        
        assert.strictEqual(result.length, close.length);
        
        // Check last 5 values match expected
        const last5 = result.slice(-5);
        assertArrayClose(
            last5,
            expected.last5Values,
            1e-8,
            "SWMA last 5 values mismatch"
        );
        
        // Compare full output with Rust
        await compareWithRust('swma', result, 'close', expected.defaultParams);
    });
    
    test('basic functionality', () => {
        const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        const period = 5;
        
        const result = wasm.swma_js(data, period);
        
        assert.strictEqual(result.length, data.length, 'Result length should match input length');
        
        // Check warmup period (first period-1 values should be NaN)
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(result[i]), `Index ${i} should be NaN during warmup`);
        }
        
        // Check that remaining values are not NaN
        for (let i = period - 1; i < result.length; i++) {
            assert(!isNaN(result[i]), `Index ${i} should not be NaN after warmup`);
        }
    });
    
    test('error handling - all NaN values', () => {
        const data = new Float64Array(10).fill(NaN);
        const period = 5;
        
        assert.throws(
            () => wasm.swma_js(data, period),
            /All values are NaN/,
            'Should throw error for all NaN values'
        );
    });
    
    test('error handling - invalid period', () => {
        const data = new Float64Array([1, 2, 3, 4, 5]);
        
        // Period exceeds data length
        assert.throws(
            () => wasm.swma_js(data, 6),
            /Invalid period/,
            'Should throw error when period exceeds data length'
        );
        
        // Period is zero
        assert.throws(
            () => wasm.swma_js(data, 0),
            /Invalid period/,
            'Should throw error for zero period'
        );
    });
    
    test('error handling - not enough valid data', () => {
        // First 8 values are NaN, only 2 valid values, but need period 5
        const data = new Float64Array([NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 1, 2]);
        const period = 5;
        
        assert.throws(
            () => wasm.swma_js(data, period),
            /Not enough valid data/,
            'Should throw error when not enough valid data after NaN values'
        );
    });
    
    test('leading NaN values', () => {
        const data = new Float64Array([NaN, NaN, 1, 2, 3, 4, 5, 6, 7, 8]);
        const period = 3;
        
        const result = wasm.swma_js(data, period);
        
        // First valid index is 2, warmup period is period-1 = 2
        // So first valid output is at index 2+(period-1) = 2+2 = 4
        for (let i = 0; i < 4; i++) {
            assert(isNaN(result[i]), `Index ${i} should be NaN`);
        }
        for (let i = 4; i < result.length; i++) {
            assert(!isNaN(result[i]), `Index ${i} should not be NaN`);
        }
    });
    
    test('compare with Rust implementation', async () => {
        const close = new Float64Array(testData.close);
        const period = 5;
        
        const result = wasm.swma_js(close, period);
        await compareWithRust('swma', result, 'close', { period });
    });
    
    test('batch calculation', () => {
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = Math.random() * 100;
        }
        
        const minPeriod = 3;
        const maxPeriod = 10;
        const stepPeriod = 2;
        
        const values = wasm.swma_batch_js(data, minPeriod, maxPeriod, stepPeriod);
        const metadata = wasm.swma_batch_metadata_js(minPeriod, maxPeriod, stepPeriod);
        
        // Expected periods: 3, 5, 7, 9
        const expectedPeriods = [];
        for (let p = minPeriod; p <= maxPeriod; p += stepPeriod) {
            expectedPeriods.push(p);
        }
        
        const rows = expectedPeriods.length;
        const cols = data.length;
        
        assert.strictEqual(metadata.length, rows, 'Metadata length should match number of periods');
        assert.strictEqual(values.length, rows * cols, 'Values array size should be rows*cols');
        
        // Verify each row matches individual calculation
        for (let i = 0; i < expectedPeriods.length; i++) {
            const period = expectedPeriods[i];
            const individual = wasm.swma_js(data, period);
            
            // Extract row from batch result
            const row = new Float64Array(cols);
            for (let j = 0; j < cols; j++) {
                row[j] = values[i * cols + j];
            }
            
            assertArrayClose(row, individual, 1e-10, `Batch row ${i} should match individual calculation`);
        }
    });
    
    test('symmetric triangular weights', () => {
        // Create a pattern where we can verify the weights
        const data = new Float64Array(30).fill(0);
        data[15] = 1.0; // Single spike in the middle
        const period = 5;
        
        const result = wasm.swma_js(data, period);
        
        // With period 5, triangular weights are [1, 2, 3, 2, 1] / 9
        // result[i] uses data[i-period+1:i+1], so:
        // result[15] uses data[11:16] - spike at 15 is at position 4 (last)
        // result[16] uses data[12:17] - spike at 15 is at position 3
        // result[17] uses data[13:18] - spike at 15 is at position 2 (center)
        // result[18] uses data[14:19] - spike at 15 is at position 1
        // result[19] uses data[15:20] - spike at 15 is at position 0 (first)
        
        assertClose(result[15], 1/9, 1e-10, 'Weight at position 4');
        assertClose(result[16], 2/9, 1e-10, 'Weight at position 3');
        assertClose(result[17], 3/9, 1e-10, 'Weight at position 2 (center)');
        assertClose(result[18], 2/9, 1e-10, 'Weight at position 1');
        assertClose(result[19], 1/9, 1e-10, 'Weight at position 0');
        
        // Verify zero outside the affected range
        assert.strictEqual(result[10], 0.0, 'Should be zero before affected range');
        assert.strictEqual(result[20], 0.0, 'Should be zero after affected range');
    });
    
    test('edge cases', () => {
        // Period equals data length
        const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        const result = wasm.swma_js(data, 10);
        
        // All but last value should be NaN
        for (let i = 0; i < 9; i++) {
            assert(isNaN(result[i]), `Index ${i} should be NaN`);
        }
        assert(!isNaN(result[9]), 'Last value should not be NaN');
        
        // Period 1 returns input as-is
        const data2 = new Float64Array([1, 2, 3]);
        const result2 = wasm.swma_js(data2, 1);
        assertArrayClose(result2, data2, 1e-10, 'Period 1 should return input as-is');
    });
    
    test('performance with large dataset', () => {
        const data = new Float64Array(100000);
        for (let i = 0; i < data.length; i++) {
            data[i] = Math.random() * 100;
        }
        const period = 20;
        
        const start = performance.now();
        const result = wasm.swma_js(data, period);
        const elapsed = performance.now() - start;
        
        assert.strictEqual(result.length, data.length, 'Result length should match input');
        assert(elapsed < 1000, `Should process 100k elements in under 1 second, took ${elapsed}ms`);
        
        // Verify warmup period
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(result[i]), `Index ${i} should be NaN`);
        }
        assert(!isNaN(result[period - 1]), 'First non-NaN value should be at period-1');
    });
    
    // Zero-copy API tests
    test('SWMA zero-copy API', () => {
        const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        const period = 5;
        
        // Allocate buffer
        const ptr = wasm.swma_alloc(data.length);
        assert(ptr !== 0, 'Failed to allocate memory');
        
        // Create view into WASM memory
        const memView = new Float64Array(
            wasm.__wasm.memory.buffer,
            ptr,
            data.length
        );
        
        // Copy data into WASM memory
        memView.set(data);
        
        // Compute SWMA in-place
        try {
            wasm.swma_into(ptr, ptr, data.length, period);
            
            // Verify results match regular API
            const regularResult = wasm.swma_js(data, period);
            for (let i = 0; i < data.length; i++) {
                if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                    continue; // Both NaN is OK
                }
                assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                       `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
            }
        } finally {
            // Always free memory
            wasm.swma_free(ptr, data.length);
        }
    });
    
    test('SWMA zero-copy with large dataset', () => {
        const size = 100000;
        const data = new Float64Array(size);
        for (let i = 0; i < size; i++) {
            data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
        }
        
        const ptr = wasm.swma_alloc(size);
        assert(ptr !== 0, 'Failed to allocate large buffer');
        
        try {
            const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
            memView.set(data);
            
            wasm.swma_into(ptr, ptr, size, 10);
            
            // Recreate view in case memory grew
            const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
            
            // Check warmup period has NaN
            for (let i = 0; i < 9; i++) {
                assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
            }
            
            // Check after warmup has values
            for (let i = 9; i < Math.min(100, size); i++) {
                assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
            }
        } finally {
            wasm.swma_free(ptr, size);
        }
    });
    
    test('SWMA zero-copy error handling', () => {
        // Test null pointer
        assert.throws(() => {
            wasm.swma_into(0, 0, 10, 5);
        }, /null pointer|Null pointer provided/i);
        
        // Test invalid parameters with allocated memory
        const ptr = wasm.swma_alloc(10);
        try {
            // Invalid period
            assert.throws(() => {
                wasm.swma_into(ptr, ptr, 10, 0);
            }, /Invalid period/);
            
            // Period exceeds length
            assert.throws(() => {
                wasm.swma_into(ptr, ptr, 10, 11);
            }, /Invalid period/);
        } finally {
            wasm.swma_free(ptr, 10);
        }
    });
    
    test('SWMA zero-copy memory management', () => {
        // Allocate and free multiple times to ensure no leaks
        const sizes = [100, 1000, 10000, 100000];
        
        for (const size of sizes) {
            const ptr = wasm.swma_alloc(size);
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
            wasm.swma_free(ptr, size);
        }
    });
    
    // Batch unified API tests
    test('SWMA batch - unified API with single parameter', () => {
        const close = new Float64Array(testData.close.slice(0, 100));
        
        const result = wasm.swma_batch(close, {
            period_range: [5, 5, 0]
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
        assert.strictEqual(combo.period, 5);
        
        // Compare with old API
        const oldResult = wasm.swma_js(close, 5);
        for (let i = 0; i < oldResult.length; i++) {
            if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
                   `Value mismatch at index ${i}`);
        }
    });
    
    test('SWMA batch - unified API with multiple parameters', () => {
        const close = new Float64Array(testData.close.slice(0, 50));
        
        const result = wasm.swma_batch(close, {
            period_range: [5, 9, 2]  // 5, 7, 9
        });
        
        // Should have 3 combinations
        assert.strictEqual(result.rows, 3);
        assert.strictEqual(result.cols, 50);
        assert.strictEqual(result.combos.length, 3);
        assert.strictEqual(result.values.length, 150);
        
        // Verify each combo
        const expectedPeriods = [5, 7, 9];
        for (let i = 0; i < expectedPeriods.length; i++) {
            assert.strictEqual(result.combos[i].period, expectedPeriods[i]);
        }
        
        // Extract and verify first row
        const firstRow = result.values.slice(0, result.cols);
        const oldResult = wasm.swma_js(close, 5);
        for (let i = 0; i < oldResult.length; i++) {
            if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
                   `Value mismatch at index ${i}`);
        }
    });
    
    test('SWMA batch - unified API error handling', () => {
        const close = new Float64Array(testData.close.slice(0, 10));
        
        // Invalid config structure
        assert.throws(() => {
            wasm.swma_batch(close, {
                period_range: [5, 5]  // Missing step
            });
        }, /Invalid config/);
        
        // Missing required field
        assert.throws(() => {
            wasm.swma_batch(close, {});
        }, /Invalid config/);
        
        // Invalid data type
        assert.throws(() => {
            wasm.swma_batch(close, {
                period_range: "invalid"
            });
        }, /Invalid config/);
    });
    
    test('SWMA batch edge cases', () => {
        const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        
        // Single value sweep
        const singleBatch = wasm.swma_batch(close, {
            period_range: [5, 5, 1]
        });
        
        assert.strictEqual(singleBatch.values.length, 10);
        assert.strictEqual(singleBatch.combos.length, 1);
        
        // Step larger than range
        const largeBatch = wasm.swma_batch(close, {
            period_range: [5, 7, 10]  // Step larger than range
        });
        
        // Should only have period=5
        assert.strictEqual(largeBatch.values.length, 10);
        assert.strictEqual(largeBatch.combos.length, 1);
        
        // Empty data should throw
        assert.throws(() => {
            wasm.swma_batch(new Float64Array([]), {
                period_range: [5, 5, 0]
            });
        }, /Input data slice is empty/);
    });
});