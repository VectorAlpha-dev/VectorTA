/**
 * WASM binding tests for TEMA indicator.
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

test.describe('TEMA (Triple Exponential Moving Average)', () => {
    test('basic functionality', () => {
        const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        const period = 3;
        
        const result = wasm.tema_js(data, period);
        
        assert.strictEqual(result.length, data.length, 'Result length should match input length');
        
        // Check warmup period ((period-1)*3 = 6 for period=3)
        for (let i = 0; i < 6; i++) {
            assert(isNaN(result[i]), `Index ${i} should be NaN during warmup`);
        }
        
        // Check that remaining values are not NaN
        for (let i = 6; i < result.length; i++) {
            assert(!isNaN(result[i]), `Index ${i} should not be NaN after warmup`);
        }
    });
    
    test('error handling - empty input', () => {
        const data = new Float64Array(0);
        const period = 3;
        
        assert.throws(
            () => wasm.tema_js(data, period),
            /Input data slice is empty/,
            'Should throw error for empty input'
        );
    });
    
    test('error handling - all NaN values', () => {
        const data = new Float64Array(10).fill(NaN);
        const period = 3;
        
        assert.throws(
            () => wasm.tema_js(data, period),
            /All values are NaN/,
            'Should throw error for all NaN values'
        );
    });
    
    test('error handling - invalid period', () => {
        const data = new Float64Array([1, 2, 3, 4, 5]);
        
        // Period exceeds data length
        assert.throws(
            () => wasm.tema_js(data, 6),
            /Invalid period/,
            'Should throw error when period exceeds data length'
        );
        
        // Period is zero
        assert.throws(
            () => wasm.tema_js(data, 0),
            /Invalid period/,
            'Should throw error for zero period'
        );
    });
    
    test('error handling - not enough valid data', () => {
        // First 8 values are NaN, only 2 valid values, but need period 9
        const data = new Float64Array([NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 1, 2]);
        const period = 9;
        
        assert.throws(
            () => wasm.tema_js(data, period),
            /Not enough valid data/,
            'Should throw error when not enough valid data after NaN values'
        );
    });
    
    test('leading NaN values', () => {
        const data = new Float64Array([NaN, NaN, 1, 2, 3, 4, 5, 6, 7, 8]);
        const period = 2;
        
        const result = wasm.tema_js(data, period);
        
        // First valid index is 2, TEMA warmup is (period-1)*3 = 3
        // So first valid output is at index 2+3 = 5
        for (let i = 0; i < 5; i++) {
            assert(isNaN(result[i]), `Index ${i} should be NaN`);
        }
        for (let i = 5; i < result.length; i++) {
            assert(!isNaN(result[i]), `Index ${i} should not be NaN`);
        }
    });
    
    test('compare with Rust implementation', async () => {
        const close = new Float64Array(testData.close);
        const period = 9;
        
        const result = wasm.tema_js(close, period);
        await compareWithRust('tema', result, 'close', { period });
    });
    
    test('batch calculation', () => {
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = Math.random() * 100;
        }
        
        const minPeriod = 5;
        const maxPeriod = 15;
        const stepPeriod = 2;
        
        // Use the deprecated API for backward compatibility test
        const values = wasm.tema_batch_js(data, minPeriod, maxPeriod, stepPeriod);
        
        // Create metadata manually since metadata_js was removed
        const metadata = [];
        for (let p = minPeriod; p <= maxPeriod; p += stepPeriod) {
            metadata.push(p);
        }
        
        // Expected periods: 5, 7, 9, 11, 13, 15
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
            const individual = wasm.tema_js(data, period);
            
            // Extract row from batch result
            const row = new Float64Array(cols);
            for (let j = 0; j < cols; j++) {
                row[j] = values[i * cols + j];
            }
            
            assertArrayClose(row, individual, 1e-10, `Batch row ${i} should match individual calculation`);
        }
    });
    
    test('warmup period validation', () => {
        const data = new Float64Array(50);
        for (let i = 0; i < 50; i++) {
            data[i] = i + 1;
        }
        
        // Test different periods
        const periods = [3, 5, 7, 9];
        for (const period of periods) {
            const result = wasm.tema_js(data, period);
            const warmup = (period - 1) * 3;
            
            // Check warmup period
            for (let i = 0; i < warmup; i++) {
                assert(isNaN(result[i]), `Period ${period}: Index ${i} should be NaN`);
            }
            
            // Check values after warmup
            for (let i = warmup; i < result.length; i++) {
                assert(!isNaN(result[i]), `Period ${period}: Index ${i} should not be NaN`);
            }
        }
    });
    
    test('edge cases', () => {
        // Period equals data length
        const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        const result = wasm.tema_js(data, 10);
        
        // All but possibly last value should be NaN
        // Warmup is (10-1)*3 = 27, which exceeds data length
        assertAllNaN(result, 'All values should be NaN when warmup exceeds data length');
        
        // Period 1 returns input as-is
        const data2 = new Float64Array([1, 2, 3]);
        const result2 = wasm.tema_js(data2, 1);
        assertArrayClose(result2, data2, 1e-10, 'Period 1 should return input as-is');
    });
    
    test('performance with large dataset', () => {
        const data = new Float64Array(100000);
        for (let i = 0; i < data.length; i++) {
            data[i] = Math.random() * 100;
        }
        const period = 20;
        
        const start = performance.now();
        const result = wasm.tema_js(data, period);
        const elapsed = performance.now() - start;
        
        assert.strictEqual(result.length, data.length, 'Result length should match input');
        assert(elapsed < 1000, `Should process 100k elements in under 1 second, took ${elapsed}ms`);
        
        // Verify warmup period
        const warmup = (period - 1) * 3;
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(result[i]), `Index ${i} should be NaN`);
        }
        assert(!isNaN(result[warmup]), 'First non-NaN value should be at warmup index');
    });
    
    test('reinput - using output as input', async () => {
        const close = new Float64Array(testData.close);
        
        // First TEMA with period 9
        const firstPeriod = 9;
        const firstResult = wasm.tema_js(close, firstPeriod);
        assert.strictEqual(firstResult.length, close.length);
        
        // Use first result as input for second TEMA with period 5
        const secondPeriod = 5;
        const secondResult = wasm.tema_js(firstResult, secondPeriod);
        assert.strictEqual(secondResult.length, firstResult.length);
        
        // Verify second result is not all NaN
        // First TEMA has warmup of (9-1)*3 = 24
        // Second TEMA adds warmup of (5-1)*3 = 12
        // So total warmup should be 24 + 12 = 36
        for (let i = 0; i < 36; i++) {
            assert(isNaN(secondResult[i]), `Index ${i} should be NaN`);
        }
        for (let i = 36; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Index ${i} should not be NaN`);
        }
        
        // Verify the output values are reasonable
        const validValues = secondResult.slice(36);
        const uniqueValues = new Set(validValues);
        assert(uniqueValues.size > 1, 'Should have multiple different values');
        
        // Check all values are finite
        for (const val of validValues) {
            assert(Number.isFinite(val), 'All values should be finite');
        }
        
        // Check has variance
        const mean = validValues.reduce((a, b) => a + b, 0) / validValues.length;
        const variance = validValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / validValues.length;
        assert(variance > 0, 'Should have non-zero variance');
    });
    
    test('accuracy check with simple data', () => {
        // Simple test data where we can verify calculations
        const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        const period = 3;
        
        const result = wasm.tema_js(data, period);
        
        // Warmup period is (3-1)*3 = 6
        for (let i = 0; i < 6; i++) {
            assert(isNaN(result[i]), `Index ${i} should be NaN`);
        }
        
        // First valid values should be reasonable
        assert(result[6] > 6.0 && result[6] < 8.0, 'First valid value should be near trend');
        assert(result[7] > 7.0 && result[7] < 9.0, 'Second valid value should be near trend');
        assert(result[8] > 8.0 && result[8] < 10.0, 'Third valid value should be near trend');
        assert(result[9] > 9.0 && result[9] < 11.0, 'Fourth valid value should be near trend');
    });
    
    test('batch metadata structure', () => {
        // Test with the new ergonomic API (need at least 15 data points for period 15)
        const result = wasm.tema_batch(new Float64Array(20), {
            period_range: [5, 15, 2]
        });
        
        // Should have periods: 5, 7, 9, 11, 13, 15
        const expectedPeriods = [5, 7, 9, 11, 13, 15];
        assert.strictEqual(result.combos.length, expectedPeriods.length);
        
        for (let i = 0; i < expectedPeriods.length; i++) {
            assert.strictEqual(result.combos[i].period, expectedPeriods[i], `Period ${i} should match`);
        }
    });
    
    test('batch with single period', () => {
        const data = new Float64Array(50);
        for (let i = 0; i < 50; i++) {
            data[i] = Math.sin(i * 0.1) * 10 + 50;
        }
        
        // Single period (step = 0)
        const singleBatch = wasm.tema_batch_js(data, 9, 9, 0);
        const singleCalc = wasm.tema_js(data, 9);
        
        assert.strictEqual(singleBatch.length, data.length);
        assertArrayClose(singleBatch, singleCalc, 1e-10, 'Single period batch should match individual');
    });
    
    test('batch warmup periods', () => {
        const data = new Float64Array(50);
        for (let i = 0; i < 50; i++) {
            data[i] = i + 1;
        }
        
        const values = wasm.tema_batch_js(data, 3, 7, 2);
        const periods = [3, 5, 7];
        
        // Check each period has correct warmup
        for (let row = 0; row < periods.length; row++) {
            const period = periods[row];
            const warmup = (period - 1) * 3;
            
            for (let col = 0; col < data.length; col++) {
                const idx = row * data.length + col;
                if (col < warmup) {
                    assert(isNaN(values[idx]), `Row ${row} col ${col} should be NaN`);
                } else {
                    assert(!isNaN(values[idx]), `Row ${row} col ${col} should not be NaN`);
                }
            }
        }
    });
    
    test('very small dataset', () => {
        // Test with minimum size for different periods
        for (const period of [1, 2, 3, 4, 5]) {
            const data = new Float64Array(period);
            for (let i = 0; i < period; i++) {
                data[i] = i + 1;
            }
            
            if (period === 1) {
                // Period 1 should work and return input
                const result = wasm.tema_js(data, period);
                assertArrayClose(result, data, 1e-10, 'Period 1 should return input');
            } else {
                // Should work but may have all NaN
                const result = wasm.tema_js(data, period);
                assert.strictEqual(result.length, data.length);
                
                const warmup = (period - 1) * 3;
                if (warmup >= data.length) {
                    assertAllNaN(result, 'Should be all NaN when warmup exceeds data length');
                }
            }
        }
    });

    // New API tests
    test('TEMA zero-copy API', () => {
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = Math.sin(i * 0.1) * 10 + 50;
        }
        const period = 14;

        // Allocate buffer
        const ptr = wasm.tema_alloc(data.length);
        assert(ptr !== 0, 'Failed to allocate memory');

        try {
            // Create view into WASM memory
            const memView = new Float64Array(
                wasm.__wasm.memory.buffer,
                ptr,
                data.length
            );

            // Copy data into WASM memory
            memView.set(data);

            // Compute TEMA in-place
            wasm.tema_into(ptr, ptr, data.length, period);

            // Verify results match regular API
            const regularResult = wasm.tema_js(data, period);
            for (let i = 0; i < data.length; i++) {
                if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                    continue; // Both NaN is OK
                }
                assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                       `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
            }
        } finally {
            // Always free memory
            wasm.tema_free(ptr, data.length);
        }
    });

    test('TEMA ergonomic batch API', () => {
        const close = new Float64Array(testData.close.slice(0, 50));
        
        const result = wasm.tema_batch(close, {
            period_range: [5, 15, 5]  // 5, 10, 15
        });

        // Check structure
        assert(result.values, 'Should have values array');
        assert(result.combos, 'Should have combos array');
        assert(typeof result.rows === 'number', 'Should have rows count');
        assert(typeof result.cols === 'number', 'Should have cols count');

        // Check dimensions
        assert.strictEqual(result.rows, 3);
        assert.strictEqual(result.cols, 50);
        assert.strictEqual(result.combos.length, 3);
        assert.strictEqual(result.values.length, 150);

        // Check combos structure
        assert.strictEqual(result.combos[0].period, 5);
        assert.strictEqual(result.combos[1].period, 10);
        assert.strictEqual(result.combos[2].period, 15);

        // Verify against individual calculations
        for (let i = 0; i < result.combos.length; i++) {
            const period = result.combos[i].period;
            const individual = wasm.tema_js(close, period);
            const batchRow = result.values.slice(i * 50, (i + 1) * 50);
            assertArrayClose(batchRow, individual, 1e-10,
                           `Batch row ${i} (period ${period}) should match individual`);
        }
    });

    test('TEMA batch zero-copy API', () => {
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = Math.random() * 100;
        }

        const periods = { start: 5, end: 15, step: 5 }; // 3 periods
        const numCombos = 3;
        const totalSize = numCombos * data.length;

        // Allocate input and output buffers
        const inPtr = wasm.tema_alloc(data.length);
        const outPtr = wasm.tema_alloc(totalSize);

        try {
            // Copy input data
            const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
            inView.set(data);

            // Run batch computation
            const rows = wasm.tema_batch_into(
                inPtr, outPtr, data.length,
                periods.start, periods.end, periods.step
            );

            assert.strictEqual(rows, numCombos, 'Should return correct number of rows');

            // Verify results
            const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
            
            // Compare with ergonomic API
            const ergonomicResult = wasm.tema_batch(data, {
                period_range: [periods.start, periods.end, periods.step]
            });

            assertArrayClose(
                Array.from(outView),
                ergonomicResult.values,
                1e-10,
                'Zero-copy batch should match ergonomic batch'
            );
        } finally {
            wasm.tema_free(inPtr, data.length);
            wasm.tema_free(outPtr, totalSize);
        }
    });

    test('TEMA error handling - null pointers', () => {
        assert.throws(() => {
            wasm.tema_into(0, 0, 10, 9);
        }, /null pointer/i);

        // Test with allocated memory but invalid parameters
        const ptr = wasm.tema_alloc(10);
        try {
            // Invalid period
            assert.throws(() => {
                wasm.tema_into(ptr, ptr, 10, 0);
            }, /Invalid period/);
        } finally {
            wasm.tema_free(ptr, 10);
        }
    });

    test('TEMA batch memory leak prevention', () => {
        // Allocate and free multiple times to ensure no leaks
        const sizes = [100, 1000, 10000];
        
        for (const size of sizes) {
            const ptr = wasm.tema_alloc(size);
            assert(ptr !== 0, `Failed to allocate ${size} elements`);
            
            // Write pattern to verify memory
            const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
            for (let i = 0; i < Math.min(10, size); i++) {
                memView[i] = i * 1.5;
            }
            
            // Verify pattern
            for (let i = 0; i < Math.min(10, size); i++) {
                assert.strictEqual(memView[i], i * 1.5, `Memory at ${i} should match pattern`);
            }
            
            // Free memory
            wasm.tema_free(ptr, size);
        }
    });
});

test.after(() => {
    console.log('TEMA WASM tests completed');
});