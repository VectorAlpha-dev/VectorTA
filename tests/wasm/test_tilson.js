import assert from 'assert';
import { describe, it, test } from 'node:test';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { loadTestData, assertArrayClose, assertAllNaN } from './test_utils.js';
import { compareWithRust } from './rust-comparison.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load WASM module
const wasmPath = join(__dirname, '../../pkg/my_project.js');
const importPath = process.platform === 'win32' 
    ? 'file:///' + wasmPath.replace(/\\/g, '/')
    : wasmPath;
const wasm = await import(importPath);

// Load test data
const testData = loadTestData();

describe('Tilson T3 Moving Average', () => {
    test('basic functionality', () => {
        const data = new Float64Array([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30
        ]);
        const period = 5;
        const volumeFactor = 0.0;
        
        const result = wasm.tilson_js(data, period, volumeFactor);
        
        assert(result instanceof Float64Array, 'Result should be Float64Array');
        assert.strictEqual(result.length, data.length, 'Result length should match input');
        
        // Tilson has warmup period of 6 * (period - 1) = 24 for period=5
        for (let i = 0; i < 24; i++) {
            assert(isNaN(result[i]), `Index ${i} should be NaN during warmup`);
        }
        
        // Check that remaining values are not NaN
        for (let i = 24; i < result.length; i++) {
            assert(!isNaN(result[i]), `Index ${i} should not be NaN after warmup`);
        }
    });
    
    test('error handling - empty input', () => {
        const data = new Float64Array(0);
        const period = 5;
        const volumeFactor = 0.0;
        
        assert.throws(
            () => wasm.tilson_js(data, period, volumeFactor),
            /Input data slice is empty/,
            'Should throw error for empty input'
        );
    });
    
    test('error handling - all NaN values', () => {
        const data = new Float64Array(10).fill(NaN);
        const period = 5;
        const volumeFactor = 0.0;
        
        assert.throws(
            () => wasm.tilson_js(data, period, volumeFactor),
            /All values are NaN/,
            'Should throw error for all NaN values'
        );
    });
    
    test('error handling - invalid period', () => {
        const data = new Float64Array([1, 2, 3, 4, 5]);
        
        // Period exceeds data length
        assert.throws(
            () => wasm.tilson_js(data, 6, 0.0),
            /Invalid period/,
            'Should throw error when period exceeds data length'
        );
        
        // Period is zero
        assert.throws(
            () => wasm.tilson_js(data, 0, 0.0),
            /Invalid period/,
            'Should throw error for zero period'
        );
    });
    
    test('error handling - not enough valid data', () => {
        // First 8 values are NaN, only 2 valid values, but need more for period 5
        const data = new Float64Array([NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 1, 2]);
        const period = 5;
        const volumeFactor = 0.0;
        
        assert.throws(
            () => wasm.tilson_js(data, period, volumeFactor),
            /Not enough valid data/,
            'Should throw error when not enough valid data after NaN values'
        );
    });
    
    test('error handling - invalid volume factor', () => {
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = Math.random() * 100;
        }
        
        // NaN volume factor
        assert.throws(
            () => wasm.tilson_js(data, 5, NaN),
            /Invalid volume factor/,
            'Should throw error for NaN volume factor'
        );
        
        // Infinite volume factor
        assert.throws(
            () => wasm.tilson_js(data, 5, Infinity),
            /Invalid volume factor/,
            'Should throw error for infinite volume factor'
        );
    });
    
    test('leading NaN values', () => {
        const data = new Float64Array([
            NaN, NaN, 1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26
        ]);
        const period = 3;
        const volumeFactor = 0.5;
        
        const result = wasm.tilson_js(data, period, volumeFactor);
        
        // First valid index is 2, Tilson warmup is 6 * (period-1) = 12
        // So first valid output is at index 2+12 = 14
        for (let i = 0; i < 14; i++) {
            assert(isNaN(result[i]), `Index ${i} should be NaN`);
        }
        for (let i = 14; i < result.length; i++) {
            assert(!isNaN(result[i]), `Index ${i} should not be NaN`);
        }
    });
    
    test('compare with Rust implementation', async () => {
        const close = new Float64Array(testData.close);
        const period = 5;
        const volumeFactor = 0.0;
        
        const result = wasm.tilson_js(close, period, volumeFactor);
        await compareWithRust('tilson', result, 'close', { 
            period,
            volume_factor: volumeFactor
        });
    });
    
    test('batch calculation', () => {
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = Math.random() * 100;
        }
        
        const periodStart = 3;
        const periodEnd = 7;
        const periodStep = 2;
        const vFactorStart = 0.0;
        const vFactorEnd = 0.6;
        const vFactorStep = 0.3;
        
        const values = wasm.tilson_batch_js(data, periodStart, periodEnd, periodStep,
                                           vFactorStart, vFactorEnd, vFactorStep);
        const metadata = wasm.tilson_batch_metadata_js(periodStart, periodEnd, periodStep,
                                                       vFactorStart, vFactorEnd, vFactorStep);
        
        // Expected combinations
        const expectedPeriods = [3, 5, 7];
        const expectedVFactors = [0.0, 0.3, 0.6];
        const expectedCombos = expectedPeriods.length * expectedVFactors.length;
        
        const rows = expectedCombos;
        const cols = data.length;
        
        assert.strictEqual(metadata.length, rows * 2, 'Metadata should contain periods and volume factors');
        assert.strictEqual(values.length, rows * cols, 'Values array size should be rows*cols');
        
        // Extract periods and volume factors from metadata
        const periods = new Float64Array(metadata.slice(0, rows));
        const vFactors = new Float64Array(metadata.slice(rows));
        
        // Verify combinations
        let comboIdx = 0;
        for (const period of expectedPeriods) {
            for (const vFactor of expectedVFactors) {
                assert.strictEqual(periods[comboIdx], period, `Period mismatch at ${comboIdx}`);
                assert(Math.abs(vFactors[comboIdx] - vFactor) < 1e-10, `Volume factor mismatch at ${comboIdx}`);
                comboIdx++;
            }
        }
    });
    
    test('warmup period validation', () => {
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = i + 1;
        }
        
        // Test different periods
        const periods = [2, 3, 4, 5, 6];
        for (const period of periods) {
            const result = wasm.tilson_js(data, period, 0.0);
            const warmup = 6 * (period - 1);
            
            // Check warmup period
            for (let i = 0; i < warmup && i < result.length; i++) {
                assert(isNaN(result[i]), `Period ${period}: Index ${i} should be NaN`);
            }
            
            // Check values after warmup
            for (let i = warmup; i < result.length; i++) {
                assert(!isNaN(result[i]), `Period ${period}: Index ${i} should not be NaN`);
            }
        }
    });
    
    test('edge cases', () => {
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = Math.random() * 100;
        }
        
        // Minimum valid period
        const result1 = wasm.tilson_js(data, 1, 0.0);
        // Period 1 has warmup = 6 * (1-1) = 0, so no NaN values
        assert(!isNaN(result1[0]), 'Period 1 should have no NaN values');
        
        // Maximum volume factor
        const result2 = wasm.tilson_js(data, 5, 1.0);
        const warmup = 6 * (5 - 1);
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(result2[i]), `Index ${i} should be NaN`);
        }
        for (let i = warmup; i < result2.length; i++) {
            assert(!isNaN(result2[i]), `Index ${i} should not be NaN`);
        }
    });
    
    test('reinput test', () => {
        const close = new Float64Array(testData.close);
        
        // First Tilson with period 5, volume_factor 0.0
        const firstResult = wasm.tilson_js(close, 5, 0.0);
        assert.strictEqual(firstResult.length, close.length);
        
        // Use first result as input for second Tilson
        const secondResult = wasm.tilson_js(firstResult, 3, 0.7);
        assert.strictEqual(secondResult.length, firstResult.length);
        
        // Verify warmup periods combine correctly
        // First Tilson has warmup of 24, second adds 12
        const totalWarmup = 36;
        for (let i = 0; i < totalWarmup && i < secondResult.length; i++) {
            assert(isNaN(secondResult[i]), `Index ${i} should be NaN in combined warmup`);
        }
        
        // Verify we have valid values after warmup
        if (secondResult.length > totalWarmup) {
            let hasValidValues = false;
            for (let i = totalWarmup; i < secondResult.length; i++) {
                if (!isNaN(secondResult[i])) {
                    hasValidValues = true;
                    break;
                }
            }
            assert(hasValidValues, 'Should have valid values after combined warmup');
        }
    });
    
    test('batch with single combination', () => {
        const data = new Float64Array(50);
        for (let i = 0; i < 50; i++) {
            data[i] = Math.random() * 100;
        }
        
        // Single combination
        const values = wasm.tilson_batch_js(data, 5, 5, 0, 0.7, 0.7, 0.0);
        const metadata = wasm.tilson_batch_metadata_js(5, 5, 0, 0.7, 0.7, 0.0);
        
        assert.strictEqual(values.length, data.length, 'Single batch should have data length');
        assert.strictEqual(metadata.length, 2, 'Metadata should have 2 values (period and volume factor)');
        assert.strictEqual(metadata[0], 5, 'Period should be 5');
        assert(Math.abs(metadata[1] - 0.7) < 1e-10, 'Volume factor should be 0.7');
        
        // Should match single calculation
        const singleResult = wasm.tilson_js(data, 5, 0.7);
        assertArrayClose(values, singleResult, 1e-10, 'Batch with single combo should match single calc');
    });
    
    test('volume factor effect', () => {
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = Math.sin(i * 0.1) * 50 + 50 + Math.random() * 5;
        }
        const period = 5;
        
        // Compare different volume factors
        const result0 = wasm.tilson_js(data, period, 0.0);
        const result5 = wasm.tilson_js(data, period, 0.5);
        const result9 = wasm.tilson_js(data, period, 0.9);
        
        const warmup = 6 * (period - 1);
        
        // Different volume factors should produce different results
        let differences0_5 = 0;
        let differences5_9 = 0;
        
        for (let i = warmup; i < data.length; i++) {
            if (Math.abs(result0[i] - result5[i]) > 1e-10) differences0_5++;
            if (Math.abs(result5[i] - result9[i]) > 1e-10) differences5_9++;
        }
        
        assert(differences0_5 > 0, 'Volume factor 0.0 vs 0.5 should produce different results');
        assert(differences5_9 > 0, 'Volume factor 0.5 vs 0.9 should produce different results');
    });
    
    test('batch warmup validation', () => {
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = i + 1;
        }
        
        const values = wasm.tilson_batch_js(data, 2, 6, 2, 0.0, 0.0, 0.0);
        const metadata = wasm.tilson_batch_metadata_js(2, 6, 2, 0.0, 0.0, 0.0);
        
        const periods = [2, 4, 6];
        const cols = data.length;
        
        for (let i = 0; i < periods.length; i++) {
            const period = periods[i];
            const warmup = 6 * (period - 1);
            
            // Extract row from batch result
            const row = new Float64Array(cols);
            for (let j = 0; j < cols; j++) {
                row[j] = values[i * cols + j];
            }
            
            // Check warmup
            for (let j = 0; j < warmup && j < cols; j++) {
                assert(isNaN(row[j]), `Period ${period}: Index ${j} should be NaN`);
            }
            
            // Check valid values
            for (let j = warmup; j < cols; j++) {
                assert(!isNaN(row[j]), `Period ${period}: Index ${j} should not be NaN`);
            }
        }
    });
    
    test('unified batch API with config object', () => {
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = Math.random() * 100;
        }
        
        const config = {
            period_range: [3, 7, 2],
            volume_factor_range: [0.0, 0.6, 0.3]
        };
        
        const result = wasm.tilson_batch(data, config);
        
        // Expected combinations
        const expectedPeriods = [3, 5, 7];
        const expectedVFactors = [0.0, 0.3, 0.6];
        const expectedCombos = expectedPeriods.length * expectedVFactors.length;
        
        assert.strictEqual(result.combos.length, expectedCombos, 'Should have correct number of combinations');
        assert.strictEqual(result.rows, expectedCombos, 'Rows should match combinations');
        assert.strictEqual(result.cols, data.length, 'Cols should match data length');
        assert.strictEqual(result.values.length, expectedCombos * data.length, 'Values array size should be rows*cols');
        
        // Verify parameter combinations
        let comboIdx = 0;
        for (const period of expectedPeriods) {
            for (const vFactor of expectedVFactors) {
                assert.strictEqual(result.combos[comboIdx].period, period, `Period mismatch at ${comboIdx}`);
                assert(Math.abs(result.combos[comboIdx].volume_factor - vFactor) < 1e-10, `Volume factor mismatch at ${comboIdx}`);
                comboIdx++;
            }
        }
    });
    
    test('zero-copy API', () => {
        const data = new Float64Array([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30
        ]);
        const period = 5;
        const volumeFactor = 0.0;
        
        // Allocate buffer
        const ptr = wasm.tilson_alloc(data.length);
        assert(ptr !== 0, 'Failed to allocate memory');
        
        // Create view into WASM memory
        const memView = new Float64Array(
            wasm.__wasm.memory.buffer,
            ptr,
            data.length
        );
        
        // Copy data into WASM memory
        memView.set(data);
        
        // Compute Tilson in-place
        try {
            wasm.tilson_into(ptr, ptr, data.length, period, volumeFactor);
            
            // Verify results match regular API
            const regularResult = wasm.tilson_js(data, period, volumeFactor);
            for (let i = 0; i < data.length; i++) {
                if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                    continue; // Both NaN is OK
                }
                assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                       `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
            }
        } finally {
            // Always free memory
            wasm.tilson_free(ptr, data.length);
        }
    });
    
    test('zero-copy batch API', () => {
        const data = new Float64Array(50);
        for (let i = 0; i < 50; i++) {
            data[i] = Math.random() * 100;
        }
        
        const periodStart = 3;
        const periodEnd = 5;
        const periodStep = 2;
        const vFactorStart = 0.0;
        const vFactorEnd = 0.6;
        const vFactorStep = 0.6;
        
        // Expected: periods [3, 5], vFactors [0.0, 0.6] = 4 combinations
        const expectedRows = 4;
        const totalSize = expectedRows * data.length;
        
        // Allocate buffers
        const inPtr = wasm.tilson_alloc(data.length);
        const outPtr = wasm.tilson_alloc(totalSize);
        
        assert(inPtr !== 0, 'Failed to allocate input buffer');
        assert(outPtr !== 0, 'Failed to allocate output buffer');
        
        try {
            // Copy data to input buffer
            const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
            inView.set(data);
            
            // Run batch computation
            const rows = wasm.tilson_batch_into(
                inPtr, outPtr, data.length,
                periodStart, periodEnd, periodStep,
                vFactorStart, vFactorEnd, vFactorStep
            );
            
            assert.strictEqual(rows, expectedRows, 'Should return correct number of rows');
            
            // Verify at least that we have data
            const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
            let hasNonNaN = false;
            for (let i = 0; i < totalSize; i++) {
                if (!isNaN(outView[i])) {
                    hasNonNaN = true;
                    break;
                }
            }
            assert(hasNonNaN, 'Output should contain some non-NaN values');
        } finally {
            wasm.tilson_free(inPtr, data.length);
            wasm.tilson_free(outPtr, totalSize);
        }
    });
    
    test('deprecated TilsonContext API', () => {
        const period = 5;
        const volumeFactor = 0.7;
        
        // Create context
        const context = new wasm.TilsonContext(period, volumeFactor);
        
        // Test warmup period
        assert.strictEqual(context.get_warmup_period(), 6 * (period - 1), 'Warmup period should be 6*(period-1)');
        
        // Feed data and verify warmup
        const testData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30];
        const results = [];
        
        for (let i = 0; i < testData.length; i++) {
            const result = context.update(testData[i]);
            results.push(result === undefined ? null : result);
        }
        
        // Check warmup period
        const warmup = 6 * (period - 1); // 24 for period=5
        for (let i = 0; i < warmup && i < results.length; i++) {
            assert(results[i] === null, `Should return null during warmup at index ${i}`);
        }
        
        // Should have values after warmup
        for (let i = warmup; i < results.length; i++) {
            assert(results[i] !== null && !isNaN(results[i]), `Should return valid value after warmup at index ${i}`);
        }
        
        // Test reset
        context.reset();
        const afterReset = context.update(10.0);
        assert(afterReset === null || afterReset === undefined, 'Should return null/undefined after reset');
        
        // Test error handling
        assert.throws(() => {
            new wasm.TilsonContext(0, 0.5);
        }, /Invalid period/, 'Should throw for invalid period');
        
        assert.throws(() => {
            new wasm.TilsonContext(5, NaN);
        }, /Invalid volume factor/, 'Should throw for NaN volume factor');
    });
});

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    console.log('Running Tilson T3 tests...');
}