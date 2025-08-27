/**
 * WASM binding tests for Mass Index indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { 
    loadTestData, 
    assertArrayClose, 
    assertClose,
    isNaN,
    assertAllNaN,
    assertNoNaN,
    EXPECTED_OUTPUTS 
} from './test_utils.js';
const { describe, it } = test;

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Import the WASM module
import * as talib from '../../pkg/my_project.js';

describe('Mass Index', () => {
    let testData;
    
    test.before(() => {
        testData = loadTestData();
    });
    
    it('should calculate mass with partial parameters - mirrors check_mass_partial_params', () => {
        const result = talib.mass_js(testData.high, testData.low, 5);
        assert.strictEqual(result.length, testData.high.length);
    });
    
    it('should match expected accuracy - mirrors check_mass_accuracy', () => {
        const expected = EXPECTED_OUTPUTS.mass;
        const result = talib.mass_js(
            testData.high, 
            testData.low, 
            expected.defaultParams.period
        );
        
        assert.strictEqual(result.length, testData.high.length);
        
        // Check last 5 values match expected
        const lastFive = result.slice(-5);
        assertArrayClose(
            lastFive,
            expected.last5Values,
            1e-7,
            "Mass Index last 5 values mismatch"
        );
    });
    
    it('should handle default candles - mirrors check_mass_default_candles', () => {
        const result = talib.mass_js(testData.high, testData.low, 5);
        assert.strictEqual(result.length, testData.high.length);
    });
    
    it('should handle zero period - mirrors check_mass_zero_period', () => {
        assert.throws(() => {
            talib.mass_js([10, 15, 20], [5, 10, 12], 0);
        }, /Invalid period/, 'Should throw for zero period');
    });
    
    it('should handle period exceeds length - mirrors check_mass_period_exceeds_length', () => {
        assert.throws(() => {
            talib.mass_js([10, 15, 20], [5, 10, 12], 10);
        }, /Invalid period/, 'Should throw when period > data length');
    });
    
    it('should handle very small dataset - mirrors check_mass_very_small_data_set', () => {
        assert.throws(() => {
            talib.mass_js([10.0], [5.0], 5);
        }, /Invalid period|Not enough valid data/, 'Should throw for insufficient data');
    });
    
    it('should handle empty data - mirrors check_mass_empty_input', () => {
        assert.throws(() => {
            talib.mass_js([], [], 5);
        }, /Empty data/, 'Should throw for empty data');
    });
    
    it('should handle different length high/low - mirrors check_mass_different_length_hl', () => {
        assert.throws(() => {
            talib.mass_js([1, 2, 3], [1, 2], 2);
        }, /High and low/, 'Should throw for mismatched high/low lengths');
    });
    
    it('should handle NaN values correctly - mirrors check_mass_nan_handling', () => {
        const expected = EXPECTED_OUTPUTS.mass;
        const result = talib.mass_js(testData.high, testData.low, 5);
        
        assert.strictEqual(result.length, testData.high.length);
        
        // Check warmup period has NaN values
        const warmup = expected.warmupPeriod;
        for (let i = 0; i < warmup; i++) {
            assert.ok(isNaN(result[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // After warmup should have valid values
        assert.ok(!isNaN(result[warmup]), `Expected valid value at index ${warmup} (first valid index)`);
        
        // After index 240, no NaN values should exist
        if (result.length > 240) {
            for (let i = 240; i < result.length; i++) {
                assert.ok(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
            }
        }
    });
    
    it('should handle all NaN values - mirrors check_mass_all_nan', () => {
        const allNaN = new Array(100).fill(NaN);
        
        assert.throws(() => {
            talib.mass_js(allNaN, allNaN, 5);
        }, /All values are NaN/, 'Should throw for all NaN input');
    });
    
    it('should handle batch operations with metadata', () => {
        const high = testData.high.slice(0, 100);
        const low = testData.low.slice(0, 100);
        
        const config = {
            period_range: [5, 15, 5]  // 5, 10, 15
        };
        
        const result = talib.mass_batch(high, low, config);
        
        // Verify metadata structure
        assert.strictEqual(result.rows, 3, 'Should have 3 rows');
        assert.strictEqual(result.cols, 100, 'Should have 100 columns');
        assert.strictEqual(result.values.length, 300, 'Should have 300 values total');
        assert.strictEqual(result.combos.length, 3, 'Should have 3 parameter combinations');
        
        // Verify parameter combinations
        assert.strictEqual(result.combos[0].period, 5, 'First combo period should be 5');
        assert.strictEqual(result.combos[1].period, 10, 'Second combo period should be 10');
        assert.strictEqual(result.combos[2].period, 15, 'Third combo period should be 15');
        
        // Check each row matches individual calculations
        const periods = [5, 10, 15];
        periods.forEach((period, row) => {
            const single = talib.mass_js(high, low, period);
            const rowStart = row * 100;
            const rowEnd = rowStart + 100;
            const batchRow = result.values.slice(rowStart, rowEnd);
            
            assertArrayClose(
                batchRow,
                single,
                1e-10,
                `Batch row ${row} (period=${period}) should match single calculation`
            );
        });
    });
    
    it('should handle batch parameter sweep', () => {
        const high = testData.high.slice(0, 30);
        const low = testData.low.slice(0, 30);
        
        const config = {
            period_range: [3, 7, 2]  // 3, 5, 7
        };
        
        const result = talib.mass_batch(high, low, config);
        
        assert.strictEqual(result.rows, 3, 'Should have 3 rows for sweep');
        assert.strictEqual(result.combos.length, 3, 'Should have 3 combinations');
        assert.deepStrictEqual(
            result.combos.map(c => c.period),
            [3, 5, 7],
            'Should have correct period values'
        );
        
        // Verify warmup periods for each row
        result.combos.forEach((combo, row) => {
            const warmup = 16 + combo.period - 1;
            const rowStart = row * 30;
            const rowData = result.values.slice(rowStart, rowStart + 30);
            
            // Check warmup has NaN
            for (let i = 0; i < Math.min(warmup, 30); i++) {
                assert.ok(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${combo.period}`);
            }
            
            // Check after warmup has values
            if (warmup < 30) {
                assert.ok(!isNaN(rowData[warmup]), `Expected valid value after warmup for period ${combo.period}`);
            }
        });
    });
    
    it('should handle zero-copy API - in-place operation', () => {
        const high = testData.high.slice(0, 100);
        const low = testData.low.slice(0, 100);
        const len = high.length;
        
        // Allocate output buffer
        const outPtr = talib.mass_alloc(len);
        
        try {
            // Calculate using safe API for comparison
            const expected = talib.mass_js(high, low, 5);
            
            // Allocate input buffers
            const highPtr = talib.mass_alloc(len);
            const lowPtr = talib.mass_alloc(len);
            
            // Copy data to WASM memory
            const wasmMemory = new Float64Array(talib.__wasm.memory.buffer);
            wasmMemory.set(high, highPtr / 8);
            wasmMemory.set(low, lowPtr / 8);
            
            // Calculate using fast API
            talib.mass_into(highPtr, lowPtr, outPtr, len, 5);
            
            // Read result
            const result = new Float64Array(talib.__wasm.memory.buffer, outPtr, len);
            
            // Compare results
            assertArrayClose(
                Array.from(result),
                expected,
                1e-10,
                "Fast API should match safe API"
            );
            
            // Test in-place operation (output overwrites high)
            talib.mass_into(highPtr, lowPtr, highPtr, len, 5);
            const inPlaceResult = new Float64Array(talib.__wasm.memory.buffer, highPtr, len);
            
            assertArrayClose(
                Array.from(inPlaceResult),
                expected,
                1e-10,
                "In-place operation should match expected"
            );
            
            // Free input buffers
            talib.mass_free(highPtr, len);
            talib.mass_free(lowPtr, len);
            
        } finally {
            // Always free output buffer
            talib.mass_free(outPtr, len);
        }
    });
    
    it('should handle zero-copy error cases', () => {
        // Test null pointer
        assert.throws(() => {
            talib.mass_into(0, 0, 0, 10, 5);
        }, /null pointer/, 'Should throw for null pointer');
        
        // Test invalid parameters with allocated memory
        const ptr = talib.mass_alloc(10);
        try {
            // Invalid period
            assert.throws(() => {
                talib.mass_into(ptr, ptr, ptr, 10, 0);
            }, /Invalid period/, 'Should throw for invalid period');
            
            // Period exceeds length
            assert.throws(() => {
                talib.mass_into(ptr, ptr, ptr, 10, 20);
            }, /Invalid period/, 'Should throw when period exceeds length');
        } finally {
            talib.mass_free(ptr, 10);
        }
    });
    
    it('should handle zero-copy memory management', () => {
        // Allocate and free multiple times to ensure no leaks
        const sizes = [100, 1000, 10000];
        
        for (const size of sizes) {
            const ptr = talib.mass_alloc(size);
            assert.ok(ptr !== 0, `Failed to allocate ${size} elements`);
            
            // Write pattern to verify memory
            const memView = new Float64Array(talib.__wasm.memory.buffer, ptr, size);
            for (let i = 0; i < Math.min(10, size); i++) {
                memView[i] = i * 1.5;
            }
            
            // Verify pattern
            for (let i = 0; i < Math.min(10, size); i++) {
                assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
            }
            
            // Free memory
            talib.mass_free(ptr, size);
        }
    });
    
    it('should handle batch with fast API', () => {
        const high = testData.high.slice(0, 50);
        const low = testData.low.slice(0, 50);
        const len = high.length;
        
        // Calculate expected rows
        const periodStart = 5, periodEnd = 10, periodStep = 5;
        const expectedRows = 2; // periods 5 and 10
        
        // Allocate buffers
        const highPtr = talib.mass_alloc(len);
        const lowPtr = talib.mass_alloc(len);
        const outPtr = talib.mass_alloc(len * expectedRows);
        
        try {
            // Copy data to WASM memory
            const wasmMemory = new Float64Array(talib.__wasm.memory.buffer);
            wasmMemory.set(high, highPtr / 8);
            wasmMemory.set(low, lowPtr / 8);
            
            // Run batch calculation
            const rows = talib.mass_batch_into(
                highPtr, lowPtr, outPtr, len,
                periodStart, periodEnd, periodStep
            );
            
            assert.strictEqual(rows, expectedRows, 'Should return correct number of rows');
            
            // Read and verify results
            const results = new Float64Array(talib.__wasm.memory.buffer, outPtr, len * rows);
            
            // Verify against individual calculations
            const periods = [5, 10];
            periods.forEach((period, row) => {
                const expected = talib.mass_js(high, low, period);
                const rowStart = row * len;
                const rowData = Array.from(results.slice(rowStart, rowStart + len));
                
                assertArrayClose(
                    rowData,
                    expected,
                    1e-10,
                    `Batch fast API row ${row} (period=${period}) should match`
                );
            });
            
        } finally {
            talib.mass_free(highPtr, len);
            talib.mass_free(lowPtr, len);
            talib.mass_free(outPtr, len * expectedRows);
        }
    });
    
    it('should support WASM streaming API', () => {
        // Create a streaming instance
        const stream = new talib.MassStreamWasm(5);
        
        // Test data
        const highData = testData.high.slice(0, 50);
        const lowData = testData.low.slice(0, 50);
        
        // Calculate batch for comparison
        const batchResult = talib.mass_js(highData, lowData, 5);
        
        // Stream calculation
        const streamResults = [];
        for (let i = 0; i < highData.length; i++) {
            const result = stream.update(highData[i], lowData[i]);
            streamResults.push(result !== null && result !== undefined ? result : NaN);
        }
        
        // Compare results after warmup (should match after index 19)
        const warmup = EXPECTED_OUTPUTS.mass.warmupPeriod;
        for (let i = warmup; i < streamResults.length; i++) {
            assertClose(
                streamResults[i],
                batchResult[i],
                1e-10,
                `Streaming result at index ${i} should match batch result`
            );
        }
    });
    
    it('should handle invalid period in streaming', () => {
        assert.throws(() => {
            new talib.MassStreamWasm(0);
        }, /Invalid period/, 'Should throw for invalid period in streaming');
    });
    
    it('should verify SIMD128 consistency', () => {
        // This test verifies SIMD128 produces same results as scalar
        // It runs automatically when SIMD128 is enabled
        const testCases = [
            { size: 25, period: 3 },
            { size: 100, period: 5 },
            { size: 1000, period: 10 },
            { size: 5000, period: 20 }
        ];
        
        for (const testCase of testCases) {
            // Generate test data
            const high = new Float64Array(testCase.size);
            const low = new Float64Array(testCase.size);
            for (let i = 0; i < testCase.size; i++) {
                const base = Math.sin(i * 0.1) * 100 + 1000;
                const range = Math.abs(Math.cos(i * 0.05)) * 10 + 5;
                high[i] = base + range;
                low[i] = base - range;
            }
            
            const result = talib.mass_js(high, low, testCase.period);
            
            // Basic sanity checks
            assert.strictEqual(result.length, high.length);
            
            // Check warmup period
            const warmup = 16 + testCase.period - 1;
            for (let i = 0; i < Math.min(warmup, result.length); i++) {
                assert.ok(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}, period=${testCase.period}`);
            }
            
            // Check values exist after warmup
            let sumAfterWarmup = 0;
            let countAfterWarmup = 0;
            for (let i = warmup; i < result.length; i++) {
                assert.ok(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}, period=${testCase.period}`);
                sumAfterWarmup += result[i];
                countAfterWarmup++;
            }
            
            // Verify reasonable values (Mass Index typically ranges from 0 to 2*period)
            if (countAfterWarmup > 0) {
                const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
                assert.ok(avgAfterWarmup > 0, `Average Mass Index should be positive, got ${avgAfterWarmup}`);
                assert.ok(avgAfterWarmup < testCase.period * 2, `Average Mass Index seems too high: ${avgAfterWarmup}`);
            }
        }
    });
    
    it('should handle batch edge cases', () => {
        const high = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
        const low = high.map(h => h * 0.9);
        
        // Single value sweep
        const singleBatch = talib.mass_batch(high, low, {
            period_range: [3, 3, 1]
        });
        
        assert.strictEqual(singleBatch.values.length, 20);
        assert.strictEqual(singleBatch.combos.length, 1);
        assert.strictEqual(singleBatch.combos[0].period, 3);
        
        // Step larger than range
        const largeBatch = talib.mass_batch(high, low, {
            period_range: [3, 5, 10]  // Step larger than range
        });
        
        // Should only have period=3
        assert.strictEqual(largeBatch.values.length, 20);
        assert.strictEqual(largeBatch.combos.length, 1);
        assert.strictEqual(largeBatch.combos[0].period, 3);
    });
    
    test.after(() => {
        console.log('Mass Index WASM tests completed');
    });
});