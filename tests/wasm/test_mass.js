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
    
    it('should calculate mass with default parameters', () => {
        const result = talib.mass_js(testData.high, testData.low, 5);
        assert.strictEqual(result.length, testData.high.length);
        
        // Check some values are not NaN after warmup
        const lastValues = result.slice(-5);
        lastValues.forEach(val => {
            assert.ok(!isNaN(val), 'Result should not be NaN');
        });
    });
    
    it('should match expected accuracy', () => {
        const result = talib.mass_js(testData.high, testData.low, 5);
        
        // Expected values from Rust tests
        const expectedLastFive = [
            4.512263952194651,
            4.126178935431121,
            3.838738456245828,
            3.6450956734739375,
            3.6748009093527125
        ];
        
        const lastFive = result.slice(-5);
        lastFive.forEach((val, i) => {
            assert.ok(Math.abs(val - expectedLastFive[i]) < 1e-7, 
                `Value at index ${i} should match expected`);
        });
    });
    
    it('should handle invalid period', () => {
        assert.throws(() => {
            talib.mass_js(testData.high, testData.low, 0);
        }, /Invalid period/);
        
        assert.throws(() => {
            talib.mass_js([1, 2, 3], [1, 2, 3], 10);
        }, /Invalid period/);
    });
    
    it('should handle empty data', () => {
        assert.throws(() => {
            talib.mass_js([], [], 5);
        }, /Empty data/);
    });
    
    it('should handle different length high/low', () => {
        assert.throws(() => {
            talib.mass_js([1, 2, 3], [1, 2], 2);
        }, /High and low/);
    });
    
    it('should handle in-place operation with fast API', () => {
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
            for (let i = 0; i < len; i++) {
                assert.ok(Math.abs(result[i] - expected[i]) < 1e-10 || 
                         (isNaN(result[i]) && isNaN(expected[i])),
                    `Fast API should match safe API at index ${i}`);
            }
            
            // Test in-place operation (output overwrites high)
            talib.mass_into(highPtr, lowPtr, highPtr, len, 5);
            const inPlaceResult = new Float64Array(talib.__wasm.memory.buffer, highPtr, len);
            
            for (let i = 0; i < len; i++) {
                assert.ok(Math.abs(inPlaceResult[i] - expected[i]) < 1e-10 || 
                         (isNaN(inPlaceResult[i]) && isNaN(expected[i])),
                    `In-place operation should match expected at index ${i}`);
            }
            
            // Free input buffers
            talib.mass_free(highPtr, len);
            talib.mass_free(lowPtr, len);
            
        } finally {
            // Always free output buffer
            talib.mass_free(outPtr, len);
        }
    });
    
    it('should handle batch operations', () => {
        const high = testData.high.slice(0, 100);
        const low = testData.low.slice(0, 100);
        
        const config = {
            period_range: [5, 15, 5]  // 5, 10, 15
        };
        
        const result = talib.mass_batch(high, low, config);
        
        assert.strictEqual(result.rows, 3, 'Should have 3 rows');
        assert.strictEqual(result.cols, 100, 'Should have 100 columns');
        assert.strictEqual(result.values.length, 300, 'Should have 300 values total');
        assert.strictEqual(result.combos.length, 3, 'Should have 3 parameter combinations');
        
        // Check each row matches individual calculations
        const periods = [5, 10, 15];
        periods.forEach((period, row) => {
            const single = talib.mass_js(high, low, period);
            const rowStart = row * 100;
            const rowEnd = rowStart + 100;
            const batchRow = result.values.slice(rowStart, rowEnd);
            
            for (let i = 0; i < 100; i++) {
                assert.ok(Math.abs(batchRow[i] - single[i]) < 1e-10 || 
                         (isNaN(batchRow[i]) && isNaN(single[i])),
                    `Batch row ${row} should match single calculation at index ${i}`);
            }
        });
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
                
                for (let i = 0; i < len; i++) {
                    const actual = results[rowStart + i];
                    const exp = expected[i];
                    assert.ok(Math.abs(actual - exp) < 1e-10 || 
                             (isNaN(actual) && isNaN(exp)),
                        `Batch fast API row ${row} should match at index ${i}`);
                }
            });
            
        } finally {
            talib.mass_free(highPtr, len);
            talib.mass_free(lowPtr, len);
            talib.mass_free(outPtr, len * expectedRows);
        }
    });
    
    it('should handle all NaN values', () => {
        const allNaN = new Array(100).fill(NaN);
        
        assert.throws(() => {
            talib.mass_js(allNaN, allNaN, 5);
        }, /All values are NaN/);
    });
    
    it('should handle very small dataset', () => {
        assert.throws(() => {
            talib.mass_js([10.0], [5.0], 5);
        }, /Invalid period|Not enough valid data/);
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
        for (let i = 20; i < streamResults.length; i++) {
            assert.ok(Math.abs(streamResults[i] - batchResult[i]) < 1e-10,
                `Streaming result at index ${i} should match batch result`);
        }
    });
    
    it('should handle invalid period in streaming', () => {
        assert.throws(() => {
            new talib.MassStreamWasm(0);
        }, /Invalid period/);
    });
});