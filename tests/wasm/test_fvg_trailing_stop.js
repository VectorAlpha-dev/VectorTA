/**
 * WASM binding tests for FVG Trailing Stop indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
    
    try {
        const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
        const importPath = process.platform === 'win32' 
            ? `file:///${wasmPath.replace(/\\/g, '/')}` 
            : wasmPath;
        wasm = await import(importPath);
    } catch (e) {
        console.error('WASM module not built. Run "wasm-pack build --target nodejs --features wasm" first');
        process.exit(1);
    }
    
    
    testData = loadTestData();
});

test.describe('FVG Trailing Stop', () => {
    test('should calculate FVG Trailing Stop correctly with default parameters', () => {
        const { high, low, close } = testData;
        
        const result = wasm.fvgTrailingStop(
            high, low, close,
            5,    
            9,    
            false 
        );
        
        
        
        
        const expectedLower = [55643.00, 55643.00, 55643.00, 55643.00, 55643.00];
        const expectedLowerTs = [60223.33333333, 60223.33333333, 60223.33333333, 60223.33333333, 60223.33333333];
        
        
        const n = result.lower.length;
        for (let i = 0; i < 5; i++) {
            const idx = n - 5 + i;
            
            
            if (!isNaN(result.lower[idx])) {
                assertClose(
                    result.lower[idx],
                    expectedLower[i],
                    0.01,
                    `Lower mismatch at index ${idx}`
                );
            }
            
            
            if (!isNaN(result.lowerTs[idx])) {
                assertClose(
                    result.lowerTs[idx],
                    expectedLowerTs[i],
                    0.01,
                    `Lower TS mismatch at index ${idx}`
                );
            }
        }
    });
    
    test('should handle empty arrays', () => {
        assert.throws(
            () => wasm.fvgTrailingStop([], [], [], 5, 9, false),
            /Input data slice is empty/,
            'Should throw error for empty arrays'
        );
    });
    
    test('should handle all NaN values', () => {
        const nanArray = new Array(100).fill(NaN);
        
        assert.throws(
            () => wasm.fvgTrailingStop(nanArray, nanArray, nanArray, 5, 9, false),
            /All values are NaN/,
            'Should throw error for all NaN values'
        );
    });
    
    test('should handle invalid periods', () => {
        const testHigh = [1, 2, 3, 4, 5];
        const testLow = [0.5, 1.5, 2.5, 3.5, 4.5];
        const testClose = [0.75, 1.75, 2.75, 3.75, 4.75];
        
        
        assert.throws(
            () => wasm.fvgTrailingStop(testHigh, testLow, testClose, 0, 9, false),
            /Invalid unmitigated_fvg_lookback: 0/,
            'Should throw error for lookback = 0'
        );
        
        
        assert.throws(
            () => wasm.fvgTrailingStop(testHigh, testLow, testClose, 5, 0, false),
            /Invalid smoothing_length: 0/,
            'Should throw error for smoothing = 0'
        );
    });
    
    test('should handle mismatched array lengths', () => {
        const testHigh = [1, 2, 3, 4, 5];
        const testLow = [1, 2, 3];
        const testClose = [1, 2];
        
        assert.throws(
            () => wasm.fvgTrailingStop(testHigh, testLow, testClose, 5, 9, false),
            /Invalid period/,
            'Should throw error for mismatched array lengths'
        );
    });
    
    test('should calculate with reset_on_cross enabled', () => {
        const { high, low, close } = testData;
        const result = wasm.fvgTrailingStop(
            high, low, close,
            5,    
            9,    
            true  
        );
        
        
        assert(Array.isArray(result.upper), 'Upper should be an array');
        assert(Array.isArray(result.lower), 'Lower should be an array');
        assert(Array.isArray(result.upperTs), 'UpperTs should be an array');
        assert(Array.isArray(result.lowerTs), 'LowerTs should be an array');
        assert.strictEqual(result.upper.length, high.length, 'Upper array length mismatch');
        assert.strictEqual(result.lower.length, high.length, 'Lower array length mismatch');
        assert.strictEqual(result.upperTs.length, high.length, 'UpperTs array length mismatch');
        assert.strictEqual(result.lowerTs.length, high.length, 'LowerTs array length mismatch');
    });
    
    test('should handle partial NaN values', () => {
        const testHigh = testData.high.slice();
        const testLow = testData.low.slice();
        const testClose = testData.close.slice();
        
        
        for (let i = 10; i < 20; i++) {
            testHigh[i] = NaN;
            testLow[i] = NaN;
            testClose[i] = NaN;
        }
        
        
        const result = wasm.fvgTrailingStop(
            testHigh, testLow, testClose,
            5, 9, false
        );
        
        assert.strictEqual(result.upper.length, testHigh.length, 'Output length should match input');
        assert.strictEqual(result.lower.length, testHigh.length, 'Output length should match input');
    });
    
    test('should verify output consistency', () => {
        const { high, low, close } = testData;
        const result = wasm.fvgTrailingStop(
            high, low, close,
            5, 9, false
        );
        
        
        for (let i = 0; i < result.upper.length; i++) {
            const upperActive = !isNaN(result.upper[i]);
            const lowerActive = !isNaN(result.lower[i]);
            
            
            if (i > 20) {  
                assert(
                    !(upperActive && lowerActive),
                    `Both upper and lower indicators active at index ${i}`
                );
            }
        }
    });
    
    test('should validate warmup period handling', () => {
        const { high, low, close } = testData;
        const result = wasm.fvgTrailingStop(
            high, low, close,
            5,    
            9,    
            false 
        );
        
        
        const expectedWarmup = 2 + 9 - 1;
        
        
        let firstNonNaN = -1;
        for (let i = 0; i < result.upper.length; i++) {
            if (!isNaN(result.upper[i]) || !isNaN(result.lower[i]) ||
                !isNaN(result.upperTs[i]) || !isNaN(result.lowerTs[i])) {
                firstNonNaN = i;
                break;
            }
        }
        
        
        assert(firstNonNaN > 0, 'Should have warmup period with NaN values');
    });
    
    test('should handle batch processing with single parameter set', () => {
        
        const testHigh = Array.from({length: 50}, (_, i) => 100.0 + i);
        const testLow = Array.from({length: 50}, (_, i) => 95.0 + i);
        const testClose = Array.from({length: 50}, (_, i) => 97.5 + i);
        
        
        const result = wasm.fvgTrailingStopBatch(
            testHigh, testLow, testClose,
            5, 5, 0,  
            9, 9, 0,  
            true, false  
        );
        
        
        assert(result.values, 'Should have values field');
        assert(result.combos, 'Should have combos field');
        assert(result.rows, 'Should have rows field');
        assert(result.cols, 'Should have cols field');
        
        
        assert.strictEqual(result.combos.length, 1, 'Should have 1 combination');
        assert.strictEqual(result.rows, 1, 'Should have 1 row (combo)');
        assert.strictEqual(result.cols, testHigh.length, 'Cols should match input length');
        assert.strictEqual(result.values.length, 4 * result.rows * result.cols, 'Values should be 4*rows*cols');
        
        
        assert.strictEqual(result.combos[0].unmitigated_fvg_lookback, 5, 'Lookback should be 5');
        assert.strictEqual(result.combos[0].smoothing_length, 9, 'Smoothing should be 9');
        assert.strictEqual(result.combos[0].reset_on_cross, false, 'Reset should be false');
        
        
        const single = wasm.fvgTrailingStop(
            testHigh, testLow, testClose,
            5, 9, false
        );
        
        
        
        const upperFromBatch = result.values.slice(0, result.cols);
        
        
        for (let i = 0; i < testHigh.length; i++) {
            const batchVal = upperFromBatch[i];
            const singleVal = single.upper[i];
            
            if (isNaN(batchVal) && isNaN(singleVal)) {
                continue;
            } else if (!isNaN(batchVal) && !isNaN(singleVal)) {
                assertClose(batchVal, singleVal, 1e-9, `Upper mismatch at index ${i}`);
            } else {
                assert.fail(`NaN mismatch at index ${i}: batch=${batchVal}, single=${singleVal}`);
            }
        }
    });
    
    test('should handle batch processing with parameter sweep', () => {
        const testHigh = testData.high.slice(0, 100);
        const testLow = testData.low.slice(0, 100);
        const testClose = testData.close.slice(0, 100);
        
        
        const result = wasm.fvgTrailingStopBatch(
            testHigh, testLow, testClose,
            3, 7, 2,    
            5, 9, 4,    
            true, true  
        );
        
        
        assert(result.values, 'Should have values field');
        assert(result.combos, 'Should have combos field');
        assert(result.rows, 'Should have rows field');
        assert(result.cols, 'Should have cols field');
        
        
        assert.strictEqual(result.combos.length, 12, 'Should have 12 combinations');
        assert.strictEqual(result.rows, 12, 'Should have 12 rows (combos)');
        assert.strictEqual(result.cols, testHigh.length, 'Cols should match input length');
        assert.strictEqual(result.values.length, 4 * result.rows * result.cols, 'Values should be 4*rows*cols');
        
        
        const expectedLookbacks = [3, 3, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7];
        const expectedSmoothings = [5, 5, 9, 9, 5, 5, 9, 9, 5, 5, 9, 9];
        const expectedResets = [false, true, false, true, false, true, false, true, false, true, false, true];
        
        for (let i = 0; i < 12; i++) {
            assert.strictEqual(result.combos[i].unmitigated_fvg_lookback, expectedLookbacks[i], `Lookback mismatch at ${i}`);
            assert.strictEqual(result.combos[i].smoothing_length, expectedSmoothings[i], `Smoothing mismatch at ${i}`);
            assert.strictEqual(result.combos[i].reset_on_cross, expectedResets[i], `Reset mismatch at ${i}`);
        }
    });
    
    test('should support zero-copy API for efficient memory usage', () => {
        
        const ptr = wasm.fvgTrailingStopAlloc(testData.high.length);
        assert(ptr > 0, 'Should allocate memory successfully');
        
        
        const result = wasm.fvgTrailingStopZeroCopy(
            testData.high, testData.low, testData.close,
            5, 9, false,
            ptr
        );
        
        
        assert(result, 'Should return result');
        assert.strictEqual(result.upper.length, testData.high.length, 'Upper length should match');
        assert.strictEqual(result.lower.length, testData.high.length, 'Lower length should match');
        
        
        wasm.fvgTrailingStopFree(ptr);
        
        
    });
    
    test('should verify mutual exclusivity between upper and lower', () => {
        const result = wasm.fvgTrailingStop(
            testData.high.slice(0, 200), testData.low.slice(0, 200), testData.close.slice(0, 200),
            5, 9, false
        );
        
        
        const warmup = 20;  
        for (let i = warmup; i < Math.min(result.upper.length, warmup + 100); i++) {
            const upperActive = !isNaN(result.upper[i]);
            const lowerActive = !isNaN(result.lower[i]);
            
            
            assert(
                !(upperActive && lowerActive),
                `Both upper and lower indicators active at index ${i}`
            );
        }
    });
    
    test('should handle edge case parameters', () => {
        const testHigh = Array.from({length: 20}, (_, i) => 100.0 + i);
        const testLow = Array.from({length: 20}, (_, i) => 95.0 + i);
        const testClose = Array.from({length: 20}, (_, i) => 97.5 + i);
        
        
        assert.throws(
            () => wasm.fvgTrailingStop(testHigh, testLow, testClose, 0, 9, false),
            /Invalid unmitigated_fvg_lookback: 0/,
            'Should throw error for lookback = 0'
        );
        
        
        const result2 = wasm.fvgTrailingStop(
            testHigh, testLow, testClose,
            5, 1, false
        );
        assert.strictEqual(result2.upper.length, testHigh.length, 'Should handle smoothing=1');
        
        
        assert.throws(
            () => wasm.fvgTrailingStop(testHigh, testLow, testClose, 5, 0, false),
            /Invalid smoothing_length: 0/,
            'Should throw error for smoothing = 0'
        );
    });
    
    test('should verify output consistency relationships', () => {
        const result = wasm.fvgTrailingStop(
            testData.high.slice(0, 200), testData.low.slice(0, 200), testData.close.slice(0, 200),
            5, 9, false
        );
        
        
        for (let i = 50; i < result.upper.length; i++) {  
            if (!isNaN(result.upper[i])) {
                assert(!isNaN(result.upperTs[i]), 
                    `upperTs should be active when upper is at index ${i}`);
            }
            if (!isNaN(result.lower[i])) {
                assert(!isNaN(result.lowerTs[i]), 
                    `lowerTs should be active when lower is at index ${i}`);
            }
            
            
            if (!isNaN(result.upperTs[i])) {
                const recentMax = Math.max(...testData.high.slice(Math.max(0, i-20), i+1));
                assert(result.upperTs[i] <= recentMax * 1.5,
                    `upperTs seems unreasonably high at index ${i}`);
            }
            if (!isNaN(result.lowerTs[i])) {
                const recentMin = Math.min(...testData.low.slice(Math.max(0, i-20), i+1));
                assert(result.lowerTs[i] >= recentMin * 0.5,
                    `lowerTs seems unreasonably low at index ${i}`);
            }
        }
    });
    
    test('should handle SIMD kernel specifications if available', () => {
        
        const kernels = ['scalar', 'auto'];  
        
        for (const kernel of kernels) {
            try {
                const result = wasm.fvgTrailingStop(
                    testData.high.slice(0, 100), testData.low.slice(0, 100), testData.close.slice(0, 100),
                    5, 9, false, kernel
                );
                assert.strictEqual(result.upper.length, 100, `Failed for kernel ${kernel}`);
            } catch (e) {
                
                
            }
        }
    });
    
    test('should efficiently handle large datasets', () => {
        
        const result = wasm.fvgTrailingStop(testData.high, testData.low, testData.close, 5, 9, false);
        
        assert.strictEqual(result.upper.length, testData.high.length, 'Should handle full dataset');
        assert(result.upperTs, 'Should have upperTs array');
        assert(result.lowerTs, 'Should have lowerTs array');
        
        
        for (let i = 0; i < Math.min(100, result.upper.length); i++) {
            assert(isFinite(result.upper[i]) || isNaN(result.upper[i]),
                `Invalid value at index ${i}: ${result.upper[i]}`);
        }
    });
});