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
    
    test('reinput - using output as input', async () => {
        const close = new Float64Array(testData.close);
        
        // First SWMA with period 5
        const firstPeriod = 5;
        const firstResult = wasm.swma_js(close, firstPeriod);
        assert.strictEqual(firstResult.length, close.length);
        
        // Use first result as input for second SWMA with period 3
        const secondPeriod = 3;
        const secondResult = wasm.swma_js(firstResult, secondPeriod);
        assert.strictEqual(secondResult.length, firstResult.length);
        
        // Verify second result is not all NaN
        // First SWMA has warmup of 4, second SWMA adds warmup of 2
        // So total warmup should be 4 + 2 = 6
        for (let i = 0; i < 6; i++) {
            assert(isNaN(secondResult[i]), `Index ${i} should be NaN`);
        }
        for (let i = 6; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Index ${i} should not be NaN`);
        }
        
        // Verify the output values are reasonable
        const validValues = secondResult.slice(6);
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
});