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
    
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
        
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = await loadTestData();
});

test.describe('SWMA (Symmetric Weighted Moving Average)', () => {
    test('SWMA empty input', () => {
        
        const empty = new Float64Array([]);
        
        assert.throws(
            () => wasm.swma_js(empty, 5),
            /Input data slice is empty/,
            'Should throw error for empty input'
        );
    });
    
    test('SWMA accuracy', async () => {
        
        const close = new Float64Array(testData.close);
        const expected = EXPECTED_OUTPUTS.swma;
        
        const result = wasm.swma_js(
            close,
            expected.defaultParams.period
        );
        
        assert.strictEqual(result.length, close.length);
        
        
        const last5 = result.slice(-5);
        assertArrayClose(
            last5,
            expected.last5Values,
            1e-8,
            "SWMA last 5 values mismatch"
        );
        
        
        await compareWithRust('swma', result, 'close', expected.defaultParams);
    });
    
    test('basic functionality', () => {
        const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        const period = 5;
        
        const result = wasm.swma_js(data, period);
        
        assert.strictEqual(result.length, data.length, 'Result length should match input length');
        
        
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(result[i]), `Index ${i} should be NaN during warmup`);
        }
        
        
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
        
        
        assert.throws(
            () => wasm.swma_js(data, 6),
            /Invalid period/,
            'Should throw error when period exceeds data length'
        );
        
        
        assert.throws(
            () => wasm.swma_js(data, 0),
            /Invalid period/,
            'Should throw error for zero period'
        );
    });
    
    test('error handling - not enough valid data', () => {
        
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
        
        
        const expectedPeriods = [];
        for (let p = minPeriod; p <= maxPeriod; p += stepPeriod) {
            expectedPeriods.push(p);
        }
        
        const rows = expectedPeriods.length;
        const cols = data.length;
        
        assert.strictEqual(metadata.length, rows, 'Metadata length should match number of periods');
        assert.strictEqual(values.length, rows * cols, 'Values array size should be rows*cols');
        
        
        for (let i = 0; i < expectedPeriods.length; i++) {
            const period = expectedPeriods[i];
            const individual = wasm.swma_js(data, period);
            
            
            const row = new Float64Array(cols);
            for (let j = 0; j < cols; j++) {
                row[j] = values[i * cols + j];
            }
            
            assertArrayClose(row, individual, 1e-10, `Batch row ${i} should match individual calculation`);
        }
    });
    
    test('symmetric triangular weights', () => {
        
        const data = new Float64Array(30).fill(0);
        data[15] = 1.0; 
        const period = 5;
        
        const result = wasm.swma_js(data, period);
        
        
        
        
        
        
        
        
        
        assertClose(result[15], 1/9, 1e-10, 'Weight at position 4');
        assertClose(result[16], 2/9, 1e-10, 'Weight at position 3');
        assertClose(result[17], 3/9, 1e-10, 'Weight at position 2 (center)');
        assertClose(result[18], 2/9, 1e-10, 'Weight at position 1');
        assertClose(result[19], 1/9, 1e-10, 'Weight at position 0');
        
        
        assert.strictEqual(result[10], 0.0, 'Should be zero before affected range');
        assert.strictEqual(result[20], 0.0, 'Should be zero after affected range');
    });
    
    test('edge cases', () => {
        
        const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        const result = wasm.swma_js(data, 10);
        
        
        for (let i = 0; i < 9; i++) {
            assert(isNaN(result[i]), `Index ${i} should be NaN`);
        }
        assert(!isNaN(result[9]), 'Last value should not be NaN');
        
        
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
        
        
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(result[i]), `Index ${i} should be NaN`);
        }
        assert(!isNaN(result[period - 1]), 'First non-NaN value should be at period-1');
    });
    
    
    test('SWMA zero-copy API', () => {
        const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        const period = 5;
        
        
        const ptr = wasm.swma_alloc(data.length);
        assert(ptr !== 0, 'Failed to allocate memory');
        
        
        const memView = new Float64Array(
            wasm.__wasm.memory.buffer,
            ptr,
            data.length
        );
        
        
        memView.set(data);
        
        
        try {
            wasm.swma_into(ptr, ptr, data.length, period);
            
            
            const regularResult = wasm.swma_js(data, period);
            for (let i = 0; i < data.length; i++) {
                if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                    continue; 
                }
                assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                       `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
            }
        } finally {
            
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
            
            
            const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
            
            
            for (let i = 0; i < 9; i++) {
                assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
            }
            
            
            for (let i = 9; i < Math.min(100, size); i++) {
                assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
            }
        } finally {
            wasm.swma_free(ptr, size);
        }
    });
    
    test('SWMA zero-copy error handling', () => {
        
        assert.throws(() => {
            wasm.swma_into(0, 0, 10, 5);
        }, /null pointer|Null pointer provided/i);
        
        
        const ptr = wasm.swma_alloc(10);
        try {
            
            assert.throws(() => {
                wasm.swma_into(ptr, ptr, 10, 0);
            }, /Invalid period/);
            
            
            assert.throws(() => {
                wasm.swma_into(ptr, ptr, 10, 11);
            }, /Invalid period/);
        } finally {
            wasm.swma_free(ptr, 10);
        }
    });
    
    test('SWMA zero-copy memory management', () => {
        
        const sizes = [100, 1000, 10000, 100000];
        
        for (const size of sizes) {
            const ptr = wasm.swma_alloc(size);
            assert(ptr !== 0, `Failed to allocate ${size} elements`);
            
            
            const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
            for (let i = 0; i < Math.min(10, size); i++) {
                memView[i] = i * 1.5;
            }
            
            
            for (let i = 0; i < Math.min(10, size); i++) {
                assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
            }
            
            
            wasm.swma_free(ptr, size);
        }
    });
    
    
    test('SWMA batch - unified API with single parameter', () => {
        const close = new Float64Array(testData.close.slice(0, 100));
        
        const result = wasm.swma_batch(close, {
            period_range: [5, 5, 0]
        });
        
        
        assert(result.values, 'Should have values array');
        assert(result.combos, 'Should have combos array');
        assert(typeof result.rows === 'number', 'Should have rows count');
        assert(typeof result.cols === 'number', 'Should have cols count');
        
        
        assert.strictEqual(result.rows, 1);
        assert.strictEqual(result.cols, close.length);
        assert.strictEqual(result.combos.length, 1);
        assert.strictEqual(result.values.length, close.length);
        
        
        const combo = result.combos[0];
        assert.strictEqual(combo.period, 5);
        
        
        const oldResult = wasm.swma_js(close, 5);
        for (let i = 0; i < oldResult.length; i++) {
            if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
                continue; 
            }
            assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
                   `Value mismatch at index ${i}`);
        }
    });
    
    test('SWMA batch - unified API with multiple parameters', () => {
        const close = new Float64Array(testData.close.slice(0, 50));
        
        const result = wasm.swma_batch(close, {
            period_range: [5, 9, 2]  
        });
        
        
        assert.strictEqual(result.rows, 3);
        assert.strictEqual(result.cols, 50);
        assert.strictEqual(result.combos.length, 3);
        assert.strictEqual(result.values.length, 150);
        
        
        const expectedPeriods = [5, 7, 9];
        for (let i = 0; i < expectedPeriods.length; i++) {
            assert.strictEqual(result.combos[i].period, expectedPeriods[i]);
        }
        
        
        const firstRow = result.values.slice(0, result.cols);
        const oldResult = wasm.swma_js(close, 5);
        for (let i = 0; i < oldResult.length; i++) {
            if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
                continue; 
            }
            assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
                   `Value mismatch at index ${i}`);
        }
    });
    
    test('SWMA batch - unified API error handling', () => {
        const close = new Float64Array(testData.close.slice(0, 10));
        
        
        assert.throws(() => {
            wasm.swma_batch(close, {
                period_range: [5, 5]  
            });
        }, /Invalid config/);
        
        
        assert.throws(() => {
            wasm.swma_batch(close, {});
        }, /Invalid config/);
        
        
        assert.throws(() => {
            wasm.swma_batch(close, {
                period_range: "invalid"
            });
        }, /Invalid config/);
    });
    
    test('SWMA batch edge cases', () => {
        const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        
        
        const singleBatch = wasm.swma_batch(close, {
            period_range: [5, 5, 1]
        });
        
        assert.strictEqual(singleBatch.values.length, 10);
        assert.strictEqual(singleBatch.combos.length, 1);
        
        
        const largeBatch = wasm.swma_batch(close, {
            period_range: [5, 7, 10]  
        });
        
        
        assert.strictEqual(largeBatch.values.length, 10);
        assert.strictEqual(largeBatch.combos.length, 1);
        
        
        assert.throws(() => {
            wasm.swma_batch(new Float64Array([]), {
                period_range: [5, 5, 0]
            });
        }, /Input data slice is empty/);
    });
});