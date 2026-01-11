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

test.describe('TEMA (Triple Exponential Moving Average)', () => {
    test('basic functionality', () => {
        const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        const period = 3;
        
        const result = wasm.tema_js(data, period);
        
        assert.strictEqual(result.length, data.length, 'Result length should match input length');
        
        
        for (let i = 0; i < 6; i++) {
            assert(isNaN(result[i]), `Index ${i} should be NaN during warmup`);
        }
        
        
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
        
        
        assert.throws(
            () => wasm.tema_js(data, 6),
            /Invalid period/,
            'Should throw error when period exceeds data length'
        );
        
        
        assert.throws(
            () => wasm.tema_js(data, 0),
            /Invalid period/,
            'Should throw error for zero period'
        );
    });
    
    test('error handling - not enough valid data', () => {
        
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
        
        await compareWithRust('tema', result, 'close', { period }, 1e-9);
    });
    
    test('batch calculation', () => {
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = Math.random() * 100;
        }
        
        const minPeriod = 5;
        const maxPeriod = 15;
        const stepPeriod = 2;
        
        
        const values = wasm.tema_batch_js(data, minPeriod, maxPeriod, stepPeriod);
        
        
        const metadata = [];
        for (let p = minPeriod; p <= maxPeriod; p += stepPeriod) {
            metadata.push(p);
        }
        
        
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
            const individual = wasm.tema_js(data, period);
            
            
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
        
        
        const periods = [3, 5, 7, 9];
        for (const period of periods) {
            const result = wasm.tema_js(data, period);
            const warmup = (period - 1) * 3;
            
            
            for (let i = 0; i < warmup; i++) {
                assert(isNaN(result[i]), `Period ${period}: Index ${i} should be NaN`);
            }
            
            
            for (let i = warmup; i < result.length; i++) {
                assert(!isNaN(result[i]), `Period ${period}: Index ${i} should not be NaN`);
            }
        }
    });
    
    test('edge cases', () => {
        
        const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        const result = wasm.tema_js(data, 10);
        
        
        
        assertAllNaN(result, 'All values should be NaN when warmup exceeds data length');
        
        
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
        
        
        const warmup = (period - 1) * 3;
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(result[i]), `Index ${i} should be NaN`);
        }
        assert(!isNaN(result[warmup]), 'First non-NaN value should be at warmup index');
    });
    
    test('all NaN input', () => {
        const allNan = new Float64Array(100).fill(NaN);
        
        assert.throws(
            () => wasm.tema_js(allNan, 9),
            /All values are NaN/,
            'Should throw error for all NaN input'
        );
    });
    
    test('accuracy check with expected values', () => {
        const close = new Float64Array(testData.close);
        const expected = EXPECTED_OUTPUTS.tema;
        
        const result = wasm.tema_js(close, expected.defaultParams.period);
        
        
        const last5 = result.slice(-5);
        assertArrayClose(
            last5,
            expected.last5Values,
            1e-9,
            "TEMA last 5 values mismatch"
        );
        
        
        const warmup = expected.warmupPeriod;
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
        }
        for (let i = warmup; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} after warmup`);
        }
    });
    
    test('batch metadata structure', () => {
        
        const result = wasm.tema_batch(new Float64Array(20), {
            period_range: [5, 15, 2]
        });
        
        
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
        
        for (const period of [1, 2, 3, 4, 5]) {
            const data = new Float64Array(period);
            for (let i = 0; i < period; i++) {
                data[i] = i + 1;
            }
            
            if (period === 1) {
                
                const result = wasm.tema_js(data, period);
                assertArrayClose(result, data, 1e-10, 'Period 1 should return input');
            } else {
                
                const result = wasm.tema_js(data, period);
                assert.strictEqual(result.length, data.length);
                
                const warmup = (period - 1) * 3;
                if (warmup >= data.length) {
                    assertAllNaN(result, 'Should be all NaN when warmup exceeds data length');
                }
            }
        }
    });

    
    test('TEMA zero-copy API', () => {
        const data = new Float64Array(100);
        for (let i = 0; i < 100; i++) {
            data[i] = Math.sin(i * 0.1) * 10 + 50;
        }
        const period = 14;

        
        const ptr = wasm.tema_alloc(data.length);
        assert(ptr !== 0, 'Failed to allocate memory');

        try {
            
            const memView = new Float64Array(
                wasm.__wasm.memory.buffer,
                ptr,
                data.length
            );

            
            memView.set(data);

            
            wasm.tema_into(ptr, ptr, data.length, period);

            
            const regularResult = wasm.tema_js(data, period);
            for (let i = 0; i < data.length; i++) {
                if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                    continue; 
                }
                assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                       `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
            }
        } finally {
            
            wasm.tema_free(ptr, data.length);
        }
    });

    test('TEMA ergonomic batch API', () => {
        const close = new Float64Array(testData.close.slice(0, 50));
        
        const result = wasm.tema_batch(close, {
            period_range: [5, 15, 5]  
        });

        
        assert(result.values, 'Should have values array');
        assert(result.combos, 'Should have combos array');
        assert(typeof result.rows === 'number', 'Should have rows count');
        assert(typeof result.cols === 'number', 'Should have cols count');

        
        assert.strictEqual(result.rows, 3);
        assert.strictEqual(result.cols, 50);
        assert.strictEqual(result.combos.length, 3);
        assert.strictEqual(result.values.length, 150);

        
        assert.strictEqual(result.combos[0].period, 5);
        assert.strictEqual(result.combos[1].period, 10);
        assert.strictEqual(result.combos[2].period, 15);

        
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

        const periods = { start: 5, end: 15, step: 5 }; 
        const numCombos = 3;
        const totalSize = numCombos * data.length;

        
        const inPtr = wasm.tema_alloc(data.length);
        const outPtr = wasm.tema_alloc(totalSize);

        try {
            
            const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
            inView.set(data);

            
            const rows = wasm.tema_batch_into(
                inPtr, outPtr, data.length,
                periods.start, periods.end, periods.step
            );

            assert.strictEqual(rows, numCombos, 'Should return correct number of rows');

            
            const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
            
            
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

        
        const ptr = wasm.tema_alloc(10);
        try {
            
            assert.throws(() => {
                wasm.tema_into(ptr, ptr, 10, 0);
            }, /Invalid period/);
        } finally {
            wasm.tema_free(ptr, 10);
        }
    });

    test('TEMA batch memory leak prevention', () => {
        
        const sizes = [100, 1000, 10000];
        
        for (const size of sizes) {
            const ptr = wasm.tema_alloc(size);
            assert(ptr !== 0, `Failed to allocate ${size} elements`);
            
            
            const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
            for (let i = 0; i < Math.min(10, size); i++) {
                memView[i] = i * 1.5;
            }
            
            
            for (let i = 0; i < Math.min(10, size); i++) {
                assert.strictEqual(memView[i], i * 1.5, `Memory at ${i} should match pattern`);
            }
            
            
            wasm.tema_free(ptr, size);
        }
    });
});

test.after(() => {
    console.log('TEMA WASM tests completed');
});
