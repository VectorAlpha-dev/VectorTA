/**
 * WASM binding tests for Ehlers KAMA indicator.
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
    
    testData = loadTestData();
});

test('Ehlers KAMA partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.ehlers_kama_js(close, 20);
    assert.strictEqual(result.length, close.length);
});

test('Ehlers KAMA accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.ehlersKama;
    
    const result = wasm.ehlers_kama_js(
        close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last6 = result.slice(-6);
    const checkValues = last6.slice(0, 5);
    assertArrayClose(
        checkValues,
        expected.last5Values,
        1e-8,
        "Ehlers KAMA last 5 values mismatch"
    );
    
    
    
});

test('Ehlers KAMA default', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.ehlers_kama_js(close, 20);
    assert.strictEqual(result.length, close.length);
});

test('Ehlers KAMA zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]);
    
    assert.throws(() => {
        wasm.ehlers_kama_js(inputData, 0);
    }, /Invalid period/);
});

test('Ehlers KAMA period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    assert.throws(() => {
        wasm.ehlers_kama_js(dataSmall, 10);
    }, /Invalid period/);
});

test('Ehlers KAMA very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.ehlers_kama_js(singlePoint, 20);
    }, /Invalid period|Not enough valid data/);
});

test('Ehlers KAMA empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.ehlers_kama_js(empty, 20);
    }, /Input data slice is empty/);
});

test('Ehlers KAMA all NaN', () => {
    
    const data = new Float64Array(10).fill(NaN);
    
    assert.throws(() => {
        wasm.ehlers_kama_js(data, 5);
    }, /All input data is NaN|All values are NaN/);
});

test('Ehlers KAMA not enough valid data', () => {
    
    const data = new Float64Array([NaN, NaN, NaN, NaN, 1.0, 2.0, 3.0]);
    
    assert.throws(() => {
        wasm.ehlers_kama_js(data, 5);
    }, /Not enough valid data|Invalid period/);
});

test('Ehlers KAMA with NaN prefix', () => {
    
    const dataWithNaN = new Float64Array([NaN, NaN, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    
    const result = wasm.ehlers_kama_js(dataWithNaN, 3);
    assert.strictEqual(result.length, dataWithNaN.length);
    
    
    assert(isNaN(result[0]));
    assert(isNaN(result[1]));
    assert(isNaN(result[2]));
    assert(isNaN(result[3]));
    
    
    assert(!isNaN(result[4]));
});

test('Ehlers KAMA memory management', () => {
    
    const len = 100;
    
    
    const ptr = wasm.ehlers_kama_alloc(len);
    assert(ptr !== 0, "Failed to allocate memory");
    
    
    assert.doesNotThrow(() => {
        wasm.ehlers_kama_free(ptr, len);
    });
});

test('Ehlers KAMA into with null pointers', () => {
    
    assert.throws(() => {
        wasm.ehlers_kama_into(0, 0, 10, 5);
    }, /null pointer/);
});

test('Ehlers KAMA batch with NaN handling', () => {
    
    const data = new Float64Array([NaN, NaN].concat(Array.from({length: 48}, (_, i) => i + 1)));
    
    const result = wasm.ehlers_kama_batch(data, {
        period_range: [10, 20, 10]  
    });
    
    
    for (let rowIdx = 0; rowIdx < 2; rowIdx++) {
        const rowStart = rowIdx * 50;
        const row = result.values.slice(rowStart, rowStart + 50);
        
        assert(isNaN(row[0]));
        assert(isNaN(row[1]));
    }
});

test('Ehlers KAMA NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.ehlers_kama_js(close, 20);
    assert.strictEqual(result.length, close.length);
    
    
    assertAllNaN(result.slice(0, 19), "Expected NaN in warmup period");
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('Ehlers KAMA warmup period validation', () => {
    
    const close = new Float64Array(testData.close);
    const period = 20;
    
    const result = wasm.ehlers_kama_js(close, period);
    
    
    let nanCount = 0;
    for (const val of result) {
        if (isNaN(val)) {
            nanCount++;
        } else {
            break;
        }
    }
    
    
    assert.strictEqual(nanCount, period - 1, `Expected ${period-1} NaN values for warmup, got ${nanCount}`);
    
    
    assert(!isNaN(result[period-1]), `Expected valid value at index ${period-1}`);
});

test('Ehlers KAMA batch single parameter', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.ehlers_kama_batch(close, {
        period_range: [20, 20, 0]  
    });
    
    
    const singleResult = wasm.ehlers_kama_js(close, 20);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    
    assertArrayClose(batchResult.values, singleResult, 1e-9, "Batch vs single mismatch");
});

test('Ehlers KAMA batch multiple periods', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const batchResult = wasm.ehlers_kama_batch(close, {
        period_range: [10, 20, 5]  
    });
    
    
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    
    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.ehlers_kama_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('Ehlers KAMA batch metadata from result', () => {
    
    const close = new Float64Array(50); 
    close.fill(100);
    
    const result = wasm.ehlers_kama_batch(close, {
        period_range: [10, 30, 10]  
    });
    
    
    assert.strictEqual(result.combos.length, 3);
    
    
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[1].period, 20);
    assert.strictEqual(result.combos[2].period, 30);
});

test('Ehlers KAMA batch into function', () => {
    
    const data = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
                                  110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0]);
    const len = data.length;
    const periods = [5, 10];
    const numRows = periods.length;
    
    
    const inPtr = wasm.ehlers_kama_alloc(len);
    const outPtr = wasm.ehlers_kama_alloc(len * numRows);
    
    try {
        
        const memory = wasm.__wasm ? wasm.__wasm.memory : wasm.memory;
        const inputArray = new Float64Array(memory.buffer, inPtr, len);
        const outputArray = new Float64Array(memory.buffer, outPtr, len * numRows);
        
        
        inputArray.set(data);
        
        
        const resultRows = wasm.ehlers_kama_batch_into(inPtr, outPtr, len, 5, 10, 5);
        assert.strictEqual(resultRows, numRows, "Batch should return correct number of rows");
        
        
        for (let i = 0; i < numRows; i++) {
            const rowStart = i * len;
            const rowEnd = rowStart + len;
            const rowData = Array.from(outputArray.slice(rowStart, rowEnd));
            
            const expected = wasm.ehlers_kama_js(data, periods[i]);
            for (let j = 0; j < len; j++) {
                if (isNaN(expected[j])) {
                    assert(isNaN(rowData[j]), `Row ${i} mismatch at index ${j}: expected NaN, got ${rowData[j]}`);
                } else {
                    assertClose(rowData[j], expected[j], 1e-10, `Row ${i} mismatch at index ${j}`);
                }
            }
        }
    } finally {
        
        wasm.ehlers_kama_free(inPtr, len);
        wasm.ehlers_kama_free(outPtr, len * numRows);
    }
});

test('Ehlers KAMA into function', () => {
    
    const data = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]);
    const len = data.length;
    
    
    const inPtr = wasm.ehlers_kama_alloc(len);
    const outPtr = wasm.ehlers_kama_alloc(len);
    
    try {
        
        const memory = wasm.__wasm ? wasm.__wasm.memory : wasm.memory;
        const inputArray = new Float64Array(memory.buffer, inPtr, len);
        const outputArray = new Float64Array(memory.buffer, outPtr, len);
        
        
        inputArray.set(data);
        
        
        wasm.ehlers_kama_into(inPtr, outPtr, len, 3);
        
        
        assert.strictEqual(outputArray.length, data.length);
        
        
        const expected = wasm.ehlers_kama_js(data, 3);
        for (let i = 0; i < len; i++) {
            if (isNaN(expected[i])) {
                assert(isNaN(outputArray[i]), `Mismatch at index ${i}: expected NaN, got ${outputArray[i]}`);
            } else {
                assertClose(outputArray[i], expected[i], 1e-10, `Mismatch at index ${i}`);
            }
        }
    } finally {
        
        wasm.ehlers_kama_free(inPtr, len);
        wasm.ehlers_kama_free(outPtr, len);
    }
});
