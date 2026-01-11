/**
 * WASM binding tests for ADXR indicator.
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

test('ADXR partial params', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.adxr_js(high, low, close, 14);
    assert.strictEqual(result.length, close.length);
});

test('ADXR accuracy', async () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.adxr;
    
    const result = wasm.adxr_js(
        high, low, close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-1,  
        "ADXR last 5 values mismatch"
    );
    
    
    await compareWithRust('adxr', result, 'hlc', expected.defaultParams);
});

test('ADXR default candles', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.adxr_js(high, low, close, 14);
    assert.strictEqual(result.length, close.length);
});

test('ADXR zero period', () => {
    
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([9.0, 19.0, 29.0]);
    const close = new Float64Array([9.5, 19.5, 29.5]);
    
    assert.throws(() => {
        wasm.adxr_js(high, low, close, 0);
    }, /Invalid period/);
});

test('ADXR period exceeds length', () => {
    
    const high = new Float64Array([10.0, 20.0]);
    const low = new Float64Array([9.0, 19.0]);
    const close = new Float64Array([9.5, 19.5]);
    
    assert.throws(() => {
        wasm.adxr_js(high, low, close, 10);
    }, /Invalid period/);
});

test('ADXR very small dataset', () => {
    
    const high = new Float64Array([100.0]);
    const low = new Float64Array([99.0]);
    const close = new Float64Array([99.5]);
    
    assert.throws(() => {
        wasm.adxr_js(high, low, close, 14);
    }, /Invalid period|Not enough data/);
});

test('ADXR empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.adxr_js(empty, empty, empty, 14);
    }, /Empty input data|Invalid period|Not enough data|All values are NaN/);
});

test('ADXR mismatched lengths', () => {
    
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([9.0, 19.0]);  
    const close = new Float64Array([9.5, 19.5, 29.5]);
    
    assert.throws(() => {
        wasm.adxr_js(high, low, close, 2);
    }, /HLC data length mismatch/);
});

test('ADXR reinput', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    
    const firstResult = wasm.adxr_js(high, low, close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    
    const secondResult = wasm.adxr_js(high, low, close, 5);
    assert.strictEqual(secondResult.length, close.length);
});

test('ADXR NaN handling', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.adxr_js(high, low, close, 14);
    assert.strictEqual(result.length, close.length);
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    
    const expectedWarmup = 2 * 14;  
    assertAllNaN(result.slice(0, expectedWarmup), "Expected NaN in warmup period");
});

test('ADXR all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.adxr_js(allNaN, allNaN, allNaN, 14);
    }, /All values are NaN/);
});

test('ADXR batch single parameter set', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.adxr_batch_js(
        high, low, close,
        14, 14, 0      
    );
    
    
    const singleResult = wasm.adxr_js(high, low, close, 14);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('ADXR batch multiple periods', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100)); 
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    
    const batchResult = wasm.adxr_batch_js(
        high, low, close,
        10, 20, 5      
    );
    
    
    assert.strictEqual(batchResult.length, 3 * 100);
    
    
    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.adxr_js(high, low, close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('ADXR batch metadata', () => {
    
    
    const metadata = wasm.adxr_batch_metadata_js(
        10, 20, 5      
    );
    
    
    assert.strictEqual(metadata.length, 3);
    
    
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 15);
    assert.strictEqual(metadata[2], 20);
});

test('ADXR batch full parameter sweep', () => {
    
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.adxr_batch_js(
        high, low, close,
        10, 14, 2      
    );
    
    const metadata = wasm.adxr_batch_metadata_js(10, 14, 2);
    
    
    const numCombos = metadata.length;
    assert.strictEqual(numCombos, 3);
    assert.strictEqual(batchResult.length, 3 * 50);
    
    
    for (let combo = 0; combo < numCombos; combo++) {
        const period = metadata[combo];
        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);
        
        
        
        
        
        
        let firstValidIndex = -1;
        for (let i = 0; i < 50; i++) {
            if (!isNaN(rowData[i])) {
                firstValidIndex = i;
                break;
            }
        }
        
        
        if (firstValidIndex !== -1) {
            for (let i = firstValidIndex + 1; i < 50; i++) {
                assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period} after first valid value at ${firstValidIndex}`);
            }
        }
    }
});

test('ADXR batch edge cases', () => {
    
    const high = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const low = new Float64Array([0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9]);
    const close = new Float64Array([0.95, 1.95, 2.95, 3.95, 4.95, 5.95, 6.95, 7.95, 8.95, 9.95]);
    
    
    const singleBatch = wasm.adxr_batch_js(
        high, low, close,
        5, 5, 1
    );
    
    assert.strictEqual(singleBatch.length, 10);
    
    
    const largeBatch = wasm.adxr_batch_js(
        high, low, close,
        5, 7, 10 
    );
    
    
    assert.strictEqual(largeBatch.length, 10);
    
    
    assert.throws(() => {
        wasm.adxr_batch_js(
            new Float64Array([]),
            new Float64Array([]),
            new Float64Array([]),
            9, 9, 0
        );
    }, /All values are NaN/);
});


test('ADXR batch - new ergonomic API with single parameter', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.adxr_batch(high, low, close, {
        period_range: [14, 14, 0]
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
    assert.strictEqual(combo.period, 14);
    
    
    const oldResult = wasm.adxr_js(high, low, close, 14);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; 
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('ADXR batch - new API with multiple parameters', () => {
    
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.adxr_batch(high, low, close, {
        period_range: [10, 14, 2]  
    });
    
    
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.values.length, 150);
    
    
    const expectedPeriods = [10, 12, 14];
    
    for (let i = 0; i < expectedPeriods.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedPeriods[i]);
    }
    
    
    const secondRow = result.values.slice(result.cols, 2 * result.cols);
    assert.strictEqual(secondRow.length, 50);
    
    
    const oldResult = wasm.adxr_js(high, low, close, 10);
    const firstRow = result.values.slice(0, result.cols);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
            continue; 
        }
        assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('ADXR batch - new API matches old API results', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const params = {
        period_range: [10, 15, 5]
    };
    
    
    const oldValues = wasm.adxr_batch_js(
        high, low, close,
        params.period_range[0], params.period_range[1], params.period_range[2]
    );
    
    
    const newResult = wasm.adxr_batch(high, low, close, params);
    
    
    assert.strictEqual(oldValues.length, newResult.values.length);
    
    for (let i = 0; i < oldValues.length; i++) {
        if (isNaN(oldValues[i]) && isNaN(newResult.values[i])) {
            continue; 
        }
        assert(Math.abs(oldValues[i] - newResult.values[i]) < 1e-10,
               `Value mismatch at index ${i}: old=${oldValues[i]}, new=${newResult.values[i]}`);
    }
});

test('ADXR batch - new API error handling', () => {
    const high = new Float64Array(testData.high.slice(0, 10));
    const low = new Float64Array(testData.low.slice(0, 10));
    const close = new Float64Array(testData.close.slice(0, 10));
    
    
    assert.throws(() => {
        wasm.adxr_batch(high, low, close, {
            period_range: [9, 9] 
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.adxr_batch(high, low, close, {
            
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.adxr_batch(high, low, close, {
            period_range: "invalid"
        });
    }, /Invalid config/);
});



test.after(() => {
    console.log('ADXR WASM tests completed');
});
