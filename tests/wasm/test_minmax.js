/**
 * WASM binding tests for MINMAX indicator.
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

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
    
    try {
        const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
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

/**
 * Helper function to unpack the new concatenated format into the old format
 * The new format returns {values: [...], rows: 4, cols: len}
 * where values = [is_min..., is_max..., last_min..., last_max...]
 */
function unpackMinmaxResult(result) {
    const { values, rows, cols } = result;
    assert.strictEqual(rows, 4, 'Expected 4 rows for minmax result');
    
    const is_min = values.slice(0, cols);
    const is_max = values.slice(cols, 2 * cols);
    const last_min = values.slice(2 * cols, 3 * cols);
    const last_max = values.slice(3 * cols, 4 * cols);
    
    return { is_min, is_max, last_min, last_max };
}

test('MINMAX partial params', () => {
    const high = testData.high;
    const low = testData.low;
    
    
    const rawResult = wasm.minmax_js(high, low, 3);
    const result = unpackMinmaxResult(rawResult);
    
    assert.strictEqual(result.is_min.length, high.length, 'is_min length mismatch');
    assert.strictEqual(result.is_max.length, high.length, 'is_max length mismatch');
    assert.strictEqual(result.last_min.length, high.length, 'last_min length mismatch');
    assert.strictEqual(result.last_max.length, high.length, 'last_max length mismatch');
});

test('MINMAX accuracy', () => {
    const high = testData.high;
    const low = testData.low;
    const order = 3;
    
    const rawResult = wasm.minmax_js(high, low, order);
    const result = unpackMinmaxResult(rawResult);
    
    assert.strictEqual(result.is_min.length, high.length, 'is_min length mismatch');
    
    
    const count = result.last_min.length;
    assert(count >= 5, 'Not enough data to check last 5');
    
    const expected_last_five_min = [57876.0, 57876.0, 57876.0, 57876.0, 57876.0];
    const expected_last_five_max = [60102.0, 60102.0, 60102.0, 60102.0, 60102.0];
    
    const last_min_slice = result.last_min.slice(-5);
    const last_max_slice = result.last_max.slice(-5);
    
    assertArrayClose(last_min_slice, expected_last_five_min, 0.1, 'MINMAX last_min mismatch');
    assertArrayClose(last_max_slice, expected_last_five_max, 0.1, 'MINMAX last_max mismatch');
});

test('MINMAX zero order', () => {
    const high = [10.0, 20.0, 30.0];
    const low = [1.0, 2.0, 3.0];
    
    assert.throws(() => {
        wasm.minmax_js(high, low, 0);
    }, /Invalid order/, 'Should throw error for zero order');
});

test('MINMAX order exceeds length', () => {
    const high = [10.0, 20.0, 30.0];
    const low = [1.0, 2.0, 3.0];
    
    assert.throws(() => {
        wasm.minmax_js(high, low, 10);
    }, /Invalid order/, 'Should throw error when order exceeds length');
});

test('MINMAX all NaN input', () => {
    const high = [NaN, NaN, NaN];
    const low = [NaN, NaN, NaN];
    
    assert.throws(() => {
        wasm.minmax_js(high, low, 1);
    }, /All values are NaN/, 'Should throw error for all NaN data');
});

test('MINMAX basic slices', () => {
    const high = [50.0, 55.0, 60.0, 55.0, 50.0, 45.0, 50.0, 55.0];
    const low = [40.0, 38.0, 35.0, 38.0, 40.0, 42.0, 41.0, 39.0];
    const order = 2;
    
    const rawResult = wasm.minmax_js(high, low, order);
    const result = unpackMinmaxResult(rawResult);
    
    assert.strictEqual(result.is_min.length, 8);
    assert.strictEqual(result.is_max.length, 8);
    assert.strictEqual(result.last_min.length, 8);
    assert.strictEqual(result.last_max.length, 8);
    
    
    
    
    assert(!isNaN(result.is_min[2]), 'Should have found a minimum at index 2');
    assert(!isNaN(result.is_max[2]), 'Should have found a maximum at index 2');
});

test('MINMAX batch processing - improved validation', () => {
    const high = testData.high.slice(0, 100); 
    const low = testData.low.slice(0, 100);
    
    
    const testConfigs = [
        { order_range: [2, 5, 1], expectedCombos: 4 },      
        { order_range: [1, 1, 0], expectedCombos: 1 },      
        { order_range: [10, 20, 5], expectedCombos: 3 },    
        { order_range: [3, 3, 0], expectedCombos: 1 },      
    ];
    
    for (const { order_range, expectedCombos } of testConfigs) {
        const result = wasm.minmax_batch(high, low, { order_range });
        
        
        assert.strictEqual(result.combos.length, expectedCombos, 
            `Should have ${expectedCombos} parameter combinations for range ${order_range}`);
        assert.strictEqual(result.rows, 4 * expectedCombos, 
            `Should have ${4 * expectedCombos} rows (4 series × ${expectedCombos} combos)`);
        assert.strictEqual(result.cols, 100, 'Should have 100 columns');
        assert.strictEqual(result.values.length, result.rows * result.cols, 'Values array size mismatch');
        
        
        for (let comboIdx = 0; comboIdx < result.combos.length; comboIdx++) {
            const order = result.combos[comboIdx].order || 3; 
            
            
            
            const batchIsMin = result.values.slice(
                comboIdx * result.cols, 
                (comboIdx + 1) * result.cols
            );
            const batchIsMax = result.values.slice(
                (expectedCombos + comboIdx) * result.cols,
                (expectedCombos + comboIdx + 1) * result.cols
            );
            const batchLastMin = result.values.slice(
                (2 * expectedCombos + comboIdx) * result.cols,
                (2 * expectedCombos + comboIdx + 1) * result.cols
            );
            const batchLastMax = result.values.slice(
                (3 * expectedCombos + comboIdx) * result.cols,
                (3 * expectedCombos + comboIdx + 1) * result.cols
            );
            
            
            const singleRawResult = wasm.minmax_js(high, low, order);
            const singleResult = unpackMinmaxResult(singleRawResult);
            
            
            let matchCount = 0;
            for (let i = order; i < Math.min(20, batchIsMin.length); i++) {
                if (!isNaN(batchIsMin[i]) && !isNaN(singleResult.is_min[i])) {
                    assertClose(batchIsMin[i], singleResult.is_min[i], 1e-9, 
                        `Batch vs single mismatch for is_min at order=${order}, index=${i}`);
                    matchCount++;
                }
            }
            
            
            if (order < 20) {
                assert(matchCount > 0, `No valid comparisons made for order=${order}`);
            }
        }
    }
});

test('MINMAX mismatched lengths', () => {
    const high = [50.0, 55.0, 60.0, 55.0, 50.0];
    const low = [40.0, 38.0, 35.0];  
    
    assert.throws(() => {
        wasm.minmax_js(high, low, 2);
    }, /Invalid order/, 'Should throw error for mismatched array lengths');
});

test('MINMAX warmup verification', () => {
    const high = testData.high.slice(0, 50);
    const low = testData.low.slice(0, 50);
    const order = 3;
    
    const rawResult = wasm.minmax_js(high, low, order);
    const result = unpackMinmaxResult(rawResult);
    
    
    let firstValidIdx = 0;
    for (let i = 0; i < high.length; i++) {
        if (!isNaN(high[i]) && !isNaN(low[i])) {
            firstValidIdx = i;
            break;
        }
    }
    
    
    for (let i = firstValidIdx; i < Math.min(firstValidIdx + order, result.is_min.length); i++) {
        assert(isNaN(result.is_min[i]), `Expected NaN at index ${i} during warmup for is_min`);
        assert(isNaN(result.is_max[i]), `Expected NaN at index ${i} during warmup for is_max`);
    }
    
    
    let hasMin = false;
    let hasMax = false;
    for (let i = firstValidIdx + order; i < result.last_min.length; i++) {
        if (!isNaN(result.last_min[i])) {
            hasMin = true;
        }
        if (!isNaN(result.last_max[i])) {
            hasMax = true;
        }
        
        
        if (hasMin && i > 0 && isNaN(result.is_min[i])) {
            assert.strictEqual(result.last_min[i], result.last_min[i-1], 
                `last_min should forward-fill at index ${i}`);
        }
        if (hasMax && i > 0 && isNaN(result.is_max[i])) {
            assert.strictEqual(result.last_max[i], result.last_max[i-1], 
                `last_max should forward-fill at index ${i}`);
        }
    }
});

test('MINMAX fast API', () => {
    const high = testData.high.slice(0, 100);
    const low = testData.low.slice(0, 100);
    const len = high.length;
    const order = 3;
    
    
    const high_ptr = wasm.minmax_alloc(len);
    const low_ptr = wasm.minmax_alloc(len);
    const is_min_ptr = wasm.minmax_alloc(len);
    const is_max_ptr = wasm.minmax_alloc(len);
    const last_min_ptr = wasm.minmax_alloc(len);
    const last_max_ptr = wasm.minmax_alloc(len);
    
    try {
        
        const highMemView = new Float64Array(wasm.__wasm.memory.buffer, high_ptr, len);
        const lowMemView = new Float64Array(wasm.__wasm.memory.buffer, low_ptr, len);
        
        
        highMemView.set(high);
        lowMemView.set(low);
        
        
        wasm.minmax_into(
            high_ptr,
            low_ptr,
            is_min_ptr,
            is_max_ptr,
            last_min_ptr,
            last_max_ptr,
            len,
            order
        );
        
        
        const is_min_result = new Float64Array(wasm.__wasm.memory.buffer, is_min_ptr, len);
        const is_max_result = new Float64Array(wasm.__wasm.memory.buffer, is_max_ptr, len);
        const last_min_result = new Float64Array(wasm.__wasm.memory.buffer, last_min_ptr, len);
        const last_max_result = new Float64Array(wasm.__wasm.memory.buffer, last_max_ptr, len);
        
        
        const safeRawResult = wasm.minmax_js(high, low, order);
        const safe_result = unpackMinmaxResult(safeRawResult);
        
        assertArrayClose(Array.from(is_min_result), safe_result.is_min, 1e-9, 'Fast vs safe API mismatch for is_min');
        assertArrayClose(Array.from(is_max_result), safe_result.is_max, 1e-9, 'Fast vs safe API mismatch for is_max');
        assertArrayClose(Array.from(last_min_result), safe_result.last_min, 1e-9, 'Fast vs safe API mismatch for last_min');
        assertArrayClose(Array.from(last_max_result), safe_result.last_max, 1e-9, 'Fast vs safe API mismatch for last_max');
        
    } finally {
        
        wasm.minmax_free(high_ptr, len);
        wasm.minmax_free(low_ptr, len);
        wasm.minmax_free(is_min_ptr, len);
        wasm.minmax_free(is_max_ptr, len);
        wasm.minmax_free(last_min_ptr, len);
        wasm.minmax_free(last_max_ptr, len);
    }
});

console.log('MINMAX WASM tests completed');