/**
 * WASM binding tests for QSTICK indicator.
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
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const wasmUrl = new URL(`file:///${wasmPath.replace(/\\/g, '/')}`).href;
        wasm = await import(wasmUrl);
        
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('QSTICK accuracy', () => {
    const open_data = testData.open;
    const close_data = testData.close;
    const period = 5;
    
    
    const result = wasm.qstick_js(open_data, close_data, period);
    
    
    const expected_last_five = [219.4, 61.6, -51.8, -53.4, -123.2];
    
    
    for (let i = 0; i < expected_last_five.length; i++) {
        const actual = result[result.length - (5 - i)];
        assertClose(actual, expected_last_five[i], 1e-1, 
            `QSTICK mismatch at index ${i}: expected ${expected_last_five[i]}, got ${actual}`);
    }
});

test('QSTICK error handling', () => {
    
    assert.throws(() => {
        wasm.qstick_js(new Float64Array([10.0, 20.0]), new Float64Array([15.0, 25.0, 30.0]), 5);
    }, /must have the same length/);
    
    
    assert.throws(() => {
        wasm.qstick_js(new Float64Array([10.0, 20.0]), new Float64Array([15.0, 25.0]), 0);
    }, /Invalid period/);
    
    
    assert.throws(() => {
        wasm.qstick_js(new Float64Array([10.0, 20.0]), new Float64Array([15.0, 25.0]), 10);
    }, /Invalid period/);
    
    
    assert.throws(() => {
        wasm.qstick_js(new Float64Array([]), new Float64Array([]), 5);
    }, /Invalid period/);
});

test('QSTICK fast API', () => {
    const open_data = testData.open;
    const close_data = testData.close;
    const period = 5;
    const len = open_data.length;
    
    
    const out_ptr = wasm.qstick_alloc(len);
    const open_ptr = wasm.qstick_alloc(len);
    const close_ptr = wasm.qstick_alloc(len);
    
    try {
        
        let memory = wasm.__wasm.memory;
        const open_array = new Float64Array(memory.buffer, open_ptr, len);
        const close_array = new Float64Array(memory.buffer, close_ptr, len);
        open_array.set(open_data);
        close_array.set(close_data);
        
        
        wasm.qstick_into(open_ptr, close_ptr, out_ptr, len, period);
        
        
        memory = wasm.__wasm.memory;
        const result = new Float64Array(memory.buffer, out_ptr, len);
        const resultCopy = [...result]; 
        
        
        const expected = wasm.qstick_js(open_data, close_data, period);
        assertArrayClose(resultCopy, expected, 1e-10, 'Fast API should match safe API');
    } finally {
        
        wasm.qstick_free(out_ptr, len);
        wasm.qstick_free(open_ptr, len);
        wasm.qstick_free(close_ptr, len);
    }
});

test('QSTICK batch API', () => {
    const open_data = testData.open;
    const close_data = testData.close;
    
    
    const config = {
        period_range: [5, 20, 5]  
    };
    
    const result = wasm.qstick_batch(open_data, close_data, config);
    
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert.equal(result.rows, 4, 'Should have 4 rows');
    assert.equal(result.cols, open_data.length, 'Should have correct columns');
    assert.equal(result.combos.length, 4, 'Should have 4 parameter combinations');
    
    
    const single_result = wasm.qstick_js(open_data, close_data, 5);
    const first_row = result.values.slice(0, open_data.length);
    assertArrayClose(first_row, single_result, 1e-10, 'First batch row should match single calculation');
});

test.after(() => {
    console.log('QSTICK WASM tests completed');
});
