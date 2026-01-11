/**
 * WASM binding tests for GatorOsc indicator.
 * These tests ensure WASM bindings work correctly for the 4-output Gator Oscillator.
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
    assertNoNaN
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

test('GatorOsc basic functionality', () => {
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.gatorosc_js(close, 13, 8, 8, 5, 5, 3);
    
    
    assert.strictEqual(result.rows, 4, 'Should have 4 output rows');
    assert.strictEqual(result.cols, close.length, 'Should have same length as input');
    assert.strictEqual(result.values.length, 4 * close.length, 'Should have 4x input length values');
    
    
    const upper = result.values.slice(0, close.length);
    const lower = result.values.slice(close.length, 2 * close.length);
    const upperChange = result.values.slice(2 * close.length, 3 * close.length);
    const lowerChange = result.values.slice(3 * close.length);
    
    
    let hasValidUpper = false;
    let hasValidLower = false;
    
    for (let i = 20; i < upper.length; i++) {
        if (!isNaN(upper[i])) {
            assert(upper[i] >= 0, `Upper value at ${i} should be non-negative: ${upper[i]}`);
            hasValidUpper = true;
        }
        if (!isNaN(lower[i])) {
            assert(lower[i] <= 0, `Lower value at ${i} should be non-positive: ${lower[i]}`);
            hasValidLower = true;
        }
    }
    
    assert(hasValidUpper, 'Should have valid upper values after warmup');
    assert(hasValidLower, 'Should have valid lower values after warmup');
});

test('GatorOsc fast API (no aliasing)', () => {
    const data = new Float64Array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]);
    const len = data.length;
    
    
    const inPtr = wasm.gatorosc_alloc(len);
    const upperPtr = wasm.gatorosc_alloc(len);
    const lowerPtr = wasm.gatorosc_alloc(len);
    const upperChangePtr = wasm.gatorosc_alloc(len);
    const lowerChangePtr = wasm.gatorosc_alloc(len);
    
    try {
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        memory.set(data, inPtr / 8);
        
        
        wasm.gatorosc_into(
            inPtr, upperPtr, lowerPtr, upperChangePtr, lowerChangePtr,
            len, 5, 3, 3, 2, 2, 1
        );
        
        
        const memory2 = new Float64Array(wasm.__wasm.memory.buffer);
        
        
        const upper = Array.from(memory2.slice(upperPtr / 8, upperPtr / 8 + len));
        const lower = Array.from(memory2.slice(lowerPtr / 8, lowerPtr / 8 + len));
        
        
        assert(upper.some(v => !isNaN(v)), 'Upper should have some non-NaN values');
        assert(lower.some(v => !isNaN(v)), 'Lower should have some non-NaN values');
        
    } finally {
        
        wasm.gatorosc_free(inPtr, len);
        wasm.gatorosc_free(upperPtr, len);
        wasm.gatorosc_free(lowerPtr, len);
        wasm.gatorosc_free(upperChangePtr, len);
        wasm.gatorosc_free(lowerChangePtr, len);
    }
});

test('GatorOsc fast API (with aliasing)', () => {
    const data = new Float64Array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]);
    const len = data.length;
    
    
    const inPtr = wasm.gatorosc_alloc(len);
    const lowerPtr = wasm.gatorosc_alloc(len);
    const upperChangePtr = wasm.gatorosc_alloc(len);
    const lowerChangePtr = wasm.gatorosc_alloc(len);
    
    try {
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        memory.set(data, inPtr / 8);
        
        
        wasm.gatorosc_into(
            inPtr, inPtr, lowerPtr, upperChangePtr, lowerChangePtr,  
            len, 5, 3, 3, 2, 2, 1
        );
        
        
        const memory2 = new Float64Array(wasm.__wasm.memory.buffer);
        
        
        const upper = Array.from(memory2.slice(inPtr / 8, inPtr / 8 + len));  
        const lower = Array.from(memory2.slice(lowerPtr / 8, lowerPtr / 8 + len));
        
        
        assert(upper.some(v => !isNaN(v)), 'Upper should have some non-NaN values');
        assert(lower.some(v => !isNaN(v)), 'Lower should have some non-NaN values');
        
    } finally {
        
        wasm.gatorosc_free(inPtr, len);
        wasm.gatorosc_free(lowerPtr, len);
        wasm.gatorosc_free(upperChangePtr, len);
        wasm.gatorosc_free(lowerChangePtr, len);
    }
});

test('GatorOsc error handling', () => {
    
    assert.throws(() => {
        wasm.gatorosc_js(new Float64Array([]), 13, 8, 8, 5, 5, 3);
    }, /empty|invalid|insufficient/i);
    
    
    const nanData = new Float64Array(50).fill(NaN);
    assert.throws(() => {
        wasm.gatorosc_js(nanData, 13, 8, 8, 5, 5, 3);
    }, /nan|invalid/i);
    
    
    const data = new Float64Array([1, 2, 3, 4, 5]);
    assert.throws(() => {
        wasm.gatorosc_js(data, 0, 8, 8, 5, 5, 3);  
    }, /invalid|parameter|length/i);
});

test('GatorOsc batch processing', () => {
    const close = new Float64Array(testData.close.slice(0, 100));  
    
    const config = {
        jaws_length_range: [10, 15, 5],
        jaws_shift_range: [6, 10, 2],
        teeth_length_range: [6, 10, 2],
        teeth_shift_range: [3, 6, 3],
        lips_length_range: [3, 6, 3],
        lips_shift_range: [2, 4, 2]
    };
    
    const result = wasm.gatorosc_batch(close, config);
    
    
    assert(result.rows > 0, 'Should have parameter combinations');
    assert.strictEqual(result.cols, close.length, 'Should match input length');
    assert.strictEqual(result.outputs, 4, 'Should have 4 outputs');
    assert.strictEqual(result.values.length, 4 * result.rows * result.cols, 'Values array size should match');
    assert(Array.isArray(result.combos), 'Should have combos array');
    assert.strictEqual(result.combos.length, result.rows, 'Combos should match rows');
    
    
    for (const combo of result.combos) {
        assert(combo.jaws_length !== undefined, 'Combo should have jaws_length');
        assert(combo.jaws_shift !== undefined, 'Combo should have jaws_shift');
        assert(combo.teeth_length !== undefined, 'Combo should have teeth_length');
        assert(combo.teeth_shift !== undefined, 'Combo should have teeth_shift');
        assert(combo.lips_length !== undefined, 'Combo should have lips_length');
        assert(combo.lips_shift !== undefined, 'Combo should have lips_shift');
    }
});

test('GatorOsc consistency', () => {
    const data = new Float64Array([
        50.0, 51.0, 52.0, 51.5, 53.0, 54.0, 53.5, 55.0, 56.0, 55.5,
        57.0, 58.0, 57.5, 59.0, 60.0, 59.5, 58.0, 57.0, 58.5, 59.0
    ]);
    
    
    const result1 = wasm.gatorosc_js(data, 5, 3, 3, 2, 2, 1);
    const result2 = wasm.gatorosc_js(data, 5, 3, 3, 2, 2, 1);
    
    
    assertArrayClose(
        result1.values, 
        result2.values, 
        1e-10,
        "GatorOsc should produce consistent results"
    );
});

test('GatorOsc memory allocation/deallocation', () => {
    
    for (let i = 0; i < 10; i++) {
        const len = 100 + i * 10;
        const ptr = wasm.gatorosc_alloc(len);
        assert(ptr > 0, 'Should return valid pointer');
        wasm.gatorosc_free(ptr, len);
    }
    
    
    wasm.gatorosc_free(0, 100);  
});