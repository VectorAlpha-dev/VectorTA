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
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);

    testData = loadTestData();
});

test('Pivot - basic functionality', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const open = new Float64Array(testData.open);
    
    
    const result = wasm.pivot_js(high, low, close, open, 3);
    
    
    assert.strictEqual(result.length, close.length * 9);
    
    
    const len = close.length;
    const r4 = result.slice(0, len);
    const r3 = result.slice(len, 2 * len);
    const r2 = result.slice(2 * len, 3 * len);
    const r1 = result.slice(3 * len, 4 * len);
    const pp = result.slice(4 * len, 5 * len);
    const s1 = result.slice(5 * len, 6 * len);
    const s2 = result.slice(6 * len, 7 * len);
    const s3 = result.slice(7 * len, 8 * len);
    const s4 = result.slice(8 * len, 9 * len);
    
    
    assert.strictEqual(r4.length, len);
    assert.strictEqual(pp.length, len);
    assert.strictEqual(s4.length, len);
});

test('Pivot - accuracy test (Camarilla)', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const open = new Float64Array(testData.open);
    
    const result = wasm.pivot_js(high, low, close, open, 3);
    const len = close.length;
    const r4 = result.slice(0, len);
    
    
    const last5_r4 = r4.slice(-5);
    const expected_r4 = [59466.5, 59357.55, 59243.6, 59334.85, 59170.35];
    
    assertArrayClose(last5_r4, expected_r4, 1e-1, "Camarilla R4 values");
});

test('Pivot - all calculation modes', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const open = new Float64Array(testData.open);
    
    
    const modes = [0, 1, 2, 3, 4]; 
    
    for (const mode of modes) {
        const result = wasm.pivot_js(high, low, close, open, mode);
        assert.strictEqual(result.length, close.length * 9, `Mode ${mode} should return 9 levels`);
        
        
        const len = close.length;
        const pp = result.slice(4 * len, 5 * len);
        let hasValidValue = false;
        for (const val of pp) {
            if (!isNaN(val)) {
                hasValidValue = true;
                break;
            }
        }
        assert(hasValidValue, `Mode ${mode} should produce valid pivot points`);
    }
});

test('Pivot - empty arrays', () => {
    const empty = new Float64Array(0);
    
    assert.throws(() => {
        wasm.pivot_js(empty, empty, empty, empty, 3);
    }, /EmptyData|required field/i);
});

test('Pivot - mismatched lengths', () => {
    const high = new Float64Array([1, 2, 3]);
    const low = new Float64Array([0.5, 1.5]);  
    const close = new Float64Array([0.8, 1.8, 2.8]);
    const open = new Float64Array([0.7, 1.7, 2.7]);
    
    assert.throws(() => {
        wasm.pivot_js(high, low, close, open, 3);
    }, /EmptyData|length/i);
});

test('Pivot - all NaN values', () => {
    const nanArray = new Float64Array(5).fill(NaN);
    
    assert.throws(() => {
        wasm.pivot_js(nanArray, nanArray, nanArray, nanArray, 3);
    }, /AllValuesNaN|all.*nan/i);
});

test('Pivot - batch processing', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const open = new Float64Array(testData.open);
    
    const config = {
        mode_range: [0, 4, 1]  
    };
    
    const result = wasm.pivot_batch(high, low, close, open, config);
    
    assert(result.values, 'Batch result should have values');
    assert(result.modes, 'Batch result should have modes');
    assert.strictEqual(result.rows, 45, 'Should have 45 rows (5 modes * 9 levels)');
    assert.strictEqual(result.cols, close.length, 'Should match input length');
    assert.strictEqual(result.modes.length, 5, 'Should have 5 modes');
    assert.strictEqual(result.values.length, result.rows * result.cols, 'Values array size should match');
});

test('Pivot - memory allocation', () => {
    const len = 1000;
    const ptr = wasm.pivot_alloc(len);
    
    assert(ptr !== 0, 'Allocated pointer should not be null');
    
    
    wasm.pivot_free(ptr, len);
    
    
    wasm.pivot_free(0, len); 
});

test('Pivot - fast API (pivot_into)', () => {
    const high = new Float64Array([100, 102, 105, 103, 104]);
    const low = new Float64Array([98, 99, 101, 100, 101]);
    const close = new Float64Array([99, 101, 104, 102, 103]);
    const open = new Float64Array([99, 100, 102, 103, 102]);
    const len = high.length;
    
    
    const high_ptr = wasm.pivot_alloc(len);
    const low_ptr = wasm.pivot_alloc(len);
    const close_ptr = wasm.pivot_alloc(len);
    const open_ptr = wasm.pivot_alloc(len);
    const r4_ptr = wasm.pivot_alloc(len);
    const r3_ptr = wasm.pivot_alloc(len);
    const r2_ptr = wasm.pivot_alloc(len);
    const r1_ptr = wasm.pivot_alloc(len);
    const pp_ptr = wasm.pivot_alloc(len);
    const s1_ptr = wasm.pivot_alloc(len);
    const s2_ptr = wasm.pivot_alloc(len);
    const s3_ptr = wasm.pivot_alloc(len);
    const s4_ptr = wasm.pivot_alloc(len);
    
    try {
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        const highStart = high_ptr / 8;
        const lowStart = low_ptr / 8;
        const closeStart = close_ptr / 8;
        const openStart = open_ptr / 8;
        memory.set(high, highStart);
        memory.set(low, lowStart);
        memory.set(close, closeStart);
        memory.set(open, openStart);
        
        
        wasm.pivot_into(
            high_ptr, low_ptr, close_ptr, open_ptr,
            r4_ptr, r3_ptr, r2_ptr, r1_ptr, pp_ptr, s1_ptr, s2_ptr, s3_ptr, s4_ptr,
            len, 3  
        );
        
        
        const memoryAfter = new Float64Array(wasm.__wasm.memory.buffer);
        const ppStart = pp_ptr / 8;
        const pp = memoryAfter.slice(ppStart, ppStart + len);
        
        
        assert(!isNaN(pp[0]), 'PP should have valid values from start for Camarilla');
        
    } finally {
        
        wasm.pivot_free(high_ptr, len);
        wasm.pivot_free(low_ptr, len);
        wasm.pivot_free(close_ptr, len);
        wasm.pivot_free(open_ptr, len);
        wasm.pivot_free(r4_ptr, len);
        wasm.pivot_free(r3_ptr, len);
        wasm.pivot_free(r2_ptr, len);
        wasm.pivot_free(r1_ptr, len);
        wasm.pivot_free(pp_ptr, len);
        wasm.pivot_free(s1_ptr, len);
        wasm.pivot_free(s2_ptr, len);
        wasm.pivot_free(s3_ptr, len);
        wasm.pivot_free(s4_ptr, len);
    }
});
