/**
 * WASM binding tests for FRAMA indicator.
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
    assertNoNaN
} from './test_utils.js';
import { compareWithRust } from './rust-comparison.js';

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

test('FRAMA partial params', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.frama_js(high, low, close, 10, 300, 1);
    assert.strictEqual(result.length, close.length);
    
    
    const resultCustomWindow = wasm.frama_js(high, low, close, 14, 300, 1);
    assert.strictEqual(resultCustomWindow.length, close.length);
    
    const resultCustomSc = wasm.frama_js(high, low, close, 10, 200, 1);
    assert.strictEqual(resultCustomSc.length, close.length);
    
    const resultCustomFc = wasm.frama_js(high, low, close, 10, 300, 2);
    assert.strictEqual(resultCustomFc.length, close.length);
});

test('FRAMA accuracy', async () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const expectedLast5 = [
        59337.23056930512,
        59321.607512374605,
        59286.677929994796,
        59268.00202402624,
        59160.03888720062
    ];
    
    const result = wasm.frama_js(high, low, close, 10, 300, 1);
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        0.1,  
        "FRAMA last 5 values mismatch"
    );
    
    
    await compareWithRust('frama', result, 'high,low,close', {window: 10, sc: 300, fc: 1});
});

test('FRAMA empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.frama_js(empty, empty, empty, 10, 300, 1);
    }, /Input data slice is empty/);
});

test('FRAMA zero window', () => {
    
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    const close = new Float64Array([7.0, 17.0, 27.0]);
    
    assert.throws(() => {
        wasm.frama_js(high, low, close, 0, 300, 1);
    }, /Invalid window/);
});

test('FRAMA window exceeds length', () => {
    
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    const close = new Float64Array([7.0, 17.0, 27.0]);
    
    assert.throws(() => {
        wasm.frama_js(high, low, close, 10, 300, 1);
    }, /Invalid window/);
});

test('FRAMA very small dataset', () => {
    
    const high = new Float64Array([42.0]);
    const low = new Float64Array([40.0]);
    const close = new Float64Array([41.0]);
    
    assert.throws(() => {
        wasm.frama_js(high, low, close, 10, 300, 1);
    }, /Invalid window|Not enough valid data/);
});

test('FRAMA mismatched lengths', () => {
    
    const high = new Float64Array([1.0, 2.0, 3.0]);
    const low = new Float64Array([0.5, 1.5]);
    const close = new Float64Array([1.0]);
    
    assert.throws(() => {
        wasm.frama_js(high, low, close, 10, 300, 1);
    }, /Mismatched slice lengths/);
});

test('FRAMA all NaN input', () => {
    
    const allNaN = new Float64Array(10);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.frama_js(allNaN, allNaN, allNaN, 10, 300, 1);
    }, /All values are NaN/);
});

test('FRAMA not enough valid data', () => {
    
    const high = new Float64Array([NaN, NaN, 10.0, 20.0, 30.0]);
    const low = new Float64Array([NaN, NaN, 5.0, 15.0, 25.0]);
    const close = new Float64Array([NaN, NaN, 7.0, 17.0, 27.0]);
    
    
    assert.throws(() => {
        wasm.frama_js(high, low, close, 10, 300, 1);
    }, /Invalid window/);
    
    
    const high2 = new Float64Array([NaN, NaN, NaN, NaN, NaN, 10.0, 20.0, 30.0, 40.0, 50.0]);
    const low2 = new Float64Array([NaN, NaN, NaN, NaN, NaN, 5.0, 15.0, 25.0, 35.0, 45.0]);
    const close2 = new Float64Array([NaN, NaN, NaN, NaN, NaN, 7.0, 17.0, 27.0, 37.0, 47.0]);
    
    
    assert.throws(() => {
        wasm.frama_js(high2, low2, close2, 10, 300, 1);
    }, /Not enough valid data/);
});

test('FRAMA NaN handling', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.frama_js(high, low, close, 10, 300, 1);
    assert.strictEqual(result.length, close.length);
    
    
    assertAllNaN(result.slice(0, 9), "Expected NaN in warmup period");
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('FRAMA batch single parameter set', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.frama_batch_js(
        high, low, close,
        10, 10, 0,    
        300, 300, 0,  
        1, 1, 0       
    );
    
    
    const singleResult = wasm.frama_js(high, low, close, 10, 300, 1);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('FRAMA batch multiple parameters', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    
    const batchResult = wasm.frama_batch_js(
        high, low, close,
        8, 12, 2,      
        200, 300, 100, 
        1, 2, 1        
    );
    
    
    assert.strictEqual(batchResult.length, 12 * 100);
    
    
    const firstRow = batchResult.slice(0, 100);
    const singleResult = wasm.frama_js(high, low, close, 8, 200, 1);
    assertArrayClose(
        firstRow, 
        singleResult, 
        1e-10, 
        "First batch row mismatch"
    );
});

test('FRAMA batch metadata', () => {
    
    const metadata = wasm.frama_batch_metadata_js(
        8, 12, 2,      
        200, 300, 100, 
        1, 2, 1        
    );
    
    
    assert.strictEqual(metadata.length, 12 * 3);
    
    
    assert.strictEqual(metadata[0], 8);    
    assert.strictEqual(metadata[1], 200);  
    assert.strictEqual(metadata[2], 1);    
    
    
    assert.strictEqual(metadata[3], 8);    
    assert.strictEqual(metadata[4], 200);  
    assert.strictEqual(metadata[5], 2);    
    
    
    assert.strictEqual(metadata[6], 8);    
    assert.strictEqual(metadata[7], 300);  
    assert.strictEqual(metadata[8], 1);    
});

test('FRAMA batch warmup validation', () => {
    
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.frama_batch_js(
        high, low, close,
        6, 10, 4,     
        300, 300, 0,  
        1, 1, 0       
    );
    
    const metadata = wasm.frama_batch_metadata_js(6, 10, 4, 300, 300, 0, 1, 1, 0);
    const numCombos = metadata.length / 3;
    assert.strictEqual(numCombos, 2);  
    
    
    for (let combo = 0; combo < numCombos; combo++) {
        const window = metadata[combo * 3];
        const warmup = window - 1;
        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);
        
        
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(rowData[i]), 
                `Expected NaN at index ${i} for window=${window}`);
        }
        
        
        for (let i = warmup; i < 50; i++) {
            assert(!isNaN(rowData[i]), 
                `Unexpected NaN at index ${i} for window=${window}`);
        }
    }
});

test('FRAMA batch edge cases', () => {
    
    const size = 20;
    const high = new Float64Array(size).fill(0).map((_, i) => i + 2);
    const low = new Float64Array(size).fill(0).map((_, i) => i);
    const close = new Float64Array(size).fill(0).map((_, i) => i + 1);
    
    
    const singleBatch = wasm.frama_batch_js(
        high, low, close,
        10, 10, 1,
        300, 300, 1,
        1, 1, 1
    );
    
    assert.strictEqual(singleBatch.length, size);
    
    
    const zeroStepBatch = wasm.frama_batch_js(
        high, low, close,
        10, 10, 0,
        300, 300, 0,
        1, 1, 0
    );
    
    assert.strictEqual(zeroStepBatch.length, size); 
    
    
    assert.throws(() => {
        wasm.frama_batch_js(
            new Float64Array([]),
            new Float64Array([]),
            new Float64Array([]),
            10, 10, 0,
            300, 300, 0,
            1, 1, 0
        );
    }, /Input data slice is empty|All values are NaN/);
});

test('FRAMA batch performance test', () => {
    
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const close = new Float64Array(testData.close.slice(0, 200));
    
    
    const startBatch = Date.now();
    const batchResult = wasm.frama_batch_js(
        high, low, close,
        8, 12, 2,     
        200, 400, 100,
        1, 2, 1       
    );
    const batchTime = Date.now() - startBatch;
    
    
    const startSingle = Date.now();
    const singleResults = [];
    for (let window = 8; window <= 12; window += 2) {
        for (let sc = 200; sc <= 400; sc += 100) {
            for (let fc = 1; fc <= 2; fc += 1) {
                singleResults.push(...wasm.frama_js(high, low, close, window, sc, fc));
            }
        }
    }
    const singleTime = Date.now() - startSingle;
    
    
    assert.strictEqual(batchResult.length, singleResults.length);
    
    
    console.log(`  FRAMA Batch time: ${batchTime}ms, Single calls time: ${singleTime}ms`);
});



test.after(() => {
    console.log('FRAMA WASM tests completed');
});