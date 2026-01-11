/**
 * WASM binding tests for WILLR indicator.
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

test('WILLR partial params', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);

    const result = wasm.willr_js(high, low, close, 14);
    assert.equal(result.length, close.length);
});

test('WILLR accuracy (last 5 match Rust refs)', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);

    const expectedLastFive = [
        -58.72876391329818,
        -61.77504393673111,
        -65.93438781487991,
        -60.27950310559006,
        -65.00449236298293,
    ];

    const result = wasm.willr_js(high, low, close, 14);
    assert.equal(result.length, close.length);

    const last5 = Array.from(result.slice(-5));
    assertArrayClose(last5, expectedLastFive, 1e-8, 'WILLR last 5 values mismatch');
});

test('WILLR zero period', () => {
    const high = new Float64Array([1.0, 2.0, 3.0]);
    const low = new Float64Array([0.5, 1.5, 2.5]);
    const close = new Float64Array([1.0, 2.0, 3.0]);
    assert.throws(() => wasm.willr_js(high, low, close, 0), /Invalid period/i);
});

test('WILLR period exceeds length', () => {
    const high = new Float64Array([1.0, 2.0, 3.0]);
    const low = new Float64Array([0.5, 1.5, 2.5]);
    const close = new Float64Array([1.0, 2.0, 3.0]);
    assert.throws(() => wasm.willr_js(high, low, close, 10), /Invalid period/i);
});

test('WILLR all NaN input', () => {
    const high = new Float64Array(10).fill(NaN);
    const low = new Float64Array(10).fill(NaN);
    const close = new Float64Array(10).fill(NaN);
    assert.throws(() => wasm.willr_js(high, low, close, 5), /All input values are NaN/i);
});

test('WILLR not enough valid data', () => {
    
    const high = new Float64Array([NaN, NaN, 2.0]);
    const low = new Float64Array([NaN, NaN, 1.0]);
    const close = new Float64Array([NaN, NaN, 1.5]);
    
    assert.throws(
        () => wasm.willr_js(high, low, close, 3),
        /Not enough valid data/i
    );
});

test('WILLR mismatched lengths', () => {
    const high = new Float64Array([1.0, 2.0, 3.0]);
    const low = new Float64Array([0.5, 1.5]);
    const close = new Float64Array([1.0, 2.0, 3.0]);
    assert.throws(
        () => wasm.willr_js(high, low, close, 2),
        /empty or mismatched|mismatched input lengths/i
    );
});

test('WILLR batch basic', () => {
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));

    const cfg = { period_range: [10, 20, 2] }; 
    const out = wasm.willr_batch(high, low, close, cfg);

    assert.equal(out.rows, 6);
    assert.equal(out.cols, 100);
    assert.equal(out.values.length, 6 * 100);

    
    const single = wasm.willr_js(high, low, close, 10);
    const row0 = Array.from(out.values.slice(0, 100));
    assertArrayClose(row0, Array.from(single), 1e-8, 'Batch row 0 vs single mismatch');
});

test.after(() => {
    
});
