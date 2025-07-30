/**
 * WASM binding tests for RSMK indicator.
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
    // Load WASM module
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('RSMK basic functionality', () => {
    // Test with default parameters
    const close = new Float64Array(testData.close);
    
    // rsmk_js returns flattened [indicator..., signal...]
    const result = wasm.rsmk_js(close, close, 90, 3, 20, null, null);
    assert.strictEqual(result.length, close.length * 2); // Two outputs
    
    // Split results
    const indicator = result.slice(0, close.length);
    const signal = result.slice(close.length);
    
    assert.strictEqual(indicator.length, close.length);
    assert.strictEqual(signal.length, close.length);
});

test('RSMK with custom MA types', () => {
    const close = new Float64Array(testData.close);
    
    const result = wasm.rsmk_js(close, close, 90, 3, 20, "sma", "ema");
    assert.strictEqual(result.length, close.length * 2);
});

test('RSMK error handling - zero period', () => {
    const inputData = new Float64Array([10.0, 11.0, 12.0]);
    
    assert.throws(() => {
        wasm.rsmk_js(inputData, inputData, 0, 0, 0, null, null);
    }, /Invalid period/);
});

test('RSMK error handling - insufficient data', () => {
    const inputData = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.rsmk_js(inputData, inputData, 90, 3, 20, null, null);
    }, /Not enough data/);
});

test('RSMK error handling - all NaN', () => {
    const inputData = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.rsmk_js(inputData, inputData, 2, 1, 1, null, null);
    }, /All values are NaN/);
});

test('RSMK error handling - invalid MA type', () => {
    const inputData = new Float64Array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    
    assert.throws(() => {
        wasm.rsmk_js(inputData, inputData, 2, 3, 3, "nonexistent_ma", "ema");
    }, /Ma function error/);
});

test('RSMK fast API - in-place operation', () => {
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate buffers
    const indicatorPtr = wasm.rsmk_alloc(len);
    const signalPtr = wasm.rsmk_alloc(len);
    
    try {
        // Test in-place (aliasing)
        const closePtr = indicatorPtr; // Aliased pointer
        const comparePtr = signalPtr; // Another aliased pointer
        
        // Copy data to buffers
        const indicatorBuf = new Float64Array(wasm.memory.buffer, indicatorPtr, len);
        const signalBuf = new Float64Array(wasm.memory.buffer, signalPtr, len);
        indicatorBuf.set(close);
        signalBuf.set(close);
        
        // Call fast API with aliased pointers
        wasm.rsmk_into(closePtr, indicatorPtr, signalPtr, len, comparePtr, 90, 3, 20, null, null);
        
        // Results should be written correctly despite aliasing
        assert.strictEqual(indicatorBuf.length, len);
        assert.strictEqual(signalBuf.length, len);
    } finally {
        wasm.rsmk_free(indicatorPtr, len);
        wasm.rsmk_free(signalPtr, len);
    }
});

test('RSMK batch processing', () => {
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use smaller dataset for batch
    
    const config = {
        lookback_range: [85, 95, 5],
        period_range: [2, 4, 1],
        signal_period_range: [18, 22, 2],
        matype: "ema",
        signal_matype: "ema"
    };
    
    const result = wasm.rsmk_batch(close, close, config);
    
    // Check structure
    assert(result.indicators);
    assert(result.signals);
    assert(result.combos);
    assert(result.rows > 0);
    assert(result.cols === close.length);
    
    // Check flattened array sizes
    assert.strictEqual(result.indicators.length, result.rows * result.cols);
    assert.strictEqual(result.signals.length, result.rows * result.cols);
    assert.strictEqual(result.combos.length, result.rows);
});

test.after(() => {
    console.log('RSMK WASM tests completed');
});
