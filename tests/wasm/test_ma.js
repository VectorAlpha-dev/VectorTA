/**
 * WASM binding tests for MA dispatcher.
 * These tests ensure the MA dispatcher correctly routes to various moving average implementations.
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

test('MA dispatcher - SMA', () => {
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.ma(close, "sma", 20);
    assert.strictEqual(result.length, close.length);
    
    
    const directResult = wasm.sma(close, 20);
    assertArrayClose(result, directResult, 1e-10, "MA dispatcher SMA mismatch");
});

test('MA dispatcher - EMA', () => {
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.ma(close, "ema", 20);
    assert.strictEqual(result.length, close.length);
    
    
    const directResult = wasm.ema_js(close, 20);
    assertArrayClose(result, directResult, 1e-10, "MA dispatcher EMA mismatch");
});

test('MA dispatcher - WMA', () => {
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.ma(close, "wma", 20);
    assert.strictEqual(result.length, close.length);
    
    
    const directResult = wasm.wma_js(close, 20);
    assertArrayClose(result, directResult, 1e-10, "MA dispatcher WMA mismatch");
});

test('MA dispatcher - ALMA with defaults', () => {
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.ma(close, "alma", 9);
    assert.strictEqual(result.length, close.length);
    
    
    const directResult = wasm.alma_js(close, 9, 0.85, 6.0);
    assertArrayClose(result, directResult, 1e-10, "MA dispatcher ALMA mismatch");
});

test('MA dispatcher - DEMA', () => {
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.ma(close, "dema", 20);
    assert.strictEqual(result.length, close.length);
    
    
    const directResult = wasm.dema_js(close, 20);
    assertArrayClose(result, directResult, 1e-10, "MA dispatcher DEMA mismatch");
});

test('MA dispatcher - TEMA', () => {
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.ma(close, "tema", 20);
    assert.strictEqual(result.length, close.length);
    
    
    const directResult = wasm.tema_js(close, 20);
    assertArrayClose(result, directResult, 1e-10, "MA dispatcher TEMA mismatch");
});

test('MA dispatcher - HMA', () => {
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.ma(close, "hma", 20);
    assert.strictEqual(result.length, close.length);
    
    
    const directResult = wasm.hma_js(close, 20);
    assertArrayClose(result, directResult, 1e-10, "MA dispatcher HMA mismatch");
});

test('MA dispatcher - case insensitive', () => {
    const close = new Float64Array(testData.close);
    
    
    const resultUpper = wasm.ma(close, "SMA", 20);
    const resultLower = wasm.ma(close, "sma", 20);
    assertArrayClose(resultUpper, resultLower, 1e-10, "MA dispatcher should be case insensitive");
    
    
    const resultMixed = wasm.ma(close, "EmA", 20);
    const emaResult = wasm.ema_js(close, 20);
    assertArrayClose(resultMixed, emaResult, 1e-10, "MA dispatcher should handle mixed case");
});

test('MA dispatcher - invalid type defaults to SMA', () => {
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.ma(close, "invalid_ma_type", 20);
    assert.strictEqual(result.length, close.length);
    
    
    const smaResult = wasm.sma(close, 20);
    assertArrayClose(result, smaResult, 1e-10, "MA dispatcher should default to SMA");
});

test('MA dispatcher - empty data should fail', () => {
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.ma(empty, "sma", 20);
    }, /empty|no data/i);
});

test('MA dispatcher - zero period should fail', () => {
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    assert.throws(() => {
        wasm.ma(data, "sma", 0);
    }, /period/i);
});

test('MA dispatcher - period exceeds length should fail', () => {
    const data = new Float64Array([1.0, 2.0, 3.0]);
    
    assert.throws(() => {
        wasm.ma(data, "sma", 10);
    }, /period|length/i);
});

test('MA dispatcher - all NaN values', () => {
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.ma(allNaN, "sma", 20);
    }, /NaN|all values/i);
});

test('MA dispatcher - multiple MA types', () => {
    const close = new Float64Array(testData.close.slice(0, 1000)); 
    
    
    const maTypes = [
        "sma", "ema", "dema", "tema", "smma", "zlema", "alma", "cwma",
        "edcf", "fwma", "gaussian", "highpass", "highpass2", "hma",
        "jma", "jsa", "kama", "linreg", "nma", "pwma", "reflex",
        "sinwma", "sqwma", "srwma", "supersmoother", "supersmoother_3_pole",
        "swma", "tilson", "trendflex", "trima", "wilders", "wma"
    ];
    
    for (const maType of maTypes) {
        try {
            const result = wasm.ma(close, maType, 20);
            assert.strictEqual(result.length, close.length, `MA type ${maType} returned wrong length`);
            
            
            let hasValidValues = false;
            for (let i = 100; i < result.length; i++) {
                if (!isNaN(result[i])) {
                    hasValidValues = true;
                    break;
                }
            }
            assert.ok(hasValidValues, `MA type ${maType} returned all NaN`);
        } catch (e) {
            assert.fail(`MA type ${maType} failed: ${e.message}`);
        }
    }
});

test('MA dispatcher - special MA types with defaults', () => {
    const close = new Float64Array(testData.close.slice(0, 1000)); 
    
    
    const hwmaResult = wasm.ma(close, "hwma", 20);
    assert.strictEqual(hwmaResult.length, close.length);
    
    
    const maaqResult = wasm.ma(close, "maaq", 20);
    assert.strictEqual(maaqResult.length, close.length);
    
    
    const mamaResult = wasm.ma(close, "mama", 20);
    assert.strictEqual(mamaResult.length, close.length);
    
    
    const mwdxResult = wasm.ma(close, "mwdx", 20);
    assert.strictEqual(mwdxResult.length, close.length);
    
    
    const ehlersResult = wasm.ma(close, "ehlers_itrend", 50);
    assert.strictEqual(ehlersResult.length, close.length);
});

test('MA dispatcher - KAMA', () => {
    const close = new Float64Array(testData.close);
    
    const result = wasm.ma(close, "kama", 10);
    assert.strictEqual(result.length, close.length);
    
    
    const directResult = wasm.kama_js(close, 10);
    assertArrayClose(result, directResult, 1e-10, "MA dispatcher KAMA mismatch");
});

test('MA dispatcher - ZLEMA', () => {
    const close = new Float64Array(testData.close);
    
    const result = wasm.ma(close, "zlema", 20);
    assert.strictEqual(result.length, close.length);
    
    
    const directResult = wasm.zlema_js(close, 20);
    assertArrayClose(result, directResult, 1e-10, "MA dispatcher ZLEMA mismatch");
});

test('MA dispatcher - Wilders', () => {
    const close = new Float64Array(testData.close);
    
    const result = wasm.ma(close, "wilders", 14);
    assert.strictEqual(result.length, close.length);
    
    
    const directResult = wasm.wilders_js(close, 14);
    assertArrayClose(result, directResult, 1e-10, "MA dispatcher Wilders mismatch");
});