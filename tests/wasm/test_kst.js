/**
 * WASM binding tests for KST indicator.
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

test('KST default params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.kst(close, 10, 10, 10, 15, 10, 15, 20, 30, 9);
    assert(result.values, 'Result should have values property');
    assert(result.rows, 'Result should have rows property');
    assert(result.cols, 'Result should have cols property');
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, 2 * close.length);
    
    
    const line = result.values.slice(0, result.cols);
    const signal = result.values.slice(result.cols, 2 * result.cols);
    assert.strictEqual(line.length, close.length);
    assert.strictEqual(signal.length, close.length);
});

test('KST accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.kst;
    
    const result = wasm.kst(
        close,
        expected.defaultParams.sma_period1,
        expected.defaultParams.sma_period2,
        expected.defaultParams.sma_period3,
        expected.defaultParams.sma_period4,
        expected.defaultParams.roc_period1,
        expected.defaultParams.roc_period2,
        expected.defaultParams.roc_period3,
        expected.defaultParams.roc_period4,
        expected.defaultParams.signal_period
    );
    
    
    const line = result.values.slice(0, result.cols);
    const signal = result.values.slice(result.cols, 2 * result.cols);
    
    assert.strictEqual(line.length, close.length);
    assert.strictEqual(signal.length, close.length);
    
    
    const last5Line = line.slice(-5);
    const last5Signal = signal.slice(-5);
    assertArrayClose(
        last5Line,
        expected.last5Values.line,
        1e-1,  
        "KST line last 5 values mismatch"
    );
    assertArrayClose(
        last5Signal,
        expected.last5Values.signal,
        1e-1,  
        "KST signal last 5 values mismatch"
    );
    
    
    
});

test('KST all NaN values', () => {
    
    const inputData = new Float64Array(10).fill(NaN);
    
    assert.throws(() => {
        wasm.kst(inputData, 10, 10, 10, 15, 10, 15, 20, 30, 9);
    }, /All values are NaN/);
});

test('KST zero periods', () => {
    
    const inputData = new Float64Array(50).fill(0).map((_, i) => 10.0 + i);
    
    
    assert.throws(() => {
        wasm.kst(inputData, 0, 10, 10, 15, 10, 15, 20, 30, 9);
    }, /period|Period/);
    
    
    assert.throws(() => {
        wasm.kst(inputData, 10, 10, 10, 15, 0, 15, 20, 30, 9);
    }, /period|Period/);
    
    
    assert.throws(() => {
        wasm.kst(inputData, 10, 10, 10, 15, 10, 15, 20, 30, 0);
    }, /period|Period/);
});

test('KST insufficient data', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    assert.throws(() => {
        wasm.kst(dataSmall, 10, 10, 10, 15, 10, 15, 20, 30, 9);
    }, /Not enough|insufficient|Invalid/);
});

test('KST with NaN prefix', () => {
    
    const inputData = new Float64Array(100);
    inputData.fill(NaN, 0, 20);
    for (let i = 20; i < 100; i++) {
        inputData[i] = 100.0 + Math.sin(i * 0.1) * 10.0;
    }
    
    const result = wasm.kst(inputData, 5, 5, 5, 10, 5, 10, 15, 20, 5);
    
    
    const line = result.values.slice(0, result.cols);
    const signal = result.values.slice(result.cols, 2 * result.cols);
    
    assert.strictEqual(line.length, inputData.length);
    assert.strictEqual(signal.length, inputData.length);
    
    
    
    
    
    
    const warmup = 49;
    for (let i = 0; i < warmup; i++) {
        assert(isNaN(line[i]), `Line[${i}] should be NaN during warmup`);
        assert(isNaN(signal[i]), `Signal[${i}] should be NaN during warmup`);
    }
    
    
    for (let i = warmup + 10; i < line.length; i++) {
        assert(!isNaN(line[i]), `Line[${i}] should not be NaN after warmup`);
        assert(!isNaN(signal[i]), `Signal[${i}] should not be NaN after warmup`);
    }
});

test('KST fast/unsafe API', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const inPtr = wasm.kst_alloc(len);
    const lineOutPtr = wasm.kst_alloc(len);
    const signalOutPtr = wasm.kst_alloc(len);
    
    try {
        
        const inArray = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        
        
        inArray.set(close);
        
        
        wasm.kst_into(
            inPtr, lineOutPtr, signalOutPtr, len,
            10, 10, 10, 15, 10, 15, 20, 30, 9
        );
        
        
        const memBuffer1 = wasm.__wasm.memory.buffer;
        const lineResult = new Float64Array(memBuffer1, lineOutPtr, len);
        const signalResult = new Float64Array(memBuffer1, signalOutPtr, len);
        
        
        const safeResult = wasm.kst(close, 10, 10, 10, 15, 10, 15, 20, 30, 9);
        const safeLine = safeResult.values.slice(0, safeResult.cols);
        const safeSignal = safeResult.values.slice(safeResult.cols, 2 * safeResult.cols);
        
        
        const memBuffer2 = wasm.__wasm.memory.buffer;
        const lineResultFinal = memBuffer1 === memBuffer2 ? lineResult : new Float64Array(memBuffer2, lineOutPtr, len);
        const signalResultFinal = memBuffer1 === memBuffer2 ? signalResult : new Float64Array(memBuffer2, signalOutPtr, len);
        
        assertArrayClose(
            Array.from(lineResultFinal),
            safeLine,
            1e-10,
            "Fast API line should match safe API"
        );
        assertArrayClose(
            Array.from(signalResultFinal),
            safeSignal,
            1e-10,
            "Fast API signal should match safe API"
        );
    } finally {
        
        wasm.kst_free(inPtr, len);
        wasm.kst_free(lineOutPtr, len);
        wasm.kst_free(signalOutPtr, len);
    }
});

test('KST fast API with aliasing', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const ptr = wasm.kst_alloc(len);
    const signalPtr = wasm.kst_alloc(len);
    
    try {
        
        const array = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        
        
        array.set(close);
        
        
        wasm.kst_into(
            ptr, ptr, signalPtr, len,
            10, 10, 10, 15, 10, 15, 20, 30, 9
        );
        
        
        const lineResult = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        const signalResult = new Float64Array(wasm.__wasm.memory.buffer, signalPtr, len);
        
        
        const safeResult = wasm.kst(close, 10, 10, 10, 15, 10, 15, 20, 30, 9);
        const safeLine = safeResult.values.slice(0, safeResult.cols);
        const safeSignal = safeResult.values.slice(safeResult.cols, 2 * safeResult.cols);
        
        assertArrayClose(
            Array.from(lineResult),
            safeLine,
            1e-10,
            "Aliased fast API line should match safe API"
        );
        assertArrayClose(
            Array.from(signalResult),
            safeSignal,
            1e-10,
            "Aliased fast API signal should match safe API"
        );
    } finally {
        
        wasm.kst_free(ptr, len);
        wasm.kst_free(signalPtr, len);
    }
});

test('KST reinput', () => {
    
    const close = new Float64Array(testData.close.slice(0, 500));
    
    
    const firstResult = wasm.kst(close, 10, 10, 10, 15, 10, 15, 20, 30, 9);
    const firstLine = firstResult.values.slice(0, firstResult.cols);
    const firstSignal = firstResult.values.slice(firstResult.cols, 2 * firstResult.cols);
    
    
    const secondResult = wasm.kst(firstLine, 10, 10, 10, 15, 10, 15, 20, 30, 9);
    const secondLine = secondResult.values.slice(0, secondResult.cols);
    const secondSignal = secondResult.values.slice(secondResult.cols, 2 * secondResult.cols);
    
    assert.strictEqual(secondLine.length, firstLine.length);
    
    
    const warmupFirst = 44;
    
    
    
    const warmupSecond = 88;
    
    
    assert(isNaN(firstLine[0]), 'First line should have NaN at start');
    
    
    assert(isNaN(secondLine[0]), 'Second line should have NaN at start');
    assert(isNaN(secondLine[43]), 'Second line should have NaN during first warmup');
    
    
    if (secondLine.length > warmupSecond + 10) {
        assert(!isNaN(secondLine[warmupSecond + 10]), 'Should have valid value after cascaded warmup');
    }
});

test('KST warmup period', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result = wasm.kst(close, 10, 10, 10, 15, 10, 15, 20, 30, 9);
    const line = result.values.slice(0, result.cols);
    const signal = result.values.slice(result.cols, 2 * result.cols);
    
    
    
    const expectedWarmup = 44;
    
    
    for (let i = 0; i < expectedWarmup; i++) {
        assert(isNaN(line[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    
    for (let i = expectedWarmup; i < Math.min(expectedWarmup + 5, line.length); i++) {
        assert(!isNaN(line[i]), `Expected valid value at index ${i} after warmup`);
    }
    
    
    const signalWarmup = expectedWarmup + 9 - 1;
    for (let i = 0; i < signalWarmup; i++) {
        assert(isNaN(signal[i]), `Expected NaN in signal at index ${i} during warmup`);
    }
});

test('KST empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.kst(empty, 10, 10, 10, 15, 10, 15, 20, 30, 9);
    }, /empty|Empty/, 'Should fail with empty input');
});

test('KST batch calculation', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    const config = {
        sma_period1: [10, 10, 0],  
        sma_period2: [10, 10, 0],
        sma_period3: [10, 10, 0],
        sma_period4: [15, 15, 0],
        roc_period1: [10, 12, 2],  
        roc_period2: [15, 15, 0],
        roc_period3: [20, 20, 0],
        roc_period4: [30, 30, 0],
        signal_period: [9, 11, 2]  
    };
    
    const result = wasm.kst_batch(close, config);
    
    assert(result.values, 'Batch result should have values property');
    assert(result.combos, 'Batch result should have combos property');
    assert(result.rows, 'Batch result should have rows property');
    assert(result.cols, 'Batch result should have cols property');
    
    
    const numCombos = 2 * 2; 
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, numCombos);
    assert.strictEqual(result.rows, numCombos * 2); 
    assert.strictEqual(result.values.length, result.rows * result.cols);
    
    
    const firstCombo = result.combos[0];
    const firstLineRow = result.values.slice(0, result.cols);
    const firstSignalRow = result.values.slice(numCombos * result.cols, (numCombos + 1) * result.cols);
    
    
    const singleResult = wasm.kst(
        close,
        firstCombo.sma_period1,
        firstCombo.sma_period2,
        firstCombo.sma_period3,
        firstCombo.sma_period4,
        firstCombo.roc_period1,
        firstCombo.roc_period2,
        firstCombo.roc_period3,
        firstCombo.roc_period4,
        firstCombo.signal_period
    );
    
    const singleLine = singleResult.values.slice(0, singleResult.cols);
    const singleSignal = singleResult.values.slice(singleResult.cols, 2 * singleResult.cols);
    
    assertArrayClose(
        firstLineRow,
        singleLine,
        1e-10,
        "Batch first row line should match single calculation"
    );
    assertArrayClose(
        firstSignalRow,
        singleSignal,
        1e-10,
        "Batch first row signal should match single calculation"
    );
});

console.log('All KST WASM tests passed! 🎉');