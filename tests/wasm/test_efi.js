/**
 * WASM binding tests for EFI indicator.
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

test('EFI partial params', () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.efi_js(close, volume, 13);
    assert.strictEqual(result.length, close.length);
});

test('EFI accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const expected = EXPECTED_OUTPUTS.efi;
    
    const result = wasm.efi_js(
        close,
        volume,
        expected.default_params.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last_5_values,
        1e-6,
        "EFI last 5 values mismatch"
    );
});

test('EFI zero period', () => {
    
    const price = new Float64Array([10.0, 20.0, 30.0]);
    const volume = new Float64Array([100.0, 200.0, 300.0]);
    
    assert.throws(() => {
        wasm.efi_js(price, volume, 0);
    }, /Invalid period/);
});

test('EFI period exceeds length', () => {
    
    const price = new Float64Array([10.0, 20.0, 30.0]);
    const volume = new Float64Array([100.0, 200.0, 300.0]);
    
    assert.throws(() => {
        wasm.efi_js(price, volume, 10);
    }, /Invalid period/);
});

test('EFI nan handling', () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.efi_js(close, volume, 13);
    assert.strictEqual(result.length, close.length);
    
    
    assert(isNaN(result[0]), "First value should be NaN");
    
    
    
    const nonNanStart = result.findIndex(v => !isNaN(v));
    assert(nonNanStart >= 0, "All values are NaN");
    
    
    if (result.length > 50) {
        const hasNaNAfter50 = result.slice(50).some(v => isNaN(v));
        assert(!hasNaNAfter50, "Found NaN values after warmup period");
    }
});

test('EFI empty data', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.efi_js(empty, empty, 13);
    }, /Empty data/);
});

test('EFI mismatched lengths', () => {
    
    const price = new Float64Array([1.0, 2.0, 3.0]);
    const volume = new Float64Array([100.0, 200.0]);  
    
    assert.throws(() => {
        wasm.efi_js(price, volume, 2);
    }, /Empty data/);
});

test('EFI all nan', () => {
    
    const allNan = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.efi_js(allNan, allNan, 13);
    }, /All values are NaN/);
});

test('EFI memory allocation', () => {
    
    const len = 1000;
    const ptr = wasm.efi_alloc(len);
    
    assert(ptr !== 0, "Failed to allocate memory");
    
    
    wasm.efi_free(ptr, len);
});

test('EFI fast API', () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const len = close.length;
    
    
    const pricePtr = wasm.efi_alloc(len);
    const volumePtr = wasm.efi_alloc(len);
    const outPtr = wasm.efi_alloc(len);
    
    try {
        
        const priceView = new Float64Array(wasm.__wasm.memory.buffer, pricePtr, len);
        const volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        priceView.set(close);
        volumeView.set(volume);
        
        
        wasm.efi_into(pricePtr, volumePtr, outPtr, len, 13);
        
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const result = Array.from(memory);
        
        
        const safeResult = wasm.efi_js(close, volume, 13);
        assertArrayClose(result, safeResult, 1e-10, "Fast API result mismatch");
    } finally {
        
        wasm.efi_free(pricePtr, len);
        wasm.efi_free(volumePtr, len);
        wasm.efi_free(outPtr, len);
    }
});

test('EFI batch processing', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    const config = {
        period_range: [10, 20, 5]  
    };
    
    const result = wasm.efi_batch(close, volume, config);
    
    assert(result.values, "Batch result missing values");
    assert(result.combos, "Batch result missing combos");
    assert.strictEqual(result.rows, 3, "Expected 3 rows");
    assert.strictEqual(result.cols, 100, "Expected 100 columns");
    assert.strictEqual(result.values.length, 300, "Expected 300 values total");
    
    
    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = result.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.efi_js(close, volume, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Batch row ${i} (period ${periods[i]}) mismatch`
        );
    }
});

test('EFI aliasing detection', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    const volume = new Float64Array(testData.volume.slice(0, 100));
    const len = close.length;
    
    
    const pricePtr = wasm.efi_alloc(len);
    const volumePtr = wasm.efi_alloc(len);
    
    try {
        
        const priceView = new Float64Array(wasm.__wasm.memory.buffer, pricePtr, len);
        const volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        priceView.set(close);
        volumeView.set(volume);
        
        
        wasm.efi_into(pricePtr, volumePtr, pricePtr, len, 13);
        
        
        const resultView = new Float64Array(wasm.__wasm.memory.buffer, pricePtr, len);
        
        
        assert(!isNaN(resultView[10]), "Aliasing produced NaN at index 10");
        assert(Math.abs(resultView[10]) > 0.001, "Aliasing produced zero");
        
        
        const safeResult = wasm.efi_js(close, volume, 13);
        assertArrayClose(Array.from(resultView), safeResult, 1e-10, "Aliasing result mismatch");
    } finally {
        
        wasm.efi_free(pricePtr, len);
        wasm.efi_free(volumePtr, len);
    }
});