/**
 * WASM binding tests for WAD indicator.
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

test('WAD partial params', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.wad_js(high, low, close);
    assert.strictEqual(result.length, close.length);
});

test('WAD accuracy', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.wad_js(high, low, close);
    
    assert.strictEqual(result.length, close.length);
    
    
    const expectedLastFive = [
        158503.46790000016,
        158279.46790000016,
        158014.46790000016,
        158186.46790000016,
        157605.46790000016,
    ];
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-4,
        "WAD last 5 values mismatch"
    );
});

test('WAD empty data', () => {
    
    assert.throws(() => {
        wasm.wad_js(new Float64Array([]), new Float64Array([]), new Float64Array([]));
    }, /Empty/);
});

test('WAD all values NaN', () => {
    
    const nanArray = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.wad_js(nanArray, nanArray, nanArray);
    }, /All values are NaN/);
});

test('WAD basic slice', () => {
    
    const high = new Float64Array([10.0, 11.0, 11.0, 12.0]);
    const low = new Float64Array([9.0, 9.0, 10.0, 10.0]);
    const close = new Float64Array([9.5, 10.5, 10.5, 11.5]);
    
    const result = wasm.wad_js(high, low, close);
    
    assert.strictEqual(result.length, 4);
    assertClose(result[0], 0.0, 1e-10);
    assertClose(result[1], 1.5, 1e-10);
    assertClose(result[2], 1.5, 1e-10);
    assertClose(result[3], 3.0, 1e-10);
});

test('WAD small example', () => {
    
    const high = new Float64Array([10.0, 11.0, 12.0, 11.5, 12.5]);
    const low = new Float64Array([9.0, 9.5, 11.0, 10.5, 11.0]);
    const close = new Float64Array([9.5, 10.5, 11.5, 11.0, 12.0]);
    const expected = [0.0, 1.0, 2.0, 1.5, 2.5];
    
    const result = wasm.wad_js(high, low, close);
    
    assert.strictEqual(result.length, 5);
    
    for (let i = 0; i < 5; i++) {
        assertClose(
            result[i],
            expected[i],
            1e-10,
            `WAD small example mismatch at index ${i}`
        );
    }
});

test('WAD batch single row', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.wad_batch(high, low, close, {
        dummy: [0, 0, 0]
    });
    
    
    const singleResult = wasm.wad_js(high, low, close);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, close.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('WAD mismatched array lengths', () => {
    
    const high = new Float64Array([1.0, 2.0, 3.0]);
    const low = new Float64Array([1.0, 2.0]);
    const close = new Float64Array([1.0, 2.0, 3.0]);
    
    assert.throws(() => {
        wasm.wad_js(high, low, close);
    }, /Empty/);
});


test('WAD zero-copy API', () => {
    const high = new Float64Array([10.0, 11.0, 12.0, 11.5, 12.5]);
    const low = new Float64Array([9.0, 9.5, 11.0, 10.5, 11.0]);
    const close = new Float64Array([9.5, 10.5, 11.5, 11.0, 12.0]);
    const len = high.length;
    
    
    const highPtr = wasm.wad_alloc(len);
    const lowPtr = wasm.wad_alloc(len);
    const closePtr = wasm.wad_alloc(len);
    const outPtr = wasm.wad_alloc(len);
    
    assert(highPtr !== 0, 'Failed to allocate high buffer');
    assert(lowPtr !== 0, 'Failed to allocate low buffer');
    assert(closePtr !== 0, 'Failed to allocate close buffer');
    assert(outPtr !== 0, 'Failed to allocate output buffer');
    
    try {
        
        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeMem = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const outMem = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        
        highMem.set(high);
        lowMem.set(low);
        closeMem.set(close);
        
        
        wasm.wad_into(highPtr, lowPtr, closePtr, outPtr, len);
        
        
        const regularResult = wasm.wad_js(high, low, close);
        for (let i = 0; i < len; i++) {
            assertClose(
                outMem[i],
                regularResult[i],
                1e-10,
                `Zero-copy mismatch at index ${i}`
            );
        }
    } finally {
        
        wasm.wad_free(highPtr, len);
        wasm.wad_free(lowPtr, len);
        wasm.wad_free(closePtr, len);
        wasm.wad_free(outPtr, len);
    }
});

test('WAD zero-copy aliasing detection', () => {
    const high = new Float64Array([10.0, 11.0, 12.0, 11.5, 12.5]);
    const low = new Float64Array([9.0, 9.5, 11.0, 10.5, 11.0]);
    const close = new Float64Array([9.5, 10.5, 11.5, 11.0, 12.0]);
    const len = high.length;
    
    
    const highPtr = wasm.wad_alloc(len);
    const lowPtr = wasm.wad_alloc(len);
    const closePtr = wasm.wad_alloc(len);
    
    try {
        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeMem = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        
        highMem.set(high);
        lowMem.set(low);
        closeMem.set(close);
        
        
        const originalHigh = new Float64Array(highMem);
        
        
        wasm.wad_into(highPtr, lowPtr, closePtr, highPtr, len);
        
        
        const expected = wasm.wad_js(originalHigh, low, close);
        
        
        const highMem2 = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        
        for (let i = 0; i < len; i++) {
            assertClose(
                highMem2[i],
                expected[i],
                1e-10,
                `Aliasing mismatch at index ${i}`
            );
        }
    } finally {
        wasm.wad_free(highPtr, len);
        wasm.wad_free(lowPtr, len);
        wasm.wad_free(closePtr, len);
    }
});

test('WAD zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.wad_into(0, 0, 0, 0, 10);
    }, /null pointer/);
    
    
    const ptr1 = wasm.wad_alloc(0);
    const ptr2 = wasm.wad_alloc(0);
    const ptr3 = wasm.wad_alloc(0);
    const ptr4 = wasm.wad_alloc(0);
    
    try {
        
        assert.throws(() => {
            wasm.wad_into(ptr1, ptr2, ptr3, ptr4, 0);
        }, /Empty/);
    } finally {
        wasm.wad_free(ptr1, 0);
        wasm.wad_free(ptr2, 0);
        wasm.wad_free(ptr3, 0);
        wasm.wad_free(ptr4, 0);
    }
});

test('WAD zero-copy memory management', () => {
    
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr1 = wasm.wad_alloc(size);
        const ptr2 = wasm.wad_alloc(size);
        const ptr3 = wasm.wad_alloc(size);
        const ptr4 = wasm.wad_alloc(size);
        
        assert(ptr1 !== 0, `Failed to allocate ${size} elements`);
        assert(ptr2 !== 0, `Failed to allocate ${size} elements`);
        assert(ptr3 !== 0, `Failed to allocate ${size} elements`);
        assert(ptr4 !== 0, `Failed to allocate ${size} elements`);
        
        
        const view1 = new Float64Array(wasm.__wasm.memory.buffer, ptr1, size);
        const view2 = new Float64Array(wasm.__wasm.memory.buffer, ptr2, size);
        const view3 = new Float64Array(wasm.__wasm.memory.buffer, ptr3, size);
        
        for (let i = 0; i < Math.min(10, size); i++) {
            view1[i] = i * 1.1;
            view2[i] = i * 2.2;
            view3[i] = i * 3.3;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(view1[i], i * 1.1, `Memory corruption at index ${i}`);
            assert.strictEqual(view2[i], i * 2.2, `Memory corruption at index ${i}`);
            assert.strictEqual(view3[i], i * 3.3, `Memory corruption at index ${i}`);
        }
        
        
        wasm.wad_free(ptr1, size);
        wasm.wad_free(ptr2, size);
        wasm.wad_free(ptr3, size);
        wasm.wad_free(ptr4, size);
    }
});

test('WAD batch edge cases', () => {
    
    const high = new Float64Array([1, 2, 3, 4, 5]);
    const low = new Float64Array([0.5, 1.5, 2.5, 3.5, 4.5]);
    const close = new Float64Array([0.8, 1.8, 2.8, 3.8, 4.8]);
    
    
    const batchResult = wasm.wad_batch(high, low, close, {
        dummy: [0, 0, 0]
    });
    
    assert.strictEqual(batchResult.values.length, 5);
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, 5);
    
    
    assert.throws(() => {
        wasm.wad_batch(new Float64Array([]), new Float64Array([]), new Float64Array([]), {
            dummy: [0, 0, 0]
        });
    }, /Empty/);
});

test.after(() => {
    console.log('WAD WASM tests completed');
});