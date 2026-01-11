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

test('Bollinger Bands - partial params', () => {
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.bollinger_bands_js(close, 22, 2.0, 2.0, "sma", 0);
    
    
    assert.strictEqual(result.values.length, close.length * 3);
});


test('Bollinger Bands - accuracy test', () => {
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.bollinger_bands_js(close, 20, 2.0, 2.0, "sma", 0);
    
    
    const len = close.length;
    const upper = result.values.slice(0, len);
    const middle = result.values.slice(len, 2 * len);
    const lower = result.values.slice(2 * len, 3 * len);
    
    
    const expectedMiddle = [
        59403.199999999975,
        59423.24999999998,
        59370.49999999998,
        59371.39999999998,
        59351.299999999974,
    ];
    const expectedLower = [
        58299.51497247008,
        58351.47038179873,
        58332.65135978715,
        58334.33194052157,
        58275.767369163135,
    ];
    const expectedUpper = [
        60506.88502752987,
        60495.029618201224,
        60408.348640212804,
        60408.468059478386,
        60426.83263083681,
    ];
    
    
    const startIdx = len - 5;
    assertArrayClose(
        upper.slice(startIdx),
        expectedUpper,
        1e-4,
        'Upper band last 5 values mismatch'
    );
    assertArrayClose(
        middle.slice(startIdx),
        expectedMiddle,
        1e-4,
        'Middle band last 5 values mismatch'
    );
    assertArrayClose(
        lower.slice(startIdx),
        expectedLower,
        1e-4,
        'Lower band last 5 values mismatch'
    )
});

test('Bollinger Bands - zero period should fail', () => {
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.bollinger_bands_js(inputData, 0, 2.0, 2.0, "sma", 0);
    }, /Invalid period/);
});

test('Bollinger Bands - period exceeds length should fail', () => {
    const smallData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.bollinger_bands_js(smallData, 10, 2.0, 2.0, "sma", 0);
    }, /Invalid period|period exceeds/);
});

test('Bollinger Bands - very small dataset should fail', () => {
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.bollinger_bands_js(singlePoint, 20, 2.0, 2.0, "sma", 0);
    }, /period|length/);
});










test('Bollinger Bands - empty data should fail', () => {
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.bollinger_bands_js(empty, 20, 2.0, 2.0, "sma", 0);
    }, /empty|no data/i);
});

test('Bollinger Bands - all NaN values', () => {
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.bollinger_bands_js(allNaN, 20, 2.0, 2.0, "sma", 0);
    }, /NaN|all values/i);
});

test('Bollinger Bands - reinput test', () => {
    const close = new Float64Array(testData.close);
    
    
    const result1 = wasm.bollinger_bands_js(close, 20, 2.0, 2.0, "sma", 0);
    const len = close.length;
    const middle1 = result1.values.slice(len, 2 * len);
    
    
    const result2 = wasm.bollinger_bands_js(middle1, 10, 2.0, 2.0, "sma", 0);
    
    assert.strictEqual(result2.values.length, middle1.length * 3);
});

test('Bollinger Bands - NaN handling', () => {
    const close = new Float64Array(testData.close);
    
    const result = wasm.bollinger_bands_js(close, 20, 2.0, 2.0, "sma", 0);
    const len = close.length;
    const upper = result.values.slice(0, len);
    const middle = result.values.slice(len, 2 * len);
    const lower = result.values.slice(2 * len, 3 * len);
    
    
    if (len > 240) {
        for (let i = 240; i < len; i++) {
            assert.ok(!isNaN(upper[i]), `NaN found in upper at ${i}`);
            assert.ok(!isNaN(middle[i]), `NaN found in middle at ${i}`);
            assert.ok(!isNaN(lower[i]), `NaN found in lower at ${i}`);
        }
    }
    
    
    for (let i = 0; i < 19; i++) {
        assert.ok(isNaN(upper[i]), `Expected NaN in upper warmup at ${i}`);
        assert.ok(isNaN(middle[i]), `Expected NaN in middle warmup at ${i}`);
        assert.ok(isNaN(lower[i]), `Expected NaN in lower warmup at ${i}`);
    }
});

test('Bollinger Bands - batch single params', () => {
    const close = new Float64Array(testData.close);
    
    const result = wasm.bollinger_bands_batch_js(
        close,
        20, 20, 0,  
        2.0, 2.0, 0.0,  
        2.0, 2.0, 0.0,  
        "sma",
        0
    );
    
    
    const expectedLength = 1 * close.length * 3;
    assert.strictEqual(result.length, expectedLength);
    
    
    const len = close.length;
    const middle = result.slice(len, 2 * len);
    
    
    const singleResult = wasm.bollinger_bands_js(close, 20, 2.0, 2.0, "sma", 0);
    const singleMiddle = singleResult.values.slice(len, 2 * len);
    
    
    assertArrayClose(middle, singleMiddle, 1e-10, "Batch vs single calculation mismatch");
});

test('Bollinger Bands - batch multiple params', () => {
    const close = new Float64Array(testData.close.slice(0, 200)); 
    
    const result = wasm.bollinger_bands_batch_js(
        close,
        10, 30, 10,  
        1.0, 3.0, 1.0,  
        2.0, 2.0, 0.0,  
        "sma",
        0
    );
    
    
    const expectedLength = 9 * close.length * 3;
    assert.strictEqual(result.length, expectedLength);
});

test('Bollinger Bands - batch metadata', () => {
    const metadata = wasm.bollinger_bands_batch_metadata_js(
        10, 30, 10,  
        1.0, 3.0, 1.0,  
        2.0, 2.0, 0.0,  
        "sma",
        0
    );
    
    
    assert.strictEqual(metadata.length, 9 * 4);
    
    
    assert.strictEqual(metadata[0], 10);  
    assert.strictEqual(metadata[1], 1.0); 
    assert.strictEqual(metadata[2], 2.0); 
    assert.strictEqual(metadata[3], 0);   
});

test('Bollinger Bands - ergonomic API', () => {
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    const config = {
        period_range: [20, 20, 0],
        devup_range: [2.0, 2.0, 0.0],
        devdn_range: [2.0, 2.0, 0.0],
        matype: "sma",
        devtype: 0
    };
    
    const result = wasm.bollinger_bands_batch(close, config);
    
    assert.ok(result.upper);
    assert.ok(result.middle);
    assert.ok(result.lower);
    assert.ok(result.combos);
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    
    
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.combos[0].period, 20);
    assert.strictEqual(result.combos[0].devup, 2.0);
    assert.strictEqual(result.combos[0].devdn, 2.0);
    assert.strictEqual(result.combos[0].matype, "sma");
    assert.strictEqual(result.combos[0].devtype, 0);
});


test('Bollinger Bands - different MA types', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    const maTypes = ["sma", "ema", "wma"];
    
    for (const maType of maTypes) {
        const result = wasm.bollinger_bands_js(close, 20, 2.0, 2.0, maType, 0);
        assert.strictEqual(result.values.length, close.length * 3, `Failed for matype: ${maType}`);
        
        
        const len = close.length;
        const upper = result.values.slice(0, len);
        const middle = result.values.slice(len, 2 * len);
        const lower = result.values.slice(2 * len, 3 * len);
        
        
        assert.ok(upper.every(v => typeof v === 'number'), 'Upper band has numeric values');
        assert.ok(middle.every(v => typeof v === 'number'), 'Middle band has numeric values');
        assert.ok(lower.every(v => typeof v === 'number'), 'Lower band has numeric values');
    }
});

test('Bollinger Bands - different deviation types', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    
    
    const result0 = wasm.bollinger_bands_js(close, 20, 2.0, 2.0, "sma", 0);
    const len = close.length;
    const upper0 = result0.values.slice(0, len);
    
    
    const result1 = wasm.bollinger_bands_js(close, 20, 2.0, 2.0, "sma", 1);
    const upper1 = result1.values.slice(0, len);
    
    
    let foundDifference = false;
    for (let i = 20; i < len; i++) {
        if (!isNaN(upper0[i]) && !isNaN(upper1[i])) {
            if (Math.abs(upper0[i] - upper1[i]) > 1e-8) {
                foundDifference = true;
                break;
            }
        }
    }
    assert.ok(foundDifference, "Different deviation types should produce different results");
});

test('Bollinger Bands - asymmetric bands', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    
    
    const result = wasm.bollinger_bands_js(close, 20, 3.0, 1.0, "sma", 0);
    const len = close.length;
    const upper = result.values.slice(0, len);
    const middle = result.values.slice(len, 2 * len);
    const lower = result.values.slice(2 * len, 3 * len);
    
    
    for (let i = 20; i < len && i < 30; i++) {
        if (!isNaN(upper[i]) && !isNaN(middle[i]) && !isNaN(lower[i])) {
            const upperDist = upper[i] - middle[i];
            const lowerDist = middle[i] - lower[i];
            
            const ratio = upperDist / lowerDist;
            assert.ok(Math.abs(ratio - 3.0) < 0.1, `Asymmetric ratio incorrect at ${i}: ${ratio}`);
        }
    }
});


test('Bollinger Bands - fast API basic operation', () => {
    const data = new Float64Array(testData.close);
    const len = data.length;
    
    
    const dataPtr = wasm.bollinger_bands_alloc(len);
    const upperPtr = wasm.bollinger_bands_alloc(len);
    const middlePtr = wasm.bollinger_bands_alloc(len);
    const lowerPtr = wasm.bollinger_bands_alloc(len);
    
    assert(dataPtr !== 0, 'Failed to allocate input memory');
    assert(upperPtr !== 0, 'Failed to allocate upper band memory');
    assert(middlePtr !== 0, 'Failed to allocate middle band memory');
    assert(lowerPtr !== 0, 'Failed to allocate lower band memory');
    
    
    const dataView = new Float64Array(wasm.__wasm.memory.buffer, dataPtr, len);
    dataView.set(data);
    
    try {
        
        wasm.bollinger_bands_into(
            dataPtr, 
            upperPtr, 
            middlePtr, 
            lowerPtr, 
            len,
            20,
            2.0,
            2.0,
            "sma",
            0
        );
        
        
        const upper = new Float64Array(wasm.__wasm.memory.buffer, upperPtr, len);
        const middle = new Float64Array(wasm.__wasm.memory.buffer, middlePtr, len);
        const lower = new Float64Array(wasm.__wasm.memory.buffer, lowerPtr, len);
        
        
        const safeResult = wasm.bollinger_bands_js(data, 20, 2.0, 2.0, "sma", 0);
        const safeUpper = safeResult.values.slice(0, len);
        const safeMiddle = safeResult.values.slice(len, 2 * len);
        const safeLower = safeResult.values.slice(2 * len, 3 * len);
        
        
        for (let i = 0; i < len; i++) {
            if (isNaN(safeUpper[i]) && isNaN(upper[i])) continue;
            if (isNaN(safeMiddle[i]) && isNaN(middle[i])) continue;
            if (isNaN(safeLower[i]) && isNaN(lower[i])) continue;
            
            assertClose(upper[i], safeUpper[i], 1e-10, `Upper band mismatch at ${i}`);
            assertClose(middle[i], safeMiddle[i], 1e-10, `Middle band mismatch at ${i}`);
            assertClose(lower[i], safeLower[i], 1e-10, `Lower band mismatch at ${i}`);
        }
    } finally {
        
    wasm.bollinger_bands_free(dataPtr, len);
    wasm.bollinger_bands_free(upperPtr, len);
    wasm.bollinger_bands_free(middlePtr, len);
    wasm.bollinger_bands_free(lowerPtr, len);
    }
});

test('Bollinger Bands - fast API in-place operation (aliasing)', () => {
    const data = new Float64Array(testData.close.slice(0, 100));
    const len = data.length;
    
    
    const ptr1 = wasm.bollinger_bands_alloc(len);
    const ptr2 = wasm.bollinger_bands_alloc(len);
    const ptr3 = wasm.bollinger_bands_alloc(len);
    
    try {
        
        const wasmData = new Float64Array(wasm.__wasm.memory.buffer, ptr1, len);
        wasmData.set(data);
        
        
        wasm.bollinger_bands_into(
            ptr1,  
            ptr1,  
            ptr2,  
            ptr3,  
            len,
            20,
            2.0,
            2.0,
            "sma",
            0
        );
        
        
        const safeResult = wasm.bollinger_bands_js(data, 20, 2.0, 2.0, "sma", 0);
        const safeUpper = safeResult.values.slice(0, len);
        
        const upper = new Float64Array(wasm.__wasm.memory.buffer, ptr1, len);
        
        
        for (let i = 0; i < len; i++) {
            if (isNaN(safeUpper[i]) && isNaN(upper[i])) continue;
            assertClose(upper[i], safeUpper[i], 1e-10, `Aliased upper band mismatch at ${i}`);
        }
    } finally {
        wasm.bollinger_bands_free(ptr1, len);
        wasm.bollinger_bands_free(ptr2, len);
        wasm.bollinger_bands_free(ptr3, len);
    }
});

test('Bollinger Bands - fast API error handling', () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    const len = data.length;
    
    
    assert.throws(() => {
        wasm.bollinger_bands_into(0, 0, 0, 0, len, 20, 2.0, 2.0, "sma", 0);
    }, /null pointer/i);
    
    
    const dataPtr = wasm.bollinger_bands_alloc(len);
    const dataView = new Float64Array(wasm.__wasm.memory.buffer, dataPtr, len);
    dataView.set(data);
    
    
    const upperPtr = wasm.bollinger_bands_alloc(len);
    const middlePtr = wasm.bollinger_bands_alloc(len);
    const lowerPtr = wasm.bollinger_bands_alloc(len);
    
    try {
        
        assert.throws(() => {
            wasm.bollinger_bands_into(
                dataPtr, upperPtr, middlePtr, lowerPtr, 
                len, 0, 2.0, 2.0, "sma", 0
            );
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.bollinger_bands_into(
                dataPtr, upperPtr, middlePtr, lowerPtr,
                len, 10, 2.0, 2.0, "sma", 0
            );
        }, /Invalid period|period exceeds/);
    } finally {
        wasm.bollinger_bands_free(dataPtr, len);
        wasm.bollinger_bands_free(upperPtr, len);
        wasm.bollinger_bands_free(middlePtr, len);
        wasm.bollinger_bands_free(lowerPtr, len);
    }
});

test('Bollinger Bands - fast API memory management', () => {
    const sizes = [100, 1000, 10000];
    
    sizes.forEach(size => {
        const upperPtr = wasm.bollinger_bands_alloc(size);
        const middlePtr = wasm.bollinger_bands_alloc(size);
        const lowerPtr = wasm.bollinger_bands_alloc(size);
        
        assert(upperPtr !== 0, `Failed to allocate upper ${size} elements`);
        assert(middlePtr !== 0, `Failed to allocate middle ${size} elements`);
        assert(lowerPtr !== 0, `Failed to allocate lower ${size} elements`);
        
        
        const upper = new Float64Array(wasm.__wasm.memory.buffer, upperPtr, size);
        const middle = new Float64Array(wasm.__wasm.memory.buffer, middlePtr, size);
        const lower = new Float64Array(wasm.__wasm.memory.buffer, lowerPtr, size);
        
        upper[0] = 42.0;
        upper[size - 1] = 99.0;
        middle[0] = 43.0;
        lower[0] = 44.0;
        
        assert.strictEqual(upper[0], 42.0);
        assert.strictEqual(upper[size - 1], 99.0);
        assert.strictEqual(middle[0], 43.0);
        assert.strictEqual(lower[0], 44.0);
        
        
        wasm.bollinger_bands_free(upperPtr, size);
        wasm.bollinger_bands_free(middlePtr, size);
        wasm.bollinger_bands_free(lowerPtr, size);
    });
});

test('Bollinger Bands - fast API complex aliasing scenarios', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const len = data.length;
    
    
    const ptr = wasm.bollinger_bands_alloc(len);
    
    try {
        
        const wasmData = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        wasmData.set(data);
        
        
        wasm.bollinger_bands_into(
            ptr,  
            ptr,  
            ptr,  
            ptr,  
            len,
            3,
            2.0,
            2.0,
            "sma",
            0
        );
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        
        
        const safeResult = wasm.bollinger_bands_js(data, 3, 2.0, 2.0, "sma", 0);
        const safeLower = safeResult.values.slice(2 * len, 3 * len);
        
        for (let i = 0; i < len; i++) {
            if (isNaN(safeLower[i]) && isNaN(result[i])) continue;
            assertClose(result[i], safeLower[i], 1e-10, 
                       `Extreme aliasing result mismatch at ${i}`);
        }
    } finally {
        wasm.bollinger_bands_free(ptr, len);
    }
});
