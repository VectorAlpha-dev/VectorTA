/**
 * WASM binding tests for Bollinger Bands Width indicator.
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

test('BBW partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.bollinger_bands_width_js(
        close,
        22,     
        2.2,    
        2.0,    
        "ema",  
        null    
    );
    
    assert.strictEqual(result.length, close.length);
});

test('BBW default params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.bollinger_bands_width_js(
        close,
        20,     
        2.0,    
        2.0,    
        null,   
        null    
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    assertAllNaN(result.slice(0, 19), "Expected NaN in warmup period");
    
    
    if (result.length > 240) {
        assertNoNaN(result.slice(240), "Expected no NaN after sufficient warmup");
    }
});

test('BBW accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.bollinger_bands_width || {};
    
    const result = wasm.bollinger_bands_width_js(
        close,
        expected.default_params?.period || 20,
        expected.default_params?.devup || 2.0,
        expected.default_params?.devdn || 2.0,
        expected.default_params?.matype || "sma",
        expected.default_params?.devtype || 0
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    if (expected.last_5_values) {
        const last5 = result.slice(-5);
        assertArrayClose(
            last5,
            expected.last_5_values,
            1e-8,
            "BBW last 5 values mismatch"
        );
    }
    
    
    await compareWithRust('bollinger_bands_width', result, 'close', {
        period: expected.default_params?.period || 20,
        devup: expected.default_params?.devup || 2.0,
        devdn: expected.default_params?.devdn || 2.0,
        matype: expected.default_params?.matype || "sma",
        devtype: expected.default_params?.devtype || 0
    });
});

test('BBW zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.bollinger_bands_width_js(inputData, 0, 2.0, 2.0, null, null);
    }, /Invalid period|period/);
});

test('BBW period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.bollinger_bands_width_js(dataSmall, 10, 2.0, 2.0, null, null);
    }, /Invalid period|period/);
});

test('BBW very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.bollinger_bands_width_js(singlePoint, 20, 2.0, 2.0, null, null);
    }, /Invalid period|Not enough valid data/);
});

test('BBW empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.bollinger_bands_width_js(empty, 20, 2.0, 2.0, null, null);
    }, /Empty data/);
});

test('BBW all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.bollinger_bands_width_js(allNaN, 20, 2.0, 2.0, null, null);
    }, /All values are NaN/);
});

test('BBW different MA types', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    const matypes = ["sma", "ema", "wma", "dema", "tema"];
    
    for (const matype of matypes) {
        const result = wasm.bollinger_bands_width_js(
            close,
            14,
            2.0,
            2.0,
            matype,
            0
        );
        
        assert.strictEqual(result.length, 100);
        
        for (let i = 0; i < 13; i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} for ${matype}`);
        }
    }
});

test('BBW different deviation types', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    const devtypes = [0, 1, 2]; 
    
    for (const devtype of devtypes) {
        const result = wasm.bollinger_bands_width_js(
            close,
            14,
            2.0,
            2.0,
            "sma",
            devtype
        );
        
        assert.strictEqual(result.length, 100);
    }
});

test('BBW NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.bollinger_bands_width_js(close, 20, 2.0, 2.0, null, null);
    assert.strictEqual(result.length, close.length);
    
    
    assertAllNaN(result.slice(0, 19), "Expected NaN in warmup period");
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('BBW batch single parameter set', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.bollinger_bands_width_batch_js(
        close,
        20, 20, 0,      
        2.0, 2.0, 0,    
        2.0, 2.0, 0     
    );
    
    
    const singleResult = wasm.bollinger_bands_width_js(close, 20, 2.0, 2.0, null, null);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('BBW batch multiple parameters', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const batchResult = wasm.bollinger_bands_width_batch_js(
        close,
        10, 30, 10,     
        1.5, 2.5, 0.5,  
        2.0, 2.0, 0     
    );
    
    
    assert.strictEqual(batchResult.length, 9 * 100);
    
    
    const periods = [10, 20, 30];
    const devups = [1.5, 2.0, 2.5];
    let rowIdx = 0;
    
    for (const period of periods) {
        for (const devup of devups) {
            const rowStart = rowIdx * 100;
            const rowEnd = rowStart + 100;
            const rowData = batchResult.slice(rowStart, rowEnd);
            
            const singleResult = wasm.bollinger_bands_width_js(
                close, period, devup, 2.0, null, null
            );
            
            assertArrayClose(
                rowData, 
                singleResult, 
                1e-10, 
                `Period ${period}, devup ${devup} mismatch`
            );
            
            rowIdx++;
        }
    }
});

test('BBW batch metadata', () => {
    
    const metadata = wasm.bollinger_bands_width_batch_metadata_js(
        10, 20, 10,     
        1.5, 2.0, 0.5,  
        2.0, 3.0, 1.0   
    );
    
    
    
    assert.strictEqual(metadata.length, 8 * 3);
    
    
    assert.strictEqual(metadata[0], 10);   
    assert.strictEqual(metadata[1], 1.5);  
    assert.strictEqual(metadata[2], 2.0);  
    
    
    assert.strictEqual(metadata[21], 20);  
    assert.strictEqual(metadata[22], 2.0); 
    assert.strictEqual(metadata[23], 3.0); 
});

test('BBW batch full parameter sweep', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.bollinger_bands_width_batch_js(
        close,
        10, 15, 5,      
        2.0, 2.5, 0.5,  
        1.5, 1.5, 0     
    );
    
    const metadata = wasm.bollinger_bands_width_batch_metadata_js(
        10, 15, 5,
        2.0, 2.5, 0.5,
        1.5, 1.5, 0
    );
    
    
    const numCombos = metadata.length / 3;
    assert.strictEqual(numCombos, 4);
    assert.strictEqual(batchResult.length, 4 * 50);
    
    
    for (let combo = 0; combo < numCombos; combo++) {
        const period = metadata[combo * 3];
        const devup = metadata[combo * 3 + 1];
        const devdn = metadata[combo * 3 + 2];
        
        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);
        
        
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        
        for (let i = period - 1; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('BBW batch edge cases', () => {
    
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    
    const singleBatch = wasm.bollinger_bands_width_batch_js(
        close,
        5, 5, 1,
        2.0, 2.0, 0.1,
        2.0, 2.0, 1.0
    );
    
    assert.strictEqual(singleBatch.length, 10);
    
    
    const largeBatch = wasm.bollinger_bands_width_batch_js(
        close,
        5, 7, 10, 
        2.0, 2.0, 0,
        2.0, 2.0, 0
    );
    
    
    assert.strictEqual(largeBatch.length, 10);
    
    
    assert.throws(() => {
        wasm.bollinger_bands_width_batch_js(
            new Float64Array([]),
            10, 10, 0,
            2.0, 2.0, 0,
            2.0, 2.0, 0
        );
    }, /All values are NaN/);
});


test('BBW batch - new ergonomic API with single parameter', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.bollinger_bands_width_batch(close, {
        period_range: [20, 20, 0],
        devup_range: [2.0, 2.0, 0],
        devdn_range: [2.0, 2.0, 0],
        matype: "sma",
        devtype: 0
    });
    
    
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');
    
    
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.values.length, close.length);
    
    
    const combo = result.combos[0];
    assert.strictEqual(combo.period, 20);
    assert.strictEqual(combo.devup, 2.0);
    assert.strictEqual(combo.devdn, 2.0);
    assert.strictEqual(combo.matype, "sma");
    assert.strictEqual(combo.devtype, 0);
    
    
    const oldResult = wasm.bollinger_bands_width_js(close, 20, 2.0, 2.0, "sma", 0);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; 
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('BBW batch - new API with multiple parameters', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.bollinger_bands_width_batch(close, {
        period_range: [10, 20, 10],     
        devup_range: [1.5, 2.0, 0.5],   
        devdn_range: [2.0, 2.0, 0],     
        matype: "ema",
        devtype: 1
    });
    
    
    assert.strictEqual(result.rows, 4);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 4);
    assert.strictEqual(result.values.length, 200);
    
    
    const expectedCombos = [
        { period: 10, devup: 1.5, devdn: 2.0 },
        { period: 10, devup: 2.0, devdn: 2.0 },
        { period: 20, devup: 1.5, devdn: 2.0 },
        { period: 20, devup: 2.0, devdn: 2.0 }
    ];
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedCombos[i].period);
        assert.strictEqual(result.combos[i].devup, expectedCombos[i].devup);
        assert.strictEqual(result.combos[i].devdn, expectedCombos[i].devdn);
        assert.strictEqual(result.combos[i].matype, "ema");
        assert.strictEqual(result.combos[i].devtype, 1);
    }
});

test('BBW batch - new API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    
    assert.throws(() => {
        wasm.bollinger_bands_width_batch(close, {
            period_range: [10, 10], 
            devup_range: [2.0, 2.0, 0],
            devdn_range: [2.0, 2.0, 0]
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.bollinger_bands_width_batch(close, {
            period_range: [10, 10, 0],
            devup_range: [2.0, 2.0, 0]
            
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.bollinger_bands_width_batch(close, {
            period_range: "invalid",
            devup_range: [2.0, 2.0, 0],
            devdn_range: [2.0, 2.0, 0]
        });
    }, /Invalid config/);
});

test.after(() => {
    console.log('Bollinger Bands Width WASM tests completed');
});