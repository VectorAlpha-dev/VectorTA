/**
 * WASM binding tests for MOD_GOD_MODE indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
import { test } from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { dirname } from 'path';
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

const __dirname = dirname(fileURLToPath(import.meta.url));

let wasm;
let testData;

test.before(async () => {
    
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const wasmUrl = new URL(`file:///${wasmPath.replace(/\\/g, '/')}`).href;
        wasm = await import(wasmUrl);
        
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('MOD_GOD_MODE basic test', () => {
    
    const length = 50;
    const high = new Float64Array(length).fill(10.0);
    const low = new Float64Array(length).fill(9.0);
    const close = new Float64Array(length);
    for (let i = 0; i < length; i++) {
        close[i] = 10.0 + (i % 5) * 0.2;
    }
    
    
    const result = wasm.mod_god_mode(
        high, 
        low, 
        close,
        undefined,  
        17,         
        6,          
        4,          
        "tradition_mg",  
        false       
    );
    
    
    assert(result, 'Should return a result');
    assert(result.wavetrend, 'Should have wavetrend');
    assert(result.signal, 'Should have signal');
    assert(result.histogram, 'Should have histogram');
    
    
    assert.equal(result.wavetrend.length, length, 'Wavetrend length should match input');
    assert.equal(result.signal.length, length, 'Signal length should match input');
    assert.equal(result.histogram.length, length, 'Histogram length should match input');
    
    
    let nonNanCount = 0;
    for (let i = 0; i < result.wavetrend.length; i++) {
        if (!isNaN(result.wavetrend[i])) {
            nonNanCount++;
        }
    }
    assert(nonNanCount > 0, 'Should have some non-NaN values in wavetrend');
});

test('MOD_GOD_MODE accuracy (matches Rust refs, tol<=4.0)', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);

    const result = wasm.mod_god_mode(
        high, low, close, volume,
        17, 
        6,  
        4,  
        'tradition_mg',
        true 
    );

    const wt = Array.from(result.wavetrend);
    const nonNan = wt.filter(v => !isNaN(v));
    assert(nonNan.length >= 5, 'Not enough non-NaN values for accuracy check');

    
    const expectedLastFive = [
        61.66219598,
        55.92955776,
        34.70836488,
        39.48824969,
        15.74958884,
    ];

    const actualLastFive = nonNan.slice(-5);
    for (let i = 0; i < 5; i++) {
        const diff = Math.abs(actualLastFive[i] - expectedLastFive[i]);
        assert(
            diff < 4.0,
            `Value ${i} mismatch: expected ${expectedLastFive[i].toFixed(8)}, got ${actualLastFive[i].toFixed(8)}, diff ${diff.toFixed(8)}`
        );
    }
});

test('MOD_GOD_MODE modes test', () => {
    const length = 50;
    const high = new Float64Array(length).fill(10.0);
    const low = new Float64Array(length).fill(9.0);
    const close = new Float64Array(length);
    for (let i = 0; i < length; i++) {
        close[i] = 10.0 + (i % 5) * 0.1;
    }
    
    const modes = ['godmode', 'tradition', 'godmode_mg', 'tradition_mg'];
    
    for (const mode of modes) {
        const result = wasm.mod_god_mode(
            high, low, close, undefined,
            17, 6, 4, mode, false
        );
        
        assert(result, `Mode ${mode} should return a result`);
        assert.equal(result.wavetrend.length, length, `Mode ${mode} wavetrend length`);
    }
});

test.after(() => {
    console.log('MOD_GOD_MODE WASM tests completed');
});
