import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('price_density_market_noise_js output contract', () => {
    const high = new Float64Array(testData.high.slice(0, 720));
    const low = new Float64Array(testData.low.slice(0, 720));
    const close = new Float64Array(testData.close.slice(0, 720));

    const result = wasm.price_density_market_noise_js(high, low, close, 14, 200);

    assert.strictEqual(result.price_density.length, close.length);
    assert.strictEqual(result.price_density_percent.length, close.length);
    assert(Array.from(result.price_density.slice(0, 13)).every(v => isNaN(v)));
    assert(Array.from(result.price_density_percent.slice(0, 212)).every(v => isNaN(v)));
    assert(Array.from(result.price_density.slice(13)).some(v => !isNaN(v)));
    assert(Array.from(result.price_density_percent.slice(212)).some(v => !isNaN(v)));
});

test('price_density_market_noise_into_host pointer path matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 320));
    const low = new Float64Array(testData.low.slice(0, 320));
    const close = new Float64Array(testData.close.slice(0, 320));
    const safe = wasm.price_density_market_noise_js(high, low, close, 12, 40);
    const outPtr = wasm.price_density_market_noise_alloc(close.length);

    try {
        wasm.price_density_market_noise_into_host(high, low, close, outPtr, 12, 40);
        const flat = wasm.read_f64_array(outPtr, close.length * 2);
        const priceDensity = flat.slice(0, close.length);
        const priceDensityPercent = flat.slice(close.length);
        assertArrayClose(priceDensity, safe.price_density, 1e-10, 'price_density pointer-path mismatch');
        assertArrayClose(priceDensityPercent, safe.price_density_percent, 1e-10, 'price_density_percent pointer-path mismatch');
    } finally {
        wasm.price_density_market_noise_free(outPtr, close.length);
    }
});

test('price_density_market_noise_batch_js single parameter set matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 320));
    const low = new Float64Array(testData.low.slice(0, 320));
    const close = new Float64Array(testData.close.slice(0, 320));
    const batch = wasm.price_density_market_noise_batch_js(high, low, close, {
        length_range: [12, 12, 0],
        eval_period_range: [40, 40, 0],
    });
    const single = wasm.price_density_market_noise_js(high, low, close, 12, 40);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.price_density.length, close.length);
    assert.strictEqual(batch.price_density_percent.length, close.length);
    assert.strictEqual(batch.combos[0].length, 12);
    assert.strictEqual(batch.combos[0].eval_period, 40);
    assertArrayClose(batch.price_density, single.price_density, 1e-10, 'price_density batch mismatch');
    assertArrayClose(batch.price_density_percent, single.price_density_percent, 1e-10, 'price_density_percent batch mismatch');
});
