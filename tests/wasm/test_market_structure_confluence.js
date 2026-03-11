import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, loadTestData } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

const OUTPUT_KEYS = [
    'basis',
    'upper_band',
    'lower_band',
    'structure_direction',
    'bullish_arrow',
    'bearish_arrow',
    'bullish_change',
    'bearish_change',
    'hh',
    'lh',
    'hl',
    'll',
    'bullish_bos',
    'bullish_choch',
    'bearish_bos',
    'bearish_choch',
];

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    const mod = await import(importPath);
    wasm = mod.default ?? mod;
    testData = loadTestData();
});

test('market_structure_confluence_js returns expected outputs', () => {
    const high = new Float64Array(testData.high.slice(0, 420));
    const low = new Float64Array(testData.low.slice(0, 420));
    const close = new Float64Array(testData.close.slice(0, 420));
    const result = wasm.market_structure_confluence_js(
        high,
        low,
        close,
        10,
        'Candle Close',
        100,
        14,
        21,
        2.0,
    );
    assert.deepStrictEqual(Object.keys(result), OUTPUT_KEYS);
    for (const key of OUTPUT_KEYS) {
        assert.strictEqual(result[key].length, close.length);
    }
    assert.ok(result.basis.some(Number.isFinite));
});

test('market_structure_confluence_batch first row matches single', () => {
    const high = new Float64Array(testData.high.slice(0, 420));
    const low = new Float64Array(testData.low.slice(0, 420));
    const close = new Float64Array(testData.close.slice(0, 420));
    const single = wasm.market_structure_confluence_js(
        high,
        low,
        close,
        10,
        'Candle Close',
        100,
        14,
        21,
        2.0,
    );
    const batch = wasm.market_structure_confluence_batch(high, low, close, {
        swing_size_range: [10, 12, 2],
        bos_confirmation_options: ['Candle Close'],
        basis_length_range: [100, 100, 0],
        atr_length_range: [14, 14, 0],
        atr_smooth_range: [21, 21, 0],
        vol_mult_range: [2.0, 2.0, 0.0],
    });
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.swing_sizes, [10, 12]);
    for (const key of OUTPUT_KEYS) {
        assertArrayClose(batch[key].slice(0, close.length), single[key], 1e-12, `${key} mismatch`);
    }
});

test('market_structure_confluence_js rejects invalid params', () => {
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    assert.throws(() => {
        wasm.market_structure_confluence_js(high, low, close, 1, 'Candle Close', 100, 14, 21, 2.0);
    }, /invalid swing_size/i);
    assert.throws(() => {
        wasm.market_structure_confluence_js(high, low, close, 10, 'bad', 100, 14, 21, 2.0);
    }, /invalid bos_confirmation/i);
});
