import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

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

test('ict_propulsion_block output contract', () => {
    const open = new Float64Array(testData.open.slice(0, 320));
    const high = new Float64Array(testData.high.slice(0, 320));
    const low = new Float64Array(testData.low.slice(0, 320));
    const close = new Float64Array(testData.close.slice(0, 320));

    const result = wasm.ict_propulsion_block(open, high, low, close, 3, 'close');
    assert.strictEqual(result.bullish_high.length, close.length);
    assert.strictEqual(result.bearish_new.length, close.length);
    assert(result.bullish_kind.some(v => Number.isFinite(v) && (v === 1 || v === 2))
        || result.bearish_kind.some(v => Number.isFinite(v) && (v === 1 || v === 2)));
});

test('ict_propulsion_block_into_host pointer path matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 220));
    const high = new Float64Array(testData.high.slice(0, 220));
    const low = new Float64Array(testData.low.slice(0, 220));
    const close = new Float64Array(testData.close.slice(0, 220));
    const safe = wasm.ict_propulsion_block(open, high, low, close, 3, 'close');
    const outPtr = wasm.ict_propulsion_block_alloc(close.length);

    try {
        wasm.ict_propulsion_block_into_host(open, high, low, close, outPtr, 3, 'close');
        const out = wasm.read_f64_array(outPtr, 12 * close.length);
        const chunks = [];
        for (let i = 0; i < 12; i++) {
            chunks.push(out.slice(i * close.length, (i + 1) * close.length));
        }
        const keys = [
            'bullish_high', 'bullish_low', 'bullish_kind', 'bullish_active', 'bullish_mitigated', 'bullish_new',
            'bearish_high', 'bearish_low', 'bearish_kind', 'bearish_active', 'bearish_mitigated', 'bearish_new',
        ];
        for (let i = 0; i < keys.length; i++) {
            assertArrayClose(chunks[i], safe[keys[i]], 1e-10, `${keys[i]} mismatch`);
        }
    } finally {
        wasm.ict_propulsion_block_free(outPtr, close.length);
    }
});

test('ict_propulsion_block_batch single parameter set matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 200));
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.ict_propulsion_block_batch(open, high, low, close, {
        swing_length_range: [3, 3, 0],
        mitigation_price_toggle: [true, false],
    });
    const single = wasm.ict_propulsion_block(open, high, low, close, 3, 'close');

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].swing_length, 3);
    assert.strictEqual(batch.combos[0].mitigation_price, 'close');
    assertArrayClose(batch.bullish_high, single.bullish_high, 1e-10, 'bullish_high mismatch');
    assertArrayClose(batch.bearish_new, single.bearish_new, 1e-10, 'bearish_new mismatch');
});

test('ict_propulsion_block_batch metadata', () => {
    const open = new Float64Array(testData.open.slice(0, 180));
    const high = new Float64Array(testData.high.slice(0, 180));
    const low = new Float64Array(testData.low.slice(0, 180));
    const close = new Float64Array(testData.close.slice(0, 180));
    const batch = wasm.ict_propulsion_block_batch(open, high, low, close, {
        swing_length_range: [3, 4, 1],
        mitigation_price_toggle: [true, true],
    });

    assert.strictEqual(batch.rows, 4);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.combos.map(combo => combo.swing_length), [3, 3, 4, 4]);
    assert.deepStrictEqual(batch.combos.map(combo => combo.mitigation_price), ['close', 'wick', 'close', 'wick']);
});

test('ict_propulsion_block_batch_into pointer path matches safe batch API', () => {
    const open = new Float64Array(testData.open.slice(0, 180));
    const high = new Float64Array(testData.high.slice(0, 180));
    const low = new Float64Array(testData.low.slice(0, 180));
    const close = new Float64Array(testData.close.slice(0, 180));
    const batch = wasm.ict_propulsion_block_batch(open, high, low, close, {
        swing_length_range: [3, 4, 1],
        mitigation_price_toggle: [true, true],
    });

    const rows = batch.rows;
    const total = rows * close.length;
    const openPtr = wasm.ict_propulsion_block_alloc(close.length);
    const highPtr = wasm.ict_propulsion_block_alloc(close.length);
    const lowPtr = wasm.ict_propulsion_block_alloc(close.length);
    const closePtr = wasm.ict_propulsion_block_alloc(close.length);
    const ptrs = Array.from({ length: 12 }, () => wasm.ict_propulsion_block_alloc(total));

    try {
        const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        if (!memory || !memory.buffer) {
            console.warn('Skipping ict_propulsion_block_batch_into test: wasm memory accessor unavailable');
            return;
        }
        new Float64Array(memory.buffer, openPtr, close.length).set(open);
        new Float64Array(memory.buffer, highPtr, close.length).set(high);
        new Float64Array(memory.buffer, lowPtr, close.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);

        const returnedRows = wasm.ict_propulsion_block_batch_into(
            openPtr, highPtr, lowPtr, closePtr,
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5],
            ptrs[6], ptrs[7], ptrs[8], ptrs[9], ptrs[10], ptrs[11],
            close.length,
            3, 4, 1,
            true, true,
        );

        assert.strictEqual(returnedRows, rows);
        const keys = [
            'bullish_high', 'bullish_low', 'bullish_kind', 'bullish_active', 'bullish_mitigated', 'bullish_new',
            'bearish_high', 'bearish_low', 'bearish_kind', 'bearish_active', 'bearish_mitigated', 'bearish_new',
        ];
        for (let i = 0; i < keys.length; i++) {
            assertArrayClose(wasm.read_f64_array(ptrs[i], total), batch[keys[i]], 1e-10, `${keys[i]} batch_into mismatch`);
        }
    } finally {
        wasm.ict_propulsion_block_free(openPtr, close.length);
        wasm.ict_propulsion_block_free(highPtr, close.length);
        wasm.ict_propulsion_block_free(lowPtr, close.length);
        wasm.ict_propulsion_block_free(closePtr, close.length);
        for (const ptr of ptrs) {
            wasm.ict_propulsion_block_free(ptr, total);
        }
    }
});
