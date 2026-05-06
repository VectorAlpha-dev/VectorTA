import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
});

function wasmMemory() {
    return wasm.__wasm?.memory ?? wasm.wasm_memory();
}

function writeF64(ptr, values) {
    new Float64Array(wasmMemory().buffer, ptr, values.length).set(values);
}

function readF64(ptr, len) {
    return Array.from(new Float64Array(wasmMemory().buffer, ptr, len));
}

function assertThrowsMessage(fn, pattern) {
    let thrown;
    try {
        fn();
    } catch (error) {
        thrown = error;
    }

    assert(thrown, 'Expected function to throw');
    const message = String(thrown?.message ?? thrown);
    assert.match(message, pattern);
}

test('global f64 allocation helpers expose writable and releasable memory', () => {
    assert.strictEqual(typeof wasm.allocate_f64_array, 'function');
    assert.strictEqual(typeof wasm.deallocate_f64_array, 'function');
    assert.strictEqual(typeof wasm.allocate_f64_matrix, 'function');
    assert.strictEqual(typeof wasm.deallocate_f64_matrix, 'function');
    assert.strictEqual(typeof wasm.copy_f64_array, 'function');
    assert.strictEqual(typeof wasm.read_f64_array, 'function');
    assert.strictEqual(typeof wasm.write_f64_array, 'function');
    assert.strictEqual(typeof wasm.read_f64_matrix, 'function');

    const ptr = wasm.allocate_f64_array(4);
    const copyPtr = wasm.copy_f64_array(new Float64Array([5.5, 6.5, 7.5]));
    const matrixPtr = wasm.allocate_f64_matrix(2, 3);

    try {
        assert.notStrictEqual(ptr, 0);
        assert.notStrictEqual(copyPtr, 0);
        assert.notStrictEqual(matrixPtr, 0);

        wasm.write_f64_array(ptr, new Float64Array([1.25, 2.5, 3.75, 5.0]));
        assert.deepStrictEqual(Array.from(wasm.read_f64_array(ptr, 4)), [1.25, 2.5, 3.75, 5.0]);
        assert.deepStrictEqual(Array.from(wasm.read_f64_array(copyPtr, 3)), [5.5, 6.5, 7.5]);

        const view = wasm.view_f64_array(ptr, 4);
        assert.deepStrictEqual(Array.from(view), [1.25, 2.5, 3.75, 5.0]);
        view[1] = 12.5;
        assert.deepStrictEqual(Array.from(wasm.read_f64_array(ptr, 4)), [1.25, 12.5, 3.75, 5.0]);

        const copied = new Float64Array(4);
        wasm.read_f64_array_into(ptr, 4, copied);
        assert.deepStrictEqual(Array.from(copied), [1.25, 12.5, 3.75, 5.0]);

        assert.strictEqual(wasm.view_f64_array(0, 0).length, 0);
        assert.doesNotThrow(() => wasm.read_f64_array_into(0, 0, new Float64Array(0)));
        assertThrowsMessage(() => wasm.view_f64_array(0, 1), /null pointer/i);
        assertThrowsMessage(() => wasm.read_f64_array_into(0, 1, copied), /null pointer/i);
        assertThrowsMessage(
            () => wasm.read_f64_array_into(ptr, 4, new Float64Array(3)),
            /output is too small/i,
        );

        wasm.write_f64_array(matrixPtr, new Float64Array([1, 2, 3, 4, 5, 6]));
        const matrix = wasm.read_f64_matrix(matrixPtr, 2, 3);
        assert.deepStrictEqual(Array.from(matrix[0]), [1, 2, 3]);
        assert.deepStrictEqual(Array.from(matrix[1]), [4, 5, 6]);

        const overflowPtr = wasm.allocate_f64_matrix(0xffffffff, 0xffffffff);
        assert.strictEqual(overflowPtr, 0);

        assert.doesNotThrow(() => wasm.deallocate_f64_array(0));
        assert.doesNotThrow(() => wasm.deallocate_f64_matrix(0));
    } finally {
        wasm.deallocate_f64_array(ptr);
        wasm.deallocate_f64_array(copyPtr);
        wasm.deallocate_f64_matrix(matrixPtr);
    }
});

test('global f64 allocation helpers do not grow memory without bound after frees', () => {
    const before = wasmMemory().buffer.byteLength;
    for (let cycle = 0; cycle < 64; cycle++) {
        const ptrs = [];
        try {
            for (let i = 0; i < 32; i++) {
                const ptr = wasm.allocate_f64_array(1024);
                assert.notStrictEqual(ptr, 0);
                ptrs.push(ptr);
            }
        } finally {
            for (const ptr of ptrs) {
                wasm.deallocate_f64_array(ptr);
            }
        }
    }
    const after = wasmMemory().buffer.byteLength;
    assert(after <= before + 16 * 1024 * 1024);
});

test('reusable flat-output WASM wrappers fill caller Float64Array', () => {
    const data = new Float64Array(64);
    for (let i = 0; i < data.length; i++) {
        data[i] = 100 + i * 0.5 + Math.sin(i / 4);
    }

    const out = new Float64Array(data.length);
    const written = wasm.sma_output_into_js(data, 5, out);
    const expected = wasm.sma(data, 5);

    assert.strictEqual(written, data.length);
    assertArrayClose(Array.from(out), Array.from(expected), 1e-12);
    assertThrowsMessage(
        () => wasm.sma_output_into_js(data, 5, new Float64Array(data.length - 1)),
        /output is too small/i,
    );
});

test('RSI reusable output wrapper matches safe API', () => {
    const data = new Float64Array(64);
    for (let i = 0; i < data.length; i++) {
        data[i] = 80 + i * 0.35 + Math.cos(i / 5);
    }

    const out = new Float64Array(data.length);
    const written = wasm.rsi_output_into_js(data, 14, out);
    const expected = wasm.rsi_js(data, 14);

    assert.strictEqual(written, data.length);
    assertArrayClose(Array.from(out), Array.from(expected), 1e-12);
    assertThrowsMessage(
        () => wasm.rsi_output_into_js(data, 14, new Float64Array(data.length - 1)),
        /output is too small/i,
    );
});

test('reusable structured-output WASM wrappers fill flat caller Float64Array', () => {
    const data = new Float64Array(72);
    for (let i = 0; i < data.length; i++) {
        data[i] = 90 + i * 0.8 + Math.cos(i / 6);
    }

    const expected = wasm.macd_js(data, 12, 26, 9, 'ema').values;
    const out = new Float64Array(expected.length);
    const written = wasm.macd_output_into_js(data, 12, 26, 9, 'ema', out);

    assert.strictEqual(written, expected.length);
    assertArrayClose(Array.from(out), Array.from(expected), 1e-12);
    assertThrowsMessage(
        () => wasm.macd_output_into_js(data, 12, 26, 9, 'ema', new Float64Array(expected.length - 1)),
        /output is too small/i,
    );
});

test('reusable named-object WASM wrappers fill caller Float64Array fields', () => {
    const len = 80;
    const high = new Float64Array(len);
    const low = new Float64Array(len);
    const close = new Float64Array(len);
    const volume = new Float64Array(len);
    for (let i = 0; i < len; i++) {
        close[i] = 100 + i * 0.4 + Math.sin(i / 5);
        high[i] = close[i] + 1.5;
        low[i] = close[i] - 1.5;
        volume[i] = 1000 + i * 3;
    }

    const expected = wasm.demand_index_js(high, low, close, volume, 13, 5, 5, 'ema');
    const out = {
        demand_index: new Float64Array(len),
        signal: new Float64Array(len),
    };
    const written = wasm.demand_index_output_into_js(high, low, close, volume, 13, 5, 5, 'ema', out);

    assert.strictEqual(written, len * 2);
    assertArrayClose(Array.from(out.demand_index), Array.from(expected.demand_index), 1e-12);
    assertArrayClose(Array.from(out.signal), Array.from(expected.signal), 1e-12);
    assertThrowsMessage(
        () => wasm.demand_index_output_into_js(high, low, close, volume, 13, 5, 5, 'ema', {
            demand_index: new Float64Array(len),
        }),
        /missing Float64Array output for field signal/i,
    );
});

test('reusable selected batch-output WASM wrappers copy requested fields only', () => {
    const data = new Float64Array(72);
    for (let i = 0; i < data.length; i++) {
        data[i] = 50 + i * 0.25 + Math.cos(i / 7);
    }
    const config = { period_range: [5, 7, 1] };
    const expected = wasm.rsi_batch(data, config);
    const out = {
        values: new Float64Array(expected.values.length),
    };

    const written = wasm.rsi_batch_unified_output_into_js(data, config, out);

    assert.strictEqual(written, expected.values.length);
    assertArrayClose(Array.from(out.values), Array.from(expected.values), 1e-12);
    assert.doesNotThrow(() => wasm.rsi_batch_unified_output_into_js(data, config, {
        values: new Float64Array(expected.values.length),
    }));
    assertThrowsMessage(
        () => wasm.rsi_batch_unified_output_into_js(data, config, {
            values: new Float64Array(expected.values.length - 1),
        }),
        /output field values is too small/i,
    );
});

test('single-output pointer API rejects null pointers and matches safe RVI output', () => {
    const data = new Float64Array([
        10, 10.5, 11, 10.75, 11.5, 12, 11.75, 12.25, 12.75, 13,
        12.5, 13.25, 13.75, 14, 13.5, 14.25, 14.75, 15, 14.5, 15.25,
        15.75, 16, 15.5, 16.25, 16.75, 17, 16.5, 17.25, 17.75, 18,
    ]);
    const len = data.length;
    const inPtr = wasm.rvi_alloc(len);
    const outPtr = wasm.rvi_alloc(len);

    try {
        writeF64(inPtr, data);

        assertThrowsMessage(
            () => wasm.rvi_into(0, outPtr, len, 5, 5, 1, 0),
            /null pointer/i,
        );
        assertThrowsMessage(
            () => wasm.rvi_into(inPtr, 0, len, 5, 5, 1, 0),
            /null pointer/i,
        );
        assertThrowsMessage(
            () => wasm.rvi_into(inPtr, outPtr, 0, 5, 5, 1, 0),
            /len cannot be 0/i,
        );

        wasm.rvi_into(inPtr, outPtr, len, 5, 5, 1, 0);
        assertArrayClose(readF64(outPtr, len), Array.from(wasm.rvi_js(data, 5, 5, 1, 0)), 1e-10);
    } finally {
        wasm.rvi_free(inPtr, len);
        wasm.rvi_free(outPtr, len);
    }
});

test('batch pointer API enforces null, rows, and cols contracts for ADX', () => {
    const len = 48;
    const high = new Float64Array(len);
    const low = new Float64Array(len);
    const close = new Float64Array(len);
    for (let i = 0; i < len; i++) {
        close[i] = 100 + i + Math.sin(i / 3);
        high[i] = close[i] + 2 + (i % 3) * 0.25;
        low[i] = close[i] - 2 - (i % 2) * 0.25;
    }

    const rows = 3;
    const highPtr = wasm.adx_alloc(len);
    const lowPtr = wasm.adx_alloc(len);
    const closePtr = wasm.adx_alloc(len);
    const outPtr = wasm.adx_alloc(rows * len);

    try {
        writeF64(highPtr, high);
        writeF64(lowPtr, low);
        writeF64(closePtr, close);

        assertThrowsMessage(
            () => wasm.adx_batch_into(0, lowPtr, closePtr, len, outPtr, rows, len, 5, 7, 1),
            /null pointer/i,
        );
        assertThrowsMessage(
            () => wasm.adx_batch_into(highPtr, lowPtr, closePtr, len, outPtr, rows, len + 1, 5, 7, 1),
            /cols must equal len/i,
        );
        assertThrowsMessage(
            () => wasm.adx_batch_into(highPtr, lowPtr, closePtr, len, outPtr, rows + 1, len, 5, 7, 1),
            /rows mismatch/i,
        );

        const actualRows = wasm.adx_batch_into(highPtr, lowPtr, closePtr, len, outPtr, rows, len, 5, 7, 1);
        assert.strictEqual(actualRows, rows);
        assertArrayClose(readF64(outPtr, rows * len), Array.from(wasm.adx_batch_js(high, low, close, 5, 7, 1)), 1e-10);
    } finally {
        wasm.adx_free(highPtr, len);
        wasm.adx_free(lowPtr, len);
        wasm.adx_free(closePtr, len);
        wasm.adx_free(outPtr, rows * len);
    }
});

test('multi-output pointer API rejects null pointers and matches safe MACD output', () => {
    const len = 64;
    const data = new Float64Array(len);
    for (let i = 0; i < len; i++) {
        data[i] = 100 + i * 0.75 + Math.cos(i / 4);
    }

    const inPtr = wasm.macd_alloc(len);
    const macdPtr = wasm.macd_alloc(len);
    const signalPtr = wasm.macd_alloc(len);
    const histPtr = wasm.macd_alloc(len);

    try {
        writeF64(inPtr, data);

        assertThrowsMessage(
            () => wasm.macd_into(0, macdPtr, signalPtr, histPtr, len, 12, 26, 9, 'ema'),
            /null pointer/i,
        );
        assertThrowsMessage(
            () => wasm.macd_into(inPtr, 0, signalPtr, histPtr, len, 12, 26, 9, 'ema'),
            /null pointer/i,
        );
        assertThrowsMessage(
            () => wasm.macd_into(inPtr, macdPtr, 0, histPtr, len, 12, 26, 9, 'ema'),
            /null pointer/i,
        );
        assertThrowsMessage(
            () => wasm.macd_into(inPtr, macdPtr, signalPtr, 0, len, 12, 26, 9, 'ema'),
            /null pointer/i,
        );

        wasm.macd_into(inPtr, macdPtr, signalPtr, histPtr, len, 12, 26, 9, 'ema');
        const expected = wasm.macd_js(data, 12, 26, 9, 'ema').values;
        assertArrayClose(readF64(macdPtr, len), Array.from(expected.slice(0, len)), 1e-10);
        assertArrayClose(readF64(signalPtr, len), Array.from(expected.slice(len, len * 2)), 1e-10);
        assertArrayClose(readF64(histPtr, len), Array.from(expected.slice(len * 2)), 1e-10);

        assert.doesNotThrow(() => wasm.macd_free(0, len));
    } finally {
        wasm.macd_free(inPtr, len);
        wasm.macd_free(macdPtr, len);
        wasm.macd_free(signalPtr, len);
        wasm.macd_free(histPtr, len);
    }
});
