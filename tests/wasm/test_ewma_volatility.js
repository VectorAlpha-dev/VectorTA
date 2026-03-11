import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function geometricSeries(length, start, ratio) {
    const out = new Float64Array(length);
    let value = start;
    for (let i = 0; i < length; i++) {
        out[i] = value;
        value *= ratio;
    }
    return out;
}

function pinePeriod(lambdaValue) {
    return Math.max(1, Math.round(2 / (1 - lambdaValue) - 1));
}

function manualEwmaVolatility(data, lambdaValue) {
    const period = pinePeriod(lambdaValue);
    const alpha = 2 / (period + 1);
    const out = new Float64Array(data.length);
    out.fill(NaN);
    const sqReturns = [];
    let seeded = false;
    let ema = NaN;

    for (let i = 1; i < data.length; i++) {
        const prev = data[i - 1];
        const curr = data[i];
        if (Number.isFinite(prev) && Number.isFinite(curr) && prev > 0 && curr > 0) {
            const ret = Math.log(curr / prev);
            const sq = ret * ret;
            sqReturns.push([i, sq]);
            if (!seeded && sqReturns.length === period) {
                seeded = true;
                let sum = 0;
                for (let j = 0; j < sqReturns.length; j++) {
                    sum += sqReturns[j][1];
                }
                ema = sum / period;
                out[i] = Math.sqrt(Math.max(ema, 0)) * 100;
            } else if (seeded) {
                ema = (1 - alpha) * ema + alpha * sq;
                out[i] = Math.sqrt(Math.max(ema, 0)) * 100;
            }
        } else if (seeded) {
            out[i] = out[i - 1];
        }
    }

    if (seeded) {
        for (let i = 1; i < out.length; i++) {
            if (Number.isNaN(out[i]) && !Number.isNaN(out[i - 1])) {
                out[i] = out[i - 1];
            }
        }
    }

    return out;
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
});

test('ewma_volatility_js constant return level', () => {
    const data = geometricSeries(128, 100, 1.01);
    const result = wasm.ewma_volatility_js(data, 0.94);
    const expected = Math.abs(Math.log(1.01)) * 100;
    const period = pinePeriod(0.94);

    for (let i = 0; i < period; i++) {
        assert.ok(isNaN(result[i]), `expected warmup NaN at ${i}`);
    }
    for (let i = period; i < result.length; i++) {
        assert.ok(Math.abs(result[i] - expected) <= 1e-12, `unexpected value at ${i}`);
    }
});

test('ewma_volatility_js matches manual reference', () => {
    const data = geometricSeries(96, 25, 1.003);
    data[12] = NaN;
    data[13] = 0;
    const expected = manualEwmaVolatility(data, 0.90);
    const result = wasm.ewma_volatility_js(data, 0.90);
    assertArrayClose(result, expected, 1e-12, 'manual reference mismatch');
});

test('ewma_volatility_batch_js first row matches single', () => {
    const data = geometricSeries(128, 100, 1.002);
    const batch = wasm.ewma_volatility_batch_js(data, {
        lambda_range: [0.90, 0.94, 0.02],
    });
    const single = wasm.ewma_volatility_js(data, 0.90);

    assert.strictEqual(batch.rows, 3);
    assert.strictEqual(batch.cols, data.length);
    assert.strictEqual(batch.combos.length, 3);
    assert.ok(Math.abs(batch.combos[0].lambda - 0.90) <= 1e-12);
    assertArrayClose(batch.values.slice(0, data.length), single, 1e-12, 'batch row mismatch');
});

test('ewma_volatility_into pointer path matches safe API', () => {
    const data = geometricSeries(128, 75, 1.004);
    const safe = wasm.ewma_volatility_js(data, 0.94);
    const inPtr = wasm.allocate_f64_array(data.length);
    const outPtr = wasm.ewma_volatility_alloc(data.length);

    try {
        wasm.write_f64_array(inPtr, data);
        wasm.ewma_volatility_into(inPtr, outPtr, data.length, 0.94);
        const pointed = wasm.read_f64_array(outPtr, data.length);
        assertArrayClose(pointed, safe, 1e-12, 'pointer path mismatch');
    } finally {
        wasm.ewma_volatility_free(outPtr, data.length);
        wasm.deallocate_f64_array(inPtr);
    }
});

test('ewma_volatility_batch_into returns row count', () => {
    const data = geometricSeries(64, 50, 1.002);
    const inPtr = wasm.allocate_f64_array(data.length);
    const rows = 3;
    const outPtr = wasm.ewma_volatility_alloc(rows * data.length);

    try {
        wasm.write_f64_array(inPtr, data);
        const returnedRows = wasm.ewma_volatility_batch_into(
            inPtr,
            outPtr,
            data.length,
            0.90,
            0.94,
            0.02,
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.ewma_volatility_free(outPtr, rows * data.length);
        wasm.deallocate_f64_array(inPtr);
    }
});

test('ewma_volatility_js rejects invalid lambda', () => {
    const data = geometricSeries(32, 10, 1.01);
    assert.throws(() => {
        wasm.ewma_volatility_js(data, 1.0);
    }, /Invalid lambda/);
});

test('ewma_volatility_batch_js rejects invalid config', () => {
    const data = geometricSeries(32, 10, 1.01);
    assert.throws(() => {
        wasm.ewma_volatility_batch_js(data, { lambda_range: [0.94, 0.95] });
    }, /Invalid config/);
});
