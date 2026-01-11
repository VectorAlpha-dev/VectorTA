/**
 * WASM binding tests for EPMA indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly
 * and use the same reference values and tolerance.
 */
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import {
  loadTestData,
  assertArrayClose,
  isNaN,
  EXPECTED_OUTPUTS,
} from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
  
  const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
  const importPath = process.platform === 'win32'
    ? 'file:///' + wasmPath.replace(/\\/g, '/')
    : wasmPath;
  try {
    wasm = await import(importPath);
  } catch (err) {
    console.error('Failed to load WASM module. Run "wasm-pack build -- --features wasm --no-default-features" first');
    throw err;
  }
  testData = loadTestData();
});

test('EPMA accuracy (last 5 values)', () => {
  const close = new Float64Array(testData.close);
  const cfg = EXPECTED_OUTPUTS.epma;
  const out = wasm.epma_js(close, cfg.defaultParams.period, cfg.defaultParams.offset);
  assert.strictEqual(out.length, close.length);

  
  const last5 = out.slice(-5);
  assertArrayClose(last5, cfg.lastFive, 1e-1, 'EPMA last 5 values mismatch');
});

test('EPMA default candles length', () => {
  const close = new Float64Array(testData.close);
  const cfg = EXPECTED_OUTPUTS.epma;
  const out = wasm.epma_js(close, cfg.defaultParams.period, cfg.defaultParams.offset);
  assert.strictEqual(out.length, close.length);
});

test('EPMA invalid parameters', () => {
  const data = new Float64Array([10.0, 20.0, 30.0]);
  assert.throws(() => wasm.epma_js(data, 0, 4), /Invalid period|Invalid params|Invalid offset/i);
  assert.throws(() => wasm.epma_js(data, 10, 0), /Invalid period|Invalid params|Invalid offset/i);
});

test('EPMA empty and NaN input handling', () => {
  const empty = new Float64Array();
  assert.throws(() => wasm.epma_js(empty, 11, 4), /empty|length/i);

  const allNaN = new Float64Array(20);
  for (let i = 0; i < allNaN.length; i++) allNaN[i] = NaN;
  assert.throws(() => wasm.epma_js(allNaN, 11, 4), /All values are NaN/i);
});

test('EPMA warmup NaNs then valid values', () => {
  const close = new Float64Array(testData.close.slice(0, 200));
  const { defaultParams, warmupPeriod } = EXPECTED_OUTPUTS.epma;
  const out = wasm.epma_js(close, defaultParams.period, defaultParams.offset);
  
  for (let i = 0; i < Math.min(out.length, warmupPeriod); i++) {
    assert.ok(isNaN(out[i]), `Expected NaN during warmup at index ${i}`);
  }
  
  for (let i = warmupPeriod; i < out.length; i++) {
    assert.ok(!isNaN(out[i]), `Unexpected NaN after warmup at index ${i}`);
  }
});

test('EPMA batch API (default row check)', () => {
  const close = new Float64Array(testData.close);
  const cfg = EXPECTED_OUTPUTS.epma;
  const batch = wasm.epma_batch(close, {
    period_range: [cfg.defaultParams.period, cfg.defaultParams.period, 1],
    offset_range: [cfg.defaultParams.offset, cfg.defaultParams.offset, 1],
  });
  assert.strictEqual(batch.rows, 1);
  assert.strictEqual(batch.cols, close.length);
  assert.strictEqual(batch.values.length, batch.rows * batch.cols);
  assert.ok(Array.isArray(batch.combos) && batch.combos.length === 1);

  const row = batch.values.slice(batch.cols * 0, batch.cols * 1);
  const last5 = row.slice(-5);
  assertArrayClose(last5, cfg.lastFive, 1e-1, 'EPMA batch default row last 5 mismatch');
});

test('EPMA fast API (epma_into) matches safe API', () => {
  const close = new Float64Array(testData.close.slice(0, 256));
  const { period, offset } = EXPECTED_OUTPUTS.epma.defaultParams;

  const len = close.length;
  const inPtr = wasm.epma_alloc(len);
  const outPtr = wasm.epma_alloc(len);
  try {
    const mem = new Float64Array(wasm.__wasm.memory.buffer);
    const inOff = inPtr / 8;
    for (let i = 0; i < len; i++) mem[inOff + i] = close[i];

    wasm.epma_into(inPtr, outPtr, len, period, offset);

    const outOff = outPtr / 8;
    const fastOut = new Float64Array(len);
    for (let i = 0; i < len; i++) fastOut[i] = mem[outOff + i];

    const safeOut = wasm.epma_js(close, period, offset);
    assertArrayClose(fastOut, safeOut, 1e-10, 'EPMA fast API mismatch');
  } finally {
    wasm.epma_free(inPtr, len);
    wasm.epma_free(outPtr, len);
  }
});
