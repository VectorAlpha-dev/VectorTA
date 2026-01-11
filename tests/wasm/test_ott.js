/**
 * WASM binding tests for OTT indicator.
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
  } catch (error) {
    console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
    throw error;
  }

  testData = loadTestData();
});

test('OTT accuracy (matches Rust refs)', () => {
  const close = new Float64Array(testData.close);
  const expected = EXPECTED_OUTPUTS.ott;

  const result = wasm.ott_js(
    close,
    expected.accuracyParams.period,
    expected.accuracyParams.percent,
    expected.accuracyParams.ma_type
  );

  assert.strictEqual(result.length, close.length);

  
  const last5 = result.slice(-5);
  assertArrayClose(last5, expected.last5Values, 1e-8, 'OTT last 5 values mismatch');
});

test('OTT reinput (apply twice) matches refs', () => {
  const close = new Float64Array(testData.close);
  const expected = EXPECTED_OUTPUTS.ott;
  const { period, percent, ma_type } = expected.accuracyParams;

  const first = wasm.ott_js(close, period, percent, ma_type);
  assert.strictEqual(first.length, close.length);

  const second = wasm.ott_js(first, period, percent, ma_type);
  assert.strictEqual(second.length, first.length);

  const last5 = second.slice(-5);
  assertArrayClose(last5, expected.reinputLast5, 1e-8, 'OTT reinput last 5 mismatch');
});

test('OTT invalid inputs error', () => {
  
  const data = new Float64Array([10, 20, 30]);
  assert.throws(() => wasm.ott_js(data, 0, 1.4, 'VAR'), /Invalid period/);

  
  const empty = new Float64Array([]);
  assert.throws(() => wasm.ott_js(empty, 2, 1.4, 'VAR'), /Input data slice is empty/);

  
  assert.throws(() => wasm.ott_js(data, 2, -1.0, 'VAR'), /Invalid percent/);
  assert.throws(() => wasm.ott_js(data, 2, NaN, 'VAR'), /Invalid percent/);

  
  assert.throws(() => wasm.ott_js(data, 2, 1.4, 'INVALID'), /Invalid moving average|Invalid MA type|Unsupported/);
});

test('OTT batch single parameter matches single', () => {
  const close = new Float64Array(testData.close);
  const expected = EXPECTED_OUTPUTS.ott;
  const { period, percent, ma_type } = expected.accuracyParams;

  const batch = wasm.ott_batch(close, {
    period_range: [period, period, 0],
    percent_range: [percent, percent, 0.0],
    ma_types: [ma_type],
  });

  assert.strictEqual(batch.rows, 1);
  assert.strictEqual(batch.cols, close.length);
  assert.strictEqual(batch.values.length, close.length);

  const single = wasm.ott_js(close, period, percent, ma_type);
  assertArrayClose(batch.values, single, 1e-8, 'Batch vs single mismatch');
});

