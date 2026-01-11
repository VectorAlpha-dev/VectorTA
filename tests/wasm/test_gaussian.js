/**
 * WASM binding tests for Gaussian indicator.
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
  EXPECTED_OUTPUTS,
} from './test_utils.js';
import { compareWithRust } from './rust-comparison.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
  
  try {
    const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
    const importPath = process.platform === 'win32' ? 'file:///' + wasmPath.replace(/\\/g, '/') : wasmPath;
    wasm = await import(importPath);
  } catch (error) {
    console.error('Failed to load WASM module. Run "wasm-pack build -- --features wasm --no-default-features" first');
    throw error;
  }

  testData = loadTestData();
});

test('Gaussian accuracy', async () => {
  const close = new Float64Array(testData.close);
  const expected = EXPECTED_OUTPUTS.gaussian;

  const result = await wasm.gaussian_js(close, expected.defaultParams.period, expected.defaultParams.poles);

  assert.strictEqual(result.length, close.length);

  
  const last5 = result.slice(-5);
  assertArrayClose(last5, expected.last5Values, 1e-4, 'Gaussian last 5 values mismatch');

  
  await compareWithRust('gaussian', result, 'close', expected.defaultParams, 1e-10);
});

test('Gaussian zero period', () => {
  const data = new Float64Array([10, 20, 30]);
  assert.throws(() => wasm.gaussian_js(data, 0, 4));
});

test('Gaussian period exceeds length', () => {
  const data = new Float64Array([10, 20, 30]);
  assert.throws(() => wasm.gaussian_js(data, 10, 4));
});

test('Gaussian very small dataset', () => {
  const data = new Float64Array([42]);
  assert.throws(() => wasm.gaussian_js(data, 14, 4));
});

test('Gaussian empty input', () => {
  const data = new Float64Array([]);
  assert.throws(() => wasm.gaussian_js(data, 14, 4));
});

test('Gaussian invalid poles', () => {
  const data = new Float64Array([1, 2, 3, 4, 5]);
  assert.throws(() => wasm.gaussian_js(data, 3, 0)); 
  assert.throws(() => wasm.gaussian_js(data, 3, 5)); 
});

test('Gaussian period one degeneracy', () => {
  const data = new Float64Array([1, 2, 3, 4, 5]);
  assert.throws(() => wasm.gaussian_js(data, 1, 2));
});

test('Gaussian NaN handling', () => {
  const close = new Float64Array(testData.close);
  const res = wasm.gaussian_js(close, 14, 4);
  assert.strictEqual(res.length, close.length);
  
  const skip = 4; 
  for (let i = skip; i < res.length; i++) {
    assert(Number.isFinite(res[i]), `Non-finite value at index ${i}`);
  }
});

test('Gaussian batch single parameter set', () => {
  const close = new Float64Array(testData.close);
  const single = wasm.gaussian_js(close, 14, 4);
  const batch = wasm.gaussian_batch(close, { period_range: [14, 14, 0], poles_range: [4, 4, 0] });

  assert.strictEqual(batch.values.length, single.length);
  assert.strictEqual(batch.rows, 1);
  assert.strictEqual(batch.cols, close.length);
  assertArrayClose(batch.values, single, 1e-10, 'Batch vs single mismatch');
});

test('Gaussian batch multiple combinations + metadata', () => {
  const close = new Float64Array(testData.close.slice(0, 50));
  const result = wasm.gaussian_batch(close, { period_range: [10, 20, 5], poles_range: [2, 4, 1] });

  
  assert.strictEqual(result.rows, 9);
  assert.strictEqual(result.cols, 50);
  assert.strictEqual(result.values.length, 9 * 50);
  assert.strictEqual(result.combos.length, 9);

  
  const idx = result.combos.findIndex(c => c.period === 15 && c.poles === 3);
  assert.ok(idx >= 0, 'Expected combo (15,3) missing');
  const row = result.values.slice(idx * result.cols, (idx + 1) * result.cols);
  const single = wasm.gaussian_js(close, 15, 3);
  assertArrayClose(row, single, 1e-10, 'Batch row mismatch for (period=15,poles=3)');
});

test('Gaussian zero-copy API (alloc/into/free)', () => {
  const data = new Float64Array(Array.from({ length: 64 }, (_, i) => i + 1));
  const period = 10;
  const poles = 3;

  
  const ptr = wasm.gaussian_alloc(data.length);
  assert(ptr !== 0, 'Failed to allocate memory');

  const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
  if (!memory || !memory.buffer) {
    console.warn('Skipping zero-copy API test: wasm memory accessor unavailable');
    wasm.gaussian_free(ptr, data.length);
    return;
  }

  const view = new Float64Array(memory.buffer, ptr, data.length);
  view.set(data);

  try {
    
    wasm.gaussian_into(ptr, ptr, data.length, period, poles);
    const regular = wasm.gaussian_js(data, period, poles);
    for (let i = 0; i < data.length; i++) {
      if (isNaN(regular[i]) && isNaN(view[i])) continue;
      assert(Math.abs(regular[i] - view[i]) < 1e-10, `Zero-copy mismatch at ${i}`);
    }
  } finally {
    wasm.gaussian_free(ptr, data.length);
  }
});

