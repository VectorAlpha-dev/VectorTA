import { fileURLToPath } from 'url';
import path from 'path';
import { loadTestData } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function debug() {
    const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
    const importPath = process.platform === 'win32' 
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    const wasm = await import(importPath);
    
    const testData = loadTestData();
    const hl2 = testData.high.slice(0, 100).map((h, i) => (h + testData.low[i]) / 2);
    const hl2Array = new Float64Array(hl2);
    
    // Test batch with multiple jaw periods to understand layout
    const batchResult = wasm.alligator_batch_js(
        hl2Array,
        13, 17, 2,  // jaw_period range: 13, 15, 17
        8, 8, 0,    // jaw_offset range
        8, 8, 0,    // teeth_period range
        5, 5, 0,    // teeth_offset range
        5, 5, 0,    // lips_period range
        3, 3, 0     // lips_offset range
    );
    
    console.log('Batch result length:', batchResult.length);
    console.log('Expected: 3 combos * 100 data points * 3 arrays = 900');
    console.log('Actual matches expected:', batchResult.length === 900);
    
    // The layout should be:
    // [jaw_combo1, jaw_combo2, jaw_combo3, teeth_combo1, teeth_combo2, teeth_combo3, lips_combo1, lips_combo2, lips_combo3]
    const totalArraySize = 3 * 100; // 3 combos * 100 data points
    
    console.log('\nChecking jaw period 13 (combo 0):');
    const jawStart0 = 0;
    const jaw0 = batchResult.slice(jawStart0, jawStart0 + 100);
    console.log('Jaw[12]:', jaw0[12]);
    console.log('Jaw around index 12:', jaw0.slice(10, 15));
    
    // Get single result for comparison
    const single13 = wasm.alligator_js(hl2Array, 13, 8, 8, 5, 5, 3);
    const singleJaw13 = single13.slice(0, 100);
    console.log('\nSingle call jaw[12]:', singleJaw13[12]);
    console.log('Single jaw around index 12:', singleJaw13.slice(10, 15));
    
    // Check if the issue is with warmup calculation
    console.log('\nWarmup info:');
    console.log('Jaw period 13 warmup should be at index:', 13 - 1, '= 12');
    console.log('With offset 8, first value appears at:', 12 + 8, '= 20');
    
    // Check hex representation of the problematic value
    const problematicValue = 3.6073928447e-313;
    console.log('\nProblematic value analysis:');
    console.log('Value:', problematicValue);
    const buf = new ArrayBuffer(8);
    const view = new DataView(buf);
    view.setFloat64(0, problematicValue, true);
    const hex = Array.from(new Uint8Array(buf))
        .map(b => b.toString(16).padStart(2, '0'))
        .join(' ');
    console.log('Hex representation:', hex);
    console.log('This looks like:', hex === '22 22 22 22 22 22 22 00' ? 'Poison value!' : 'Something else');
}

debug().catch(console.error);