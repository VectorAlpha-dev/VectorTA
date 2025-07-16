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
    const hl2 = testData.high.map((h, i) => (h + testData.low[i]) / 2);
    const hl2Array = new Float64Array(hl2);
    
    console.log('Testing with full data, length:', hl2Array.length);
    
    // Get single result
    const singleResult = wasm.alligator_js(hl2Array, 13, 8, 8, 5, 5, 3);
    const len = hl2Array.length;
    const singleLips = singleResult.slice(2 * len, 3 * len);
    
    console.log('\nSingle call - Lips values:');
    console.log('Index 4:', singleLips[4]);
    console.log('First 10 values:', Array.from(singleLips.slice(0, 10)).map((v, i) => `[${i}]=${v}`));
    
    // Find first non-NaN, non-zero value
    let firstValidLips = -1;
    for (let i = 0; i < singleLips.length; i++) {
        if (!isNaN(singleLips[i]) && singleLips[i] !== 0) {
            firstValidLips = i;
            break;
        }
    }
    console.log('First valid lips value at index:', firstValidLips);
    if (firstValidLips >= 0) {
        console.log('Value:', singleLips[firstValidLips]);
    }
    
    // Now test batch
    const batchResult = wasm.alligator_batch_js(
        hl2Array,
        13, 13, 0,  // jaw_period range
        8, 8, 0,    // jaw_offset range
        8, 8, 0,    // teeth_period range
        5, 5, 0,    // teeth_offset range
        5, 5, 0,    // lips_period range
        3, 3, 0     // lips_offset range
    );
    
    const batchLips = batchResult.slice(2 * len, 3 * len);
    
    console.log('\nBatch call - Lips values:');
    console.log('Index 4:', batchLips[4]);
    console.log('First 10 values:', Array.from(batchLips.slice(0, 10)).map((v, i) => `[${i}]=${v}`));
    
    // Check where they differ
    console.log('\nFinding differences:');
    let diffCount = 0;
    for (let i = 0; i < Math.min(20, len); i++) {
        const s = singleLips[i];
        const b = batchLips[i];
        if (isNaN(s) && isNaN(b)) continue;
        if (Math.abs(s - b) > 1e-10) {
            console.log(`Diff at index ${i}: single=${s}, batch=${b}`);
            diffCount++;
            if (diffCount >= 5) break;
        }
    }
}

debug().catch(console.error);