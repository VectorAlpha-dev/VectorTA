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
    
    console.log('Test data properties:');
    console.log('Length:', hl2Array.length);
    console.log('First 10 values:', Array.from(hl2Array.slice(0, 10)).map(v => v.toFixed(2)));
    
    // Check for NaN values
    let firstNonNaN = -1;
    for (let i = 0; i < hl2Array.length; i++) {
        if (!isNaN(hl2Array[i])) {
            firstNonNaN = i;
            break;
        }
    }
    console.log('First non-NaN index:', firstNonNaN);
    
    // Test single call
    const singleResult = wasm.alligator_js(hl2Array, 13, 8, 8, 5, 5, 3);
    const len = hl2Array.length;
    console.log('\nData length:', len);
    const singleJaw = singleResult.slice(0, len);
    const singleTeeth = singleResult.slice(len, 2 * len);
    const singleLips = singleResult.slice(2 * len, 3 * len);
    
    console.log('\nSingle call results:');
    console.log('Lips at index 4:', singleLips[4]);
    console.log('Lips around index 4:', singleLips.slice(2, 7));
    
    // Test batch call
    const batchResult = wasm.alligator_batch_js(
        hl2Array,
        13, 13, 0,  // jaw_period (just one value: 13)
        8, 8, 0,    // jaw_offset
        8, 8, 0,    // teeth_period
        5, 5, 0,    // teeth_offset
        5, 5, 0,    // lips_period
        3, 3, 0     // lips_offset
    );
    
    const batchJaw = batchResult.slice(0, len);
    const batchTeeth = batchResult.slice(len, 2 * len);
    const batchLips = batchResult.slice(2 * len, 3 * len);
    
    console.log('\nBatch call results:');
    console.log('Lips at index 4:', batchLips[4]);
    console.log('Lips around index 4:', batchLips.slice(2, 7));
    
    // Compare
    console.log('\nComparison:');
    console.log('Single lips[4]:', singleLips[4]);
    console.log('Batch lips[4]:', batchLips[4]);
    console.log('Difference:', Math.abs(singleLips[4] - batchLips[4]));
    
    // Check the expected value
    console.log('\nExpected value from test: 59653.53285975852');
    console.log('Actual batch value: 2725.2459520000007');
    
    // Check the actual test data value
    console.log('\nActual data around the expected value:');
    console.log('hl2Array[4]:', hl2Array[4]);
    
    // Let's check the raw test data
    console.log('\nChecking warmup calculations:');
    console.log('Lips period: 5, offset: 3');
    console.log('Expected first valid index for lips:', firstNonNaN + 5 - 1);
    console.log('With offset, values should appear starting at:', firstNonNaN + 5 - 1 + 3);
}

debug().catch(console.error);