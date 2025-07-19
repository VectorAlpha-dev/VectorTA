// Verify LinReg WASM computation correctness
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function verifyCorrectness() {
    // Load WASM module
    const wasmPath = join(__dirname, 'pkg/my_project.js');
    const importPath = process.platform === 'win32' 
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    const wasm = await import(importPath);
    
    // Create known test data where we can predict the LinReg output
    // For a perfect linear sequence, LinReg should predict the next value
    const testData = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 3;
    
    console.log("Testing LinReg correctness with linear data:");
    console.log("Input:", Array.from(testData));
    
    // Test Safe API
    console.log("\nSafe API result:");
    const safeResult = wasm.linreg_js(testData, period);
    console.log("Output:", Array.from(safeResult));
    
    // Test Fast API
    const len = testData.length;
    const inPtr = wasm.linreg_alloc(len);
    const outPtr = wasm.linreg_alloc(len);
    
    const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
    inView.set(testData);
    
    wasm.linreg_into(inPtr, outPtr, len, period);
    
    const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
    const fastResult = Array.from(outView);
    
    console.log("\nFast API result:");
    console.log("Output:", fastResult);
    
    // Verify results match
    let match = true;
    for (let i = 0; i < len; i++) {
        if (Math.abs(safeResult[i] - fastResult[i]) > 1e-10) {
            console.error(`Mismatch at index ${i}: safe=${safeResult[i]}, fast=${fastResult[i]}`);
            match = false;
        }
    }
    
    if (match) {
        console.log("\n✓ Safe and Fast API results match!");
    } else {
        console.log("\n✗ Safe and Fast API results DO NOT match!");
    }
    
    // Test with more complex data
    console.log("\n\nTesting with sinusoidal data:");
    const complexData = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        complexData[i] = Math.sin(i * 0.1) * 50 + 100;
    }
    
    const complexResult = wasm.linreg_js(complexData, 14);
    console.log("First 20 values:", Array.from(complexResult.slice(0, 20)));
    console.log("Last 5 values:", Array.from(complexResult.slice(-5)));
    
    // Check for NaN handling
    const nanCount = Array.from(complexResult).filter(x => isNaN(x)).length;
    console.log(`\nNaN count: ${nanCount} (expected: ${14 - 1} = 13)`);
    
    // Verify warmup period
    for (let i = 0; i < 13; i++) {
        if (!isNaN(complexResult[i])) {
            console.error(`Expected NaN at index ${i} but got ${complexResult[i]}`);
        }
    }
    
    // Verify no NaN after warmup
    for (let i = 13; i < complexResult.length; i++) {
        if (isNaN(complexResult[i])) {
            console.error(`Unexpected NaN at index ${i}`);
        }
    }
    
    // Cleanup
    wasm.linreg_free(inPtr, len);
    wasm.linreg_free(outPtr, len);
    
    console.log("\nCorrectness verification complete!");
}

verifyCorrectness().catch(console.error);