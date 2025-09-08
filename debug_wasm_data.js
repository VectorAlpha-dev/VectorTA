const wasm = require('./pkg/my_project.js');
const fs = require('fs');

// Load data directly from CSV
const csvContent = fs.readFileSync('src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv', 'utf8');
const lines = csvContent.split('\n').filter(line => line.trim());
console.log('CSV lines (including header):', lines.length);
console.log('CSV data lines (excluding header):', lines.length - 1);

// Load using test_utils
const { loadTestData } = require('./tests/wasm/test_utils.js');
const testData = loadTestData();
console.log('test_utils data length:', testData.close.length);

// Create simple test arrays
const simpleHigh = new Float64Array([10, 11, 12]);
const simpleLow = new Float64Array([8, 9, 10]);
const simpleClose = new Float64Array([9, 10, 11]);

// Test with simple data
const simpleResult = wasm.chandelier_exit_wasm(simpleHigh, simpleLow, simpleClose, 2, 1.0, true);
console.log('\nSimple test:');
console.log('Input length:', simpleClose.length);
console.log('Output length:', simpleResult.short_stop.length);
console.log('Short stops:', simpleResult.short_stop);