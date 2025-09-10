const wasm = require('./pkg/my_project.js');
const { loadTestData } = require('./tests/wasm/test_utils.js');

const testData = loadTestData();

const result = wasm.chandelier_exit_wasm(
    testData.high,
    testData.low,
    testData.close,
    22,     // period
    3.0,    // mult
    true    // use_close
);

console.log('Result length:', result.short_stop.length);

// Check indices 15386-15390
const expectedIndices = [15386, 15387, 15388, 15389, 15390];
const expectedValues = [68719.23648167, 68705.54391432, 68244.42828185, 67599.49972358, 66883.02246342];

console.log('\nValues at expected indices:');
for (let i = 0; i < expectedIndices.length; i++) {
    const idx = expectedIndices[i];
    const actual = result.short_stop[idx];
    const expected = expectedValues[i];
    const match = Math.abs(actual - expected) < 0.01 ? '✓' : '✗';
    console.log(`  short_stop[${idx}] = ${actual.toFixed(8)} (expected ${expected}) ${match}`);
}

// Also show a few before and after
console.log('\nContext around 15384-15392:');
for (let i = 15384; i < 15392; i++) {
    const short = result.short_stop[i];
    const long = result.long_stop[i];
    console.log(`  [${i}]: short=${!isNaN(short) ? short.toFixed(8) : 'NaN'}, long=${!isNaN(long) ? long.toFixed(8) : 'NaN'}`);
}

// Show what the test is actually getting
console.log('\nWhat test_chandelier_exit.js gets:');
const testIndices = [15386, 15387, 15388, 15389, 15390];
const testActualValues = testIndices.map(i => result.short_stop[i]);
console.log('Indices:', testIndices);
console.log('Values:', testActualValues.map(v => v.toFixed(8)));

// Check if there's an off-by-one error
console.log('\nChecking for off-by-one error:');
console.log('Value at 15385:', result.short_stop[15385].toFixed(8));
console.log('Value at 15386:', result.short_stop[15386].toFixed(8));