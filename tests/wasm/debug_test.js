import { loadTestData } from './test_utils.js';

const testData = loadTestData();
console.log('Test data loaded:');
console.log('- Open length:', testData.open.length);
console.log('- High length:', testData.high.length);
console.log('- Low length:', testData.low.length);
console.log('- Close length:', testData.close.length);
console.log('- Volume length:', testData.volume.length);

console.log('\nFirst 5 close values:', testData.close.slice(0, 5));
console.log('Last 5 close values:', testData.close.slice(-5));
