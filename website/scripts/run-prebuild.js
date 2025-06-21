import { execSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log('Running prebuild scripts...');

try {
  // Run scan-indicators
  console.log('Scanning indicators...');
  execSync('node scripts/scan-indicators.js', { 
    stdio: 'inherit',
    cwd: join(__dirname, '..')
  });
  
  // Run cache-data
  console.log('Caching sample data...');
  execSync('node scripts/cache-sample-data.js', { 
    stdio: 'inherit',
    cwd: join(__dirname, '..')
  });
  
  console.log('Prebuild complete!');
} catch (error) {
  console.error('Prebuild failed:', error.message);
  process.exit(1);
}