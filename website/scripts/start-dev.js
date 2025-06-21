import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const rootDir = join(__dirname, '..');

// First run prebuild
console.log('Running prebuild...');
const prebuild = spawn('node', ['scripts/run-prebuild.js'], {
  stdio: 'inherit',
  cwd: rootDir,
  shell: true
});

prebuild.on('close', (code) => {
  if (code !== 0) {
    console.error('Prebuild failed');
    process.exit(1);
  }
  
  console.log('\nStarting Astro dev server...');
  
  // Try to find astro executable
  const astroPath = join(rootDir, 'node_modules', '.bin', 'astro');
  const astroCmd = process.platform === 'win32' ? `${astroPath}.cmd` : astroPath;
  
  // Start Astro
  const astro = spawn(astroCmd, ['dev'], {
    stdio: 'inherit',
    cwd: rootDir,
    shell: true
  });
  
  astro.on('error', (err) => {
    console.error('Failed to start Astro:', err);
    // Fallback to using node directly
    const astroJs = join(rootDir, 'node_modules', 'astro', 'astro.js');
    const fallback = spawn('node', [astroJs, 'dev'], {
      stdio: 'inherit',
      cwd: rootDir,
      shell: true
    });
  });
});