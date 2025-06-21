import { createServer } from 'vite';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const rootDir = join(__dirname, '..');

async function startDevServer() {
  try {
    // First run prebuild
    console.log('Running prebuild scripts...');
    const { execSync } = await import('child_process');
    execSync('node scripts/run-prebuild.js', { 
      stdio: 'inherit',
      cwd: rootDir
    });
    
    console.log('\nStarting development server...');
    console.log('Visit http://localhost:4321 to see your site\n');
    
    // Try to use Astro's Vite config
    let astroConfig;
    try {
      const astro = await import('astro');
      const config = await import('../astro.config.mjs');
      astroConfig = config.default;
    } catch (e) {
      console.log('Using basic Vite config...');
    }
    
    // Create Vite server
    const server = await createServer({
      root: rootDir,
      server: {
        port: 4321,
        host: true
      },
      configFile: false,
      ...astroConfig?.vite
    });
    
    await server.listen();
    server.printUrls();
    
  } catch (error) {
    console.error('Error starting dev server:', error);
    console.log('\nTrying alternative approach...\n');
    
    // Fallback: just serve the dist folder if it exists
    const express = await import('express').catch(() => null);
    if (express) {
      const app = express.default();
      app.use(express.default.static(join(rootDir, 'dist')));
      app.listen(4321, () => {
        console.log('Static server running at http://localhost:4321');
        console.log('Run "npm run build" first if you see a blank page');
      });
    } else {
      console.log('Please install dependencies and try again:');
      console.log('  1. Delete node_modules and package-lock.json');
      console.log('  2. Run: npm install');
      console.log('  3. Run: npm run dev');
    }
  }
}

startDevServer();