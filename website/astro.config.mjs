import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import sitemap from '@astrojs/sitemap';
import react from '@astrojs/react';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';
import { resolve } from 'path';

export default defineConfig({
  site: 'https://vectoralpha.dev',
  base: '/ta',
  integrations: [
    tailwind(), 
    sitemap(), 
    react()
  ],
  vite: {
    plugins: [wasm(), topLevelAwait()],
    resolve: {
      alias: {
        '@': resolve('./src')
      }
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            'vendor-charts': ['lightweight-charts'],
            'vendor-react': ['react', 'react-dom'],
            'vendor-utils': ['papaparse', 'recharts'],
            'vendor-search': ['fuse.js'],
          }
        }
      }
    },
    optimizeDeps: {
      include: ['lightweight-charts', 'react', 'react-dom', 'fuse.js']
    }
  },
  output: 'static',
  build: {
    inlineStylesheets: 'auto'
  },
  // Remove experimental features that are no longer supported in Astro v5
  // Prefetch configuration for better navigation performance
  prefetch: {
    prefetchAll: false,
    defaultStrategy: 'viewport'
  },
  // Enable compression for smaller bundle sizes
  compressHTML: true,
});