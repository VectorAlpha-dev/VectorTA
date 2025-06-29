import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import sitemap from '@astrojs/sitemap';
import react from '@astrojs/react';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

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
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            'vendor-charts': ['lightweight-charts'],
            'vendor-react': ['react', 'react-dom'],
            'vendor-utils': ['papaparse', 'recharts'],
          }
        }
      }
    },
    optimizeDeps: {
      include: ['lightweight-charts', 'react', 'react-dom']
    }
  },
  output: 'static',
  build: {
    inlineStylesheets: 'auto'
  }
});