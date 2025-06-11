// @ts-check
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
  build: {
    assets: '_astro'
  },
  vite: {
    optimizeDeps: {
      exclude: ['lightweight-charts']
    },
    server: {
      fs: {
        allow: ['..']
      }
    }
  }
});
