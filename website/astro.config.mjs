import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import sitemap from '@astrojs/sitemap';
import react from '@astrojs/react';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
  site: 'https://ta-indicators-demo.com', // Update with actual domain
  integrations: [tailwind(), sitemap(), react()],
  vite: {
    plugins: [wasm(), topLevelAwait()]
  }
});