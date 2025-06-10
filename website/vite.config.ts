import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	optimizeDeps: {
		exclude: ['lightweight-charts'],
		include: ['@sveltejs/kit'],
		esbuildOptions: {
			target: 'esnext'
		}
	},
	server: {
		fs: {
			allow: ['..']
		},
		hmr: {
			overlay: false // Reduces HMR overhead
		}
	},
	build: {
		target: 'esnext',
		rollupOptions: {
			output: {
				manualChunks: {
					vendor: ['svelte']
				}
			}
		}
	},
	esbuild: {
		target: 'esnext'
	},
	define: {
		'process.env.NODE_ENV': '"development"'
	}
});
