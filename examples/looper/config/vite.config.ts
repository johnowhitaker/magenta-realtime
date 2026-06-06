import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react-swc';
import path from 'path';
import {fileURLToPath} from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const abs = (relativePath: string) =>
  path.resolve(__dirname, '..', relativePath);

export default defineConfig({
  base: './',
  server: {
    port: 62430,
    proxy: {
      '/api': 'http://127.0.0.1:8765',
      '/sa3': 'http://127.0.0.1:8766',
    },
  },
  plugins: [react()],
  resolve: {
    alias: {
      '@': abs('src'),
    },
  },
  build: {
    outDir: abs('dist'),
    emptyOutDir: true,
  },
});
