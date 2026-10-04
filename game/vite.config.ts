import { defineConfig } from 'vite';
import { resolve } from 'node:path';

export default defineConfig({
  server: { host: true, port: 5173 },
  build: {
    target: 'es2022',
    rollupOptions: {
      input: { main: resolve(__dirname, 'index.html'), board: resolve(__dirname, 'board.html') },
    },
  },
});
