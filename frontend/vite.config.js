import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/run': { target: 'http://localhost:7860', changeOrigin: true },
      '/api': { target: 'http://localhost:7860', changeOrigin: true },
      '/queue': { target: 'http://localhost:7860', changeOrigin: true, ws: true },
      '/upload': { target: 'http://localhost:7860', changeOrigin: true },
      '/yf': {
        target: 'https://query1.finance.yahoo.com',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/yf/, ''),
      },
    },
  },
})
