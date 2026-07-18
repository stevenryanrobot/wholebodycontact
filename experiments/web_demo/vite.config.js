import { defineConfig } from 'vite';

export default defineConfig({
  // @mujoco/mujoco resolves its .wasm next to mujoco.js via import.meta.url;
  // esbuild pre-bundling would break that, so keep both wasm-heavy deps out.
  optimizeDeps: {
    exclude: ['@mujoco/mujoco', 'onnxruntime-web'],
  },
  build: {
    target: 'esnext',
    // .wasm and .onnx assets are large; silence the size warning.
    chunkSizeWarningLimit: 8192,
  },
  server: {
    // Not strictly required (single-threaded mujoco build + ort falls back to
    // 1 thread without cross-origin isolation), but harmless and lets the
    // multi-threaded paths work if enabled later.
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
