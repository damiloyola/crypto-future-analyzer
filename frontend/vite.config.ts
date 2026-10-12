/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const backendPort = process.env.BACKEND_PORT ?? "8000";
const frontendPort = Number(process.env.FRONTEND_PORT ?? "5173");

// La UI le habla a la API por /api (mismo origen). Vite lo reenvía al backend,
// así desde el celular alcanza con abrir http://<ip-de-la-mac>:5173.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true, // escucha en 0.0.0.0 para la red wifi local
    port: frontendPort,
    proxy: {
      "/api": `http://127.0.0.1:${backendPort}`,
    },
  },
  test: {
    environment: "node",
  },
});
