import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, "..", "");
  const apiTarget = env.VITE_API_PROXY_TARGET || "http://127.0.0.1:5000";

  return {
    envDir: "..",
    plugins: [react()],
    server: {
      host: env.VITE_DEV_HOST || "127.0.0.1",
      port: Number(env.VITE_DEV_PORT || 5173),
      proxy: {
        "/api": apiTarget,
      },
    },
  };
});
