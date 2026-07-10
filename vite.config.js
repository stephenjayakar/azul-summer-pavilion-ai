import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  root: "frontend",
  plugins: [react()],
  build: {
    outDir: "../src/azul/web_static",
    emptyOutDir: false,
    rollupOptions: {
      output: { entryFileNames: "app.js", assetFileNames: "styles.css" }
    }
  }
});
