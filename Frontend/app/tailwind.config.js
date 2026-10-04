/** Design tokens from the Stitch "Obsidian Lumina" system. */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        base: "#090b10",
        raised: "#0f141f",
        overlay: "#161e2e",
        container: "#1e1f25",
        accent: { DEFAULT: "#6366f1", soft: "#818cf8" },
        violet: "#a855f7",
        ok: { DEFAULT: "#10b981", soft: "#34d399" },
        warn: "#f59e0b",
        danger: "#f43f5e",
        cyan: "#06b6d4",
        ink: { DEFAULT: "#f8fafc", muted: "#94a3b8", faint: "#64748b" },
      },
      fontFamily: {
        sans: ["Geist", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
