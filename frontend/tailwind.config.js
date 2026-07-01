/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        base: "#0B1220",
        surface: "#141B2E",
        surfaceSoft: "#1B2440",
        primary: "#2F4BE0",
        positive: "#5FA777",
        attention: "#D99A3C",
        muted: "#7C879C",
      },
      fontFamily: {
        display: ["Space Grotesk", "Inter", "sans-serif"],
        body: ["Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      borderRadius: {
        ui: "8px",
      },
    },
  },
  plugins: [],
};
