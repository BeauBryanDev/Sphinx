/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        "egypt-gold":      "#d4a64a",
        "egypt-gold-dark": "#8a6a20",
        "egypt-sand":      "#c89b5a",
        "egypt-stone":     "#3a2a14",
      },
      fontFamily: {
        "serif-egypt": ["Cinzel", "Trajan Pro", "Times New Roman", "serif"],
      },
    },
  },
  plugins: [],
};
