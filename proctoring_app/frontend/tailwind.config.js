/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        status: {
          normal: '#22c55e',
          suspicious: '#eab308',
          cheating: '#ef4444',
        },
      },
    },
  },
  plugins: [],
}
