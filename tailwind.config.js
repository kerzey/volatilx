/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./templates/**/*.html",
    "./action_center2/**/*.{ts,tsx}",
    "./report_center_frontend/**/*.{ts,tsx}",
    "./signin_frontend/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  corePlugins: {
    preflight: false,
  },
};
