import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          '50': '#f8f9fa',
          '100': '#1a1a2e',
          '200': '#16213e',
          '300': '#0f3460',
        },
      },
    },
  },
  plugins: [],
};

export default config;
