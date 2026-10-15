import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// Tests run without the Tailwind/Vite app plugins: the components under test
// only need the React transform and a DOM environment.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/setupTests.js'],
    include: ['src/**/*.test.{js,jsx}'],
  },
})
