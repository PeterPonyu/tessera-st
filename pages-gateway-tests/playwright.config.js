import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 2 : undefined,
  reporter: [
    ['list'],
    ['html', { open: 'never', outputFolder: 'playwright-report' }],
    ['json', { outputFile: 'test-results/results.json' }],
  ],
  use: {
    baseURL: 'https://peterponyu.github.io',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'off',
  },
  timeout: 60_000,
  expect: {
    timeout: 15_000,
  },
});
