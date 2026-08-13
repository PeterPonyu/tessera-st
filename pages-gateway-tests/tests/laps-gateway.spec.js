import { test, expect } from '@playwright/test';
import {
  ALLOWED_PUBLISHED_ARTICLE_DOIS,
  HOMEPAGE_URL,
  LAPS_GATEWAY_SITES,
  SCPORTAL_URL,
  VIEWPORTS,
} from '../sites.mjs';
import {
  classifyArticleDoiLeak,
  extractDoiFromHref,
  findMarketingH1Leaks,
  findSubmissionPackagingLeaks,
  hasBibTeXKit,
} from '../lib/policy.mjs';

/** @type {Map<string, { status: number, ok: boolean }>} */
const githubHeadCache = new Map();

/**
 * @param {string} repoPath e.g. PeterPonyu/HetCLOP
 */
async function githubRepoIsPublic(repoPath) {
  if (githubHeadCache.has(repoPath)) return githubHeadCache.get(repoPath);
  const url = `https://github.com/${repoPath}`;
  let status = 0;
  let ok = false;
  try {
    const response = await fetch(url, {
      method: 'HEAD',
      redirect: 'follow',
      headers: { 'user-agent': 'laps-gateway-tests/1.0' },
      signal: AbortSignal.timeout(20_000),
    });
    status = response.status;
    ok = response.ok;
  } catch {
    ok = false;
  }
  const result = { status, ok };
  githubHeadCache.set(repoPath, result);
  return result;
}

/**
 * @param {import('@playwright/test').Page} page
 */
async function visibleBodyText(page) {
  return page.evaluate(() => {
    const clone = document.body.cloneNode(true);
    clone.querySelectorAll('script, style, noscript').forEach((node) => node.remove());
    return clone.innerText.replace(/\s+/g, ' ').trim();
  });
}

/**
 * @param {import('@playwright/test').Page} page
 */
async function hasHorizontalOverflow(page) {
  return page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1);
}

for (const site of LAPS_GATEWAY_SITES) {
  test.describe(`${site.label} (${site.id})`, () => {
    test('Home responds HTTP 200', async ({ page }) => {
      const response = await page.goto(site.url, { waitUntil: 'domcontentloaded' });
      expect(response?.status(), `${site.url} should return 200`).toBe(200);
    });

    test('Sticky Homepage + SCPortal chrome', async ({ page }) => {
      await page.goto(site.url, { waitUntil: 'networkidle' });
      const homepageLink = page.locator(`a[href="${HOMEPAGE_URL}"], a[href="/"], a[href="https://peterponyu.github.io"]`).first();
      const scportalLink = page.locator(`a[href="${SCPORTAL_URL}"], a[href="/scportal/"], a[href="https://peterponyu.github.io/scportal/"]`).first();
      await expect(homepageLink, 'Homepage link missing').toBeVisible();
      await expect(scportalLink, 'SCPortal link missing').toBeVisible();

      const stickyChrome = await homepageLink.evaluate((el) => {
        let node = el;
        while (node && node !== document.body) {
          const position = getComputedStyle(node).position;
          if (position === 'sticky' || position === 'fixed') return position;
          node = node.parentElement;
        }
        return null;
      });
      expect(stickyChrome, 'Homepage/SCPortal not inside sticky/fixed chrome').not.toBeNull();
    });

    for (const [viewportName, viewport] of Object.entries(VIEWPORTS)) {
      test(`Home has no horizontal overflow @ ${viewport.width}px (${viewportName})`, async ({ page }) => {
        await page.setViewportSize(viewport);
        await page.goto(site.url, { waitUntil: 'networkidle' });
        const overflow = await hasHorizontalOverflow(page);
        expect(overflow, `horizontal overflow at ${viewport.width}px`).toBe(false);
      });
    }

    test('No marketing / product splash H1', async ({ page }) => {
      await page.goto(site.url, { waitUntil: 'domcontentloaded' });
      const h1Text = await page.locator('h1').first().innerText().catch(() => '');
      const leaks = findMarketingH1Leaks(h1Text);
      expect(leaks, `marketing H1: ${h1Text}`).toEqual([]);
    });

    test('No journal-submission packaging on unpublished gateway', async ({ page }) => {
      test.skip(Boolean(site.allowBibTeXKit), 'scCCVGBen published — BibTeX allowed');

      await page.goto(site.url, { waitUntil: 'networkidle' });
      const html = await page.content();
      const text = await visibleBodyText(page);

      const packagingHits = findSubmissionPackagingLeaks(text);
      expect(packagingHits, `submission packaging leaks in visible text on ${site.url}`).toEqual([]);
      expect(hasBibTeXKit(html), 'BibTeX kit present on unpublished gateway').toBe(false);
    });

    test('scCCVGBen Frontiers DOI allowed; bioRxiv submission DOI forbidden', async ({ page }) => {
      test.skip(!site.publishedArticleDoi, 'paper leaf without published article DOI');

      await page.goto(site.url, { waitUntil: 'networkidle' });
      const html = await page.content();
      expect(html).toContain(site.publishedArticleDoi);
      expect(html).not.toContain('10.1101/2025.02.01.636000');
    });

    test('Fail-closed: no href to private GitHub repos (404)', async ({ page }) => {
      await page.goto(site.url, { waitUntil: 'networkidle' });
      const hrefs = await page.evaluate(() =>
        [...document.querySelectorAll('a[href*="github.com/PeterPonyu/"]')].map((a) => a.href),
      );

      const forbiddenRepos = new Set([
        ...(site.privateGithubRepos ?? []),
        'PeterPonyu/HetCLOP',
        'PeterPonyu/tessera-st',
      ]);

      /** @type {string[]} */
      const failures = [];
      for (const href of hrefs) {
        const match = href.match(/github\.com\/(PeterPonyu\/[^/#?]+)/i);
        if (!match) continue;
        const repo = match[1];
        if (!forbiddenRepos.has(repo)) continue;
        const { status, ok } = await githubRepoIsPublic(repo);
        if (!ok) failures.push(`${repo} linked but HEAD ${status || 'failed'} (${href})`);
      }
      expect(failures, failures.join('\n')).toEqual([]);
    });

    test('Fail-closed: no invented article DOI', async ({ page }) => {
      await page.goto(site.url, { waitUntil: 'networkidle' });
      const hrefs = await page.evaluate(() =>
        [...document.querySelectorAll('a[href*="doi.org/"]')].map((a) => a.href),
      );

      /** @type {string[]} */
      const doiLeaks = [];
      for (const href of hrefs) {
        const doi = extractDoiFromHref(href);
        if (!doi) continue;
        const leak = classifyArticleDoiLeak(doi, site, ALLOWED_PUBLISHED_ARTICLE_DOIS);
        if (leak) doiLeaks.push(`${leak}: ${doi} (${href})`);
      }
      expect(doiLeaks, doiLeaks.join('\n')).toEqual([]);
    });
  });
}
