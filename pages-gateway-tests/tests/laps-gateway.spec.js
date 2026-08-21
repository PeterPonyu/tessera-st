import { test, expect } from '@playwright/test';
import {
  ALLOWED_PUBLISHED_ARTICLE_DOIS,
  GITHUB_DEFAULT_404_SNIPPETS,
  HOMEPAGE_URL,
  LEAK_TOKENS,
  RETIRED_ROUTES,
  SCPORTAL_URL,
  TESSERA_GITHUB_URL,
  TESSERA_PAGES_URL,
  VIEWPORTS,
} from '../sites.mjs';
import {
  classifyArticleDoiLeak,
  extractDoiFromHref,
  findMarketingH1Leaks,
  findSubmissionPackagingLeaks,
  hasBibTeXKit,
} from '../lib/policy.mjs';

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

/**
 * @param {string} haystack
 * @returns {string[]}
 */
function findResultLeaks(haystack) {
  const hits = [];
  const lower = haystack.toLowerCase();
  for (const token of LEAK_TOKENS) {
    if (lower.includes(token.toLowerCase())) hits.push(token);
  }
  if (haystack.includes('ρ') || haystack.includes('ϱ')) hits.push('ρ');
  if (/(^|[^a-z])rho([^a-z]|$)/i.test(haystack)) hits.push('rho');
  return hits;
}

test.describe('Tessera-ST Pages landing', () => {
  test('Home responds HTTP 200', async ({ page }) => {
    const response = await page.goto(TESSERA_PAGES_URL, { waitUntil: 'domcontentloaded' });
    expect(response?.status(), `${TESSERA_PAGES_URL} should return 200`).toBe(200);
  });

  test('Home is a code leaf: no unpublished-result tokens', async ({ page }) => {
    await page.goto(TESSERA_PAGES_URL, { waitUntil: 'domcontentloaded' });
    const html = await page.content();
    const text = await visibleBodyText(page);
    expect(findResultLeaks(`${text}\n${html}`), 'unpublished result tokens on landing').toEqual([]);
  });

  test('Sticky Homepage + SCPortal chrome', async ({ page }) => {
    await page.goto(TESSERA_PAGES_URL, { waitUntil: 'networkidle' });
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
      await page.goto(TESSERA_PAGES_URL, { waitUntil: 'networkidle' });
      const overflow = await hasHorizontalOverflow(page);
      expect(overflow, `horizontal overflow at ${viewport.width}px`).toBe(false);
    });
  }

  test('No marketing / product splash H1', async ({ page }) => {
    await page.goto(TESSERA_PAGES_URL, { waitUntil: 'domcontentloaded' });
    const h1Text = await page.locator('h1').first().innerText().catch(() => '');
    const leaks = findMarketingH1Leaks(h1Text);
    expect(leaks, `marketing H1: ${h1Text}`).toEqual([]);
  });

  test('No journal-submission packaging', async ({ page }) => {
    await page.goto(TESSERA_PAGES_URL, { waitUntil: 'networkidle' });
    const html = await page.content();
    const text = await visibleBodyText(page);
    expect(findSubmissionPackagingLeaks(text), `submission packaging leaks on ${TESSERA_PAGES_URL}`).toEqual([]);
    expect(hasBibTeXKit(html), 'BibTeX kit present on unpublished leaf').toBe(false);
  });

  test('Public code href is the tessera-st GitHub repo', async ({ page }) => {
    await page.goto(TESSERA_PAGES_URL, { waitUntil: 'networkidle' });
    const codeLink = page.locator(`a[href="${TESSERA_GITHUB_URL}"]`).first();
    await expect(codeLink, 'Code link to PeterPonyu/tessera-st missing').toBeVisible();
  });

  test('Fail-closed: no href to private GitHub repos (404)', async ({ page }) => {
    await page.goto(TESSERA_PAGES_URL, { waitUntil: 'networkidle' });
    const hrefs = await page.evaluate(() =>
      [...document.querySelectorAll('a[href*="github.com/PeterPonyu/"]')].map((a) => a.href),
    );

    const forbiddenRepos = new Set(['PeterPonyu/HetCLOP']);
    /** @type {string[]} */
    const failures = [];
    for (const href of hrefs) {
      const match = href.match(/github\.com\/(PeterPonyu\/[^/#?]+)/i);
      if (!match) continue;
      const repo = match[1];
      if (!forbiddenRepos.has(repo)) continue;
      failures.push(`${repo} must not be linked from this leaf (${href})`);
    }
    expect(failures, failures.join('\n')).toEqual([]);
  });

  test('Fail-closed: no invented article DOI', async ({ page }) => {
    await page.goto(TESSERA_PAGES_URL, { waitUntil: 'networkidle' });
    const hrefs = await page.evaluate(() =>
      [...document.querySelectorAll('a[href*="doi.org/"]')].map((a) => a.href),
    );

    /** @type {string[]} */
    const doiLeaks = [];
    for (const href of hrefs) {
      const doi = extractDoiFromHref(href);
      if (!doi) continue;
      const leak = classifyArticleDoiLeak(doi, {}, ALLOWED_PUBLISHED_ARTICLE_DOIS);
      if (leak) doiLeaks.push(`${leak}: ${doi} (${href})`);
    }
    expect(doiLeaks, doiLeaks.join('\n')).toEqual([]);
  });

  for (const url of RETIRED_ROUTES) {
    test(`Retired route custom 404: ${new URL(url).pathname}`, async ({ page }) => {
      const response = await page.goto(url, { waitUntil: 'domcontentloaded' });
      expect(response?.status(), `${url} must stay 404, not a restored results page`).toBe(404);
      const html = await page.content();
      const text = await visibleBodyText(page);
      const combined = `${text}\n${html}`;
      expect(combined, `${url} missing custom retired copy`).toMatch(/path retired|not a results route/i);
      for (const snippet of GITHUB_DEFAULT_404_SNIPPETS) {
        expect(combined.toLowerCase(), `${url} looks like GitHub default 404`).not.toContain(snippet.toLowerCase());
      }
      expect(findResultLeaks(combined), `unpublished result tokens on ${url}`).toEqual([]);
    });
  }
});
