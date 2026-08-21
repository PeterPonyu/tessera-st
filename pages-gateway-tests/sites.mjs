/** Live Tessera-ST Pages leaf only. Sibling result sites are out of this repo's CI. */

export const TESSERA_PAGES_URL = 'https://peterponyu.github.io/tessera-st/';
export const HOMEPAGE_URL = 'https://peterponyu.github.io/';
export const SCPORTAL_URL = 'https://peterponyu.github.io/scportal/';
export const TESSERA_GITHUB_URL = 'https://github.com/PeterPonyu/tessera-st';

/** Retired science-gateway routes. Must stay custom 404, not a results page. */
export const RETIRED_ROUTES = [
  'https://peterponyu.github.io/tessera-st/results/',
  'https://peterponyu.github.io/tessera-st/methods/',
  'https://peterponyu.github.io/tessera-st/evidence/',
  'https://peterponyu.github.io/tessera-st/claims/',
];

/**
 * Unpublished-result tokens that must not appear on the public leaf.
 * Align with `.github/workflows/pages.yml` fail-closed grep.
 */
export const LEAK_TOKENS = [
  '0.022',
  '0.931',
  'Spearman',
  'unpublished results',
  'causal law',
  '32×32',
  '32x32',
  'ARI 0.',
];

/** GitHub's stock project-Pages 404 copy — custom 404.html must not look like this. */
export const GITHUB_DEFAULT_404_SNIPPETS = [
  "There isn't a GitHub Pages site here",
  'with GitHub Pages you can easily',
];

export const ALLOWED_PUBLISHED_ARTICLE_DOIS = new Set([]);

export const VIEWPORTS = {
  desktop: { width: 1280, height: 800 },
  mobile: { width: 390, height: 844 },
};
