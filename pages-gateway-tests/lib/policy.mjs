/** Submission / venue packaging phrases on unpublished gateways. */
export const SUBMISSION_PACKAGING_PATTERNS = [
  /\bunder review\b/i,
  /\bsubmitted to\b/i,
  /\bmanuscript under\b/i,
  /\bpeerj portal\b/i,
  /\bsubmit to peerj\b/i,
  /\bcite this manuscript\b/i,
  /\breviewer faq\b/i,
  /\bportal inputs\b/i,
  /\bbiorxiv preprint\b/i,
  /\bbiorxiv doi\b/i,
];

/** Venue-intended markup (e.g. `<span class="venue">· JCIM intended</span>`). */
export const VENUE_INTENDED_PATTERN =
  /\b(?:peerj|jpr|jcim|jcbb|mdpi|proteins|tcbb|jbhi)\s+intended\b/i;

/** bioRxiv preprint cited as submission packaging — forbidden on all LAPS gateways. */
export const FORBIDDEN_BIORXIV_DOI = '10.1101/2025.02.01.636000';

/** Marketing / product splash H1 — not descriptive paper titles. */
export const MARKETING_H1_PATTERNS = [
  /^welcome to\b/i,
  /^discover\b/i,
  /^introducing\b/i,
  /^the (?:future|next generation|ultimate|complete)\b/i,
  /\bplatform\b.*\b(?:solution|suite)\b/i,
  /\bsupercharge\b/i,
  /\brevolutioniz/i,
  /\btransform your\b/i,
  /\bunlock the power\b/i,
  /^your .* hub$/i,
  /^the .* platform$/i,
];

export const BIBTEX_KIT_PATTERNS = [
  /\bcopy bibtex\b/i,
  /\bbibtex kit\b/i,
  /\bpre-publication bibtex\b/i,
  /data-copy=["']#bibtex["']/i,
  /id=["']bibtex["']/i,
];

/**
 * @param {string} text
 * @returns {string[]}
 */
export function findSubmissionPackagingLeaks(text) {
  const hits = [];
  for (const pattern of SUBMISSION_PACKAGING_PATTERNS) {
    if (pattern.test(text)) hits.push(pattern.source);
  }
  if (VENUE_INTENDED_PATTERN.test(text)) hits.push('venue_intended');
  return hits;
}

/**
 * @param {string} h1Text
 * @returns {string[]}
 */
export function findMarketingH1Leaks(h1Text) {
  const trimmed = h1Text.trim();
  if (!trimmed) return ['empty_h1'];
  const hits = [];
  for (const pattern of MARKETING_H1_PATTERNS) {
    if (pattern.test(trimmed)) hits.push(pattern.source);
  }
  return hits;
}

/**
 * @param {string} html
 * @returns {boolean}
 */
export function hasBibTeXKit(html) {
  return BIBTEX_KIT_PATTERNS.some((pattern) => pattern.test(html));
}

/**
 * @param {string} href
 * @returns {string | null} normalized DOI without https://doi.org/
 */
export function extractDoiFromHref(href) {
  try {
    const url = new URL(href);
    if (url.hostname.replace(/^www\./, '') !== 'doi.org') return null;
    return decodeURIComponent(url.pathname.replace(/^\//, ''));
  } catch {
    return null;
  }
}

/**
 * @param {string} doi
 * @param {{ publishedArticleDoi?: string, allowBibTeXKit?: boolean }} site
 * @param {Set<string>} allowedPublished
 * @returns {string | null} reason if forbidden
 */
export function classifyArticleDoiLeak(doi, site, allowedPublished) {
  if (!doi) return null;
  if (doi.startsWith('10.5281/')) return null;
  if (doi === FORBIDDEN_BIORXIV_DOI) return 'forbidden_biorxiv_submission_doi';
  if (allowedPublished.has(doi)) return null;
  if (site.publishedArticleDoi && doi === site.publishedArticleDoi) return null;
  if (/^10\.1101\//.test(doi)) return 'biorxiv_article_doi_on_unpublished_gateway';
  if (/^10\.3389\//.test(doi) || /^10\.1002\//.test(doi) || /^10\.1021\//.test(doi)) {
    return 'invented_or_unpublished_journal_doi';
  }
  if (/^10\.\d{4,}\/(?!5281)/.test(doi)) {
    return 'invented_or_unpublished_journal_doi';
  }
  return null;
}
