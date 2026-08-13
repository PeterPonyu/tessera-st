/** @typedef {'paper-leaf' | 'scCCVGBen-atlas' | 'scCCVGBen-companion'} SiteKind */

/**
 * Live LAPS paper-gateway surfaces (2026-08-13 deploy table).
 * @type {Array<{
 *   id: string,
 *   label: string,
 *   url: string,
 *   kind: SiteKind,
 *   publishedArticleDoi?: string,
 *   allowBibTeXKit?: boolean,
 *   privateGithubRepos?: string[],
 * }>}
 */
export const LAPS_GATEWAY_SITES = [
  {
    id: 'hetclop',
    label: 'HetCLOP',
    url: 'https://peterponyu.github.io/HetCLOP/',
    kind: 'paper-leaf',
    privateGithubRepos: ['PeterPonyu/HetCLOP'],
  },
  {
    id: 'dpmm',
    label: 'PanODE-DPMM',
    url: 'https://peterponyu.github.io/PanODE-DPMM/',
    kind: 'paper-leaf',
  },
  {
    id: 'topic',
    label: 'PanODE-Topic',
    url: 'https://peterponyu.github.io/PanODE-Topic/',
    kind: 'paper-leaf',
  },
  {
    id: 'mocoo',
    label: 'MoCoO',
    url: 'https://peterponyu.github.io/MoCoO/',
    kind: 'paper-leaf',
  },
  {
    id: 'b1',
    label: 'B1 cofolding',
    url: 'https://peterponyu.github.io/b1-site/',
    kind: 'paper-leaf',
  },
  {
    id: 'a3',
    label: 'A3 PLM boundary',
    url: 'https://peterponyu.github.io/a3-site/',
    kind: 'paper-leaf',
  },
  {
    id: 'b28',
    label: 'B28 proteomics transfer',
    url: 'https://peterponyu.github.io/b28-site/',
    kind: 'paper-leaf',
  },
  {
    id: 'spgd',
    label: 'SPGD deconv',
    url: 'https://peterponyu.github.io/SPGD-site/',
    kind: 'paper-leaf',
  },
  {
    id: 'tessera',
    label: 'Tessera ST',
    url: 'https://peterponyu.github.io/tessera-st/',
    kind: 'paper-leaf',
    privateGithubRepos: ['PeterPonyu/tessera-st'],
  },
  {
    id: 'scccvgben-atlas',
    label: 'scCCVGBen Hugo atlas',
    url: 'https://peterponyu.github.io/scCCVGBen/',
    kind: 'scCCVGBen-atlas',
    publishedArticleDoi: '10.3389/fgene.2026.1822168',
    allowBibTeXKit: true,
  },
  {
    id: 'scccvgben-next',
    label: 'scCCVGBen Next companion',
    url: 'https://peterponyu.github.io/scccvgben-next/',
    kind: 'scCCVGBen-companion',
    publishedArticleDoi: '10.3389/fgene.2026.1822168',
    allowBibTeXKit: true,
  },
];

export const HOMEPAGE_URL = 'https://peterponyu.github.io/';
export const SCPORTAL_URL = 'https://peterponyu.github.io/scportal/';

/** Frontiers fact DOI — only published article DOI allowed on scCCVGBen surfaces. */
export const ALLOWED_PUBLISHED_ARTICLE_DOIS = new Set(['10.3389/fgene.2026.1822168']);

/** Zenodo archive prefix — always allowed when real. */
export const ZENODO_DOI_PREFIX = '10.5281/';

/** bioRxiv preprint cited as submission packaging — forbidden on all LAPS gateways. */
export const FORBIDDEN_BIORXIV_DOI = '10.1101/2025.02.01.636000';

export const VIEWPORTS = {
  desktop: { width: 1280, height: 800 },
  mobile: { width: 390, height: 844 },
};
