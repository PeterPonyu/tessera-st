/**
 * Tessera-ST science gateway — unpublished spatial-domain law.
 * Fail-closed badges; no journal packaging.
 */
export const SITE = {
  slug: 'tessera-st',
  shortName: 'Tessera-ST',
  title: 'The value of a spatial prior is set by the ground truth’s spatial contiguity',
  kicker: 'ZF Lab · spatial domains',
  lead:
    'A causal law for spatial-domain detection: shuffle coordinates, hold expression and labels. Objective geometry and adjacency — not a method crown or journal companion.',
  physicalObject:
    'Tissue tessellation on a lattice: contiguous stripe domains, disc niches, and predicted seams where smoothing must stop.',
  primaryClaim:
    'Spatial-prior advantage tracks ground-truth contiguity; scrambling coordinates while holding expression and labels fixed makes advantage rise with contiguity and flip sign.',
  homepage: 'https://peterponyu.github.io/',
  scportal: 'https://peterponyu.github.io/scportal/',
  siteUrl: 'https://peterponyu.github.io/tessera-st/',
} as const;

export type BadgeConfig = {
  label: string;
  href?: string;
  enabled: boolean;
  disabledReason?: string;
};

export const BADGES = {
  code: {
    label: 'Code',
    enabled: false,
    disabledReason: 'Workbench not linked until anonymous public HTTPS 200',
  } satisfies BadgeConfig,
  site: {
    label: 'Site',
    href: SITE.siteUrl,
    enabled: true,
  } satisfies BadgeConfig,
  archive: {
    label: 'Archive',
    enabled: false,
    disabledReason: 'No public archive DOI on this leaf',
  } satisfies BadgeConfig,
  articleDoi: {
    label: 'Article DOI',
    enabled: false,
    disabledReason: 'On acceptance',
  } satisfies BadgeConfig,
} as const;

export const ROUTES = [
  { href: '/results', label: 'Results', number: '01', blurb: 'Lattice maps, seams, coordinate-shuffle intervention.' },
  { href: '/methods', label: 'Methods', number: '02', blurb: 'Synthetic 32×32 tessellation and eleven-platform panel.' },
  { href: '/evidence', label: 'Evidence', number: '03', blurb: 'Contiguity, ARI, Spearman law across platforms.' },
  { href: '/claims', label: 'Claims', number: '04', blurb: 'Falsifiable law statements and refutation hooks.' },
] as const;

export const STAT_TILES = [
  { value: '32×32', label: 'synthetic lattice' },
  { value: '11', label: 'observational platforms' },
  { value: '0.022', label: 'CODEX adjacency — scattered niches' },
  { value: '0.931', label: 'MERFISH adjacency — contiguous layers' },
] as const;

export const SPATIAL_MAPS = [
  {
    id: 'ras1',
    src: '/figures/fig_spatialmap_ras1.png',
    title: 'Map A · Ground truth',
    caption: 'Four contiguous stripe domains plus a disc niche — the tessellation the prior must recover.',
  },
  {
    id: 'ras2',
    src: '/figures/fig_spatialmap_ras2.png',
    title: 'Map B · Non-spatial KMeans (ARI 0.25)',
    caption: 'Expression without coordinates scatters domains across the lattice.',
  },
  {
    id: 'ras3',
    src: '/figures/fig_spatialmap_ras3.png',
    title: 'Map C · Neighbour-mean (ARI 0.96)',
    caption: 'Parameter-free spatial prior recovers tiles when ground truth is contiguous.',
  },
  {
    id: 'ras4',
    src: '/figures/fig_spatialmap_ras4.png',
    title: 'Map D · Predicted seams',
    caption: 'Seams as explicit lattice output — where smoothing must stop.',
  },
] as const;

export const ADJACENCY_ROWS = [
  { platform: 'MERFISH', contiguity: '0.931', geometry: 'contiguous layers', winnerAri: '0.429', spatialAdv: '+0.314' },
  { platform: 'STARmap', contiguity: '0.921', geometry: 'contiguous', winnerAri: '0.579', spatialAdv: '+0.168' },
  { platform: 'DLPFC', contiguity: '0.905', geometry: 'cortical layers', winnerAri: '0.584', spatialAdv: '+0.101' },
  { platform: 'BRCA', contiguity: '0.865', geometry: 'contiguous', winnerAri: '0.616', spatialAdv: '+0.042' },
  { platform: 'osmFISH', contiguity: '0.817', geometry: 'contiguous', winnerAri: '0.438', spatialAdv: '+0.196' },
  { platform: 'IMC', contiguity: '0.630', geometry: 'mixed', winnerAri: '0.485', spatialAdv: '+0.139' },
  { platform: 'seqFISH', contiguity: '0.621', geometry: 'mixed', winnerAri: '0.407', spatialAdv: '−0.005' },
  { platform: 'Open-ST', contiguity: '0.562', geometry: 'mixed', winnerAri: '0.264', spatialAdv: '−0.005' },
  { platform: 'Slide-seqV2', contiguity: '0.274', geometry: 'scattered', winnerAri: '0.118', spatialAdv: '+0.008' },
  { platform: 'MIBI-TOF', contiguity: '0.271', geometry: 'scattered', winnerAri: '0.247', spatialAdv: '+0.022' },
  { platform: 'CODEX', contiguity: '0.022', geometry: 'scattered niches', winnerAri: '0.081', spatialAdv: '+0.024' },
] as const;
