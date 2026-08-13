# LAPS Pages gateway tests (Playwright)

Browser policy gate for the ten LAPS GitHub Pages science gateways. Fails CI on submission packaging, marketing splash H1, missing series chrome, horizontal overflow, private GitHub hrefs, and invented article DOIs.

## Policy (fail-closed)

| Check | Rule |
|---|---|
| Submission packaging | No venue-intended, under review, PeerJ/JPR/JCIM/bioRxiv **as submission**, or BibTeX kits on unpublished leaves |
| Published exception | scCCVGBen Frontiers `10.3389/fgene.2026.1822168` allowed |
| Marketing H1 | No product splash headlines |
| Chrome | Sticky header with Homepage + SCPortal |
| Layout | No horizontal overflow on Home @ 1280 and 390 |
| Links | No `href` to private GitHub repos that 404 anonymously |
| DOI | No invented journal DOI; Zenodo archives OK |

## Run locally

```bash
cd pages-gateway-tests
npm ci
npx playwright install chromium
npm test
```

## Copy into a Pages repo

1. Copy this directory to `<pages-repo>/pages-gateway-tests/`.
2. Copy `.github/workflows/laps-gateway-tests.yml` to `<pages-repo>/.github/workflows/`.
3. Push — workflow runs on PR/push, daily schedule, after Pages deploy, or manual dispatch.

The canonical copy lives under `labs/pages-gateway-tests/`. `b1-site` hosts the live Actions run for the series.
