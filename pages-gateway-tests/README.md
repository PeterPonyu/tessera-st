# Tessera-ST Pages landing tests (Playwright)

Browser policy gate for **this repo's** GitHub Pages leaf only (`https://peterponyu.github.io/tessera-st/`). It does **not** crawl sibling paper sites.

## Contract

| Check | Rule |
|---|---|
| Home | HTTP 200 code-description leaf |
| Leak tokens | No `0.022` / `0.931` / ρ / Spearman / unpublished-results copy |
| Retired routes | `/results/` `/methods/` `/evidence/` `/claims/` stay custom 404 (`path retired`) |
| Chrome | Sticky header with Homepage + SCPortal |
| Layout | No horizontal overflow on Home @ 1280 and 390 |
| Packaging | No venue-intended / under review / BibTeX kit |
| Links | Public `PeterPonyu/tessera-st`; no HetCLOP href; no invented article DOI |

## Run locally

```bash
cd pages-gateway-tests
npm ci
npx playwright install chromium
npm test
```
