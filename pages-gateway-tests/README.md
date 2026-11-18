# Tests of the Tessera-ST Pages leaf (Playwright)

Browser tests for this repository's GitHub Pages leaf only (`https://peterponyu.github.io/tessera-st/`). Other sites
are not crawled.

## What is checked

| Check | Rule |
|---|---|
| Home | HTTP 200 |
| Heading | The main heading is descriptive, not a marketing phrase |
| Content | No result values; the list of checked tokens is in `sites.mjs` |
| Retired routes | `/results/`, `/methods/`, `/evidence/` and `/claims/` return the custom 404 page |
| Header | Sticky header with links to Homepage and SCPortal |
| Layout | No horizontal overflow at 1280 px and 390 px |
| Citation and venue text | No journal or citation material on the page |
| Links | The public `PeterPonyu/tessera-st` repository; no link to a listed private repository; no article DOI |

## Run locally

```bash
cd pages-gateway-tests
npm ci
npx playwright install chromium
npm test
```
