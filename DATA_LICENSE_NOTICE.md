# Data sources, permissions and redistribution

**Release:** FIFA World Cup Intelligence, AQX Sports Analytics Data Bowl 4.0 submission (October 26, 2026).

This file documents upstream dataset permissions separately from the project's own software license. It is a provenance and risk notice, not a claim of FIFA endorsement or a legal opinion.

## 1. Data included in this public release

| Data | Source | License / status | Public-release treatment |
|---|---|---|---|
| World Cup match, squad and tournament JSON (1930–2026) | [OpenFootball / `worldcup.json`](https://github.com/openfootball/worldcup.json) | **CC0 1.0 Universal (public-domain dedication)**; see the preserved `data/raw/openfootball/LICENSE.md` | **Included**, with source and original license notice |
| Cleaned match, team-year, tournament, feature, Elo and derived analytic outputs | Generated locally from the bundled OpenFootball JSON using the project's scripts/notebooks | Derived from the CC0 data, with project computations and assumptions | **Included** for reproduction; these are not independent official FIFA statistics |
| Local country-boundary data (`assets/world_lowres.json`) | Identified by this project as a simplified **Natural Earth** boundary extract | Natural Earth mapping data is [public domain](https://www.naturalearthdata.com/about/terms-of-use/); the exact transformation/source file of this local copy has not been independently traced | Included; keep this mapping attribution and verify source chain before separate redistribution as a dataset |

OpenFootball: https://github.com/openfootball/worldcup.json

CC0: https://creativecommons.org/publicdomain/zero/1.0/

Natural Earth: https://www.naturalearthdata.com/about/terms-of-use/

## 2. Kaggle dataset used in earlier local work — not distributed

**Source:** Sourav Banerjee, [FIFA - Football World Cup Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/fifa-football-world-cup-dataset).

- The public Kaggle listing is free to access/download, but its license field reads **“Other (specified in description).”** The page's description refers to FIFA as the data source without identifying a concrete reuse/redistribution license or permission grant.
- **Public availability is not proof of permission to redistribute a third-party dataset.** This project does not assume that access, citation, or educational intent substitutes for a license.
- Therefore, **no original Kaggle CSVs or Kaggle-derived standings/summary tables are included** in the submitted archive. The previously used table equivalents were rebuilt from OpenFootball and can be regenerated using `notebooks/01_data_cleaning.ipynb` and `src/openfootball_summaries.py`.
- If the Kaggle data is reintroduced into a public version, first obtain and document explicit permission or reliable upstream licensing for those particular files, including the right to publish transformations.

The dataset is credited for research provenance only; it is **not** an active data dependency of this public release.

## 3. Code, visualization dependencies and marks

- The project's authored Python and UI source is provided under its root **MIT `LICENSE`**. That license **does not relicense third-party data, software, graphics or brands**.
- The globe and 3D pitch dynamically load **Three.js 0.149.0** from cdnjs; Three.js uses the MIT license. External JavaScript is not bundled. Other Python libraries retain their own respective licenses.
- “FIFA” and “World Cup” identify the sports competition discussed. This is an independent educational sports-analytics project, **not affiliated with or endorsed by FIFA**. CC0 does not grant trademark rights.

## 4. Accuracy / interpretation notice

- Team-year results are inferred from interpretable **regulation-time** match records and are **not FIFA's official standings**.
- Winners/titles may depend on extra time or penalty shootouts; 90-minute goal trends and model targets intentionally use a different score definition.
- Historical match metadata can be incomplete; the model's 2026 results are a **retrospective evaluation**, not live match forecasts or betting advice.

**Summary:** OpenFootball's clearly stated CC0 terms support the public prototype. The Kaggle source's redistribution permission remains unconfirmed, so the release excludes its files. Preserve the two upstream provenance links and the bundled CC0 notice.
