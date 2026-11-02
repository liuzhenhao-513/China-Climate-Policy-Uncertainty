# Data dictionary

All CSV files are UTF-8 with BOM, comma-delimited.

## `data/labels/cpu_labels.csv` — annotated articles

| Field | Meaning |
|---|---|
| `article_id` | `news-<8-digit physical row of the raw CSV>`; joins every file in the project |
| `date`, `newspaper` | Article date (ISO) and one of the six newspapers |
| `role` | `main` = stratified candidate sample; `noncandidate` = random article without climate terms (filter check) |
| `label` | 1 = climate policy uncertainty, 0 = not (see `docs/ANNOTATION_GUIDE.md`) |
| `dimension` | For 1: A (who) / B (when, what) / C (consequences), joined by `+`; for 0: reason class |
| `confidence` | `high` / `low` |
| `label_source` | `single`, `double_agree` or `adjudicated` |
| `climate_hits`, `rule_label` | Number of climate-term matches; dictionary-rule label |
| `title`, `text` | Title and the ≤600-character focus window that was annotated |

`label_report.json` summarises the labels and agreement. `adjudication.csv` lists the 7 resolved disagreements.
`human_review_sample.csv` is the blind review sheet (fill `human_label`).

## `data/reconstructed/` — indices built by this project

`<name>_<frequency>.csv` with `frequency` ∈ {daily, monthly, annual}:

| Field | Meaning |
|---|---|
| `date` | Day; first day of month; or January 1 for annual |
| `ccpu` | Index value Z_t (unit-SD scaled per newspaper, equal-weighted, no centring) |
| `newspapers` | Number of newspapers observed in the period |
| `series_origin` | `reconstructed_from_text` |

`<name>_<frequency>_components.csv` traces every value: `period, newspaper, articles, cpu_articles, ratio, sigma,
standardized_ratio`. Variants: `ccpu_macbert` (main classifier, share), `ccpu_macbert_count` (fixed denominator),
`ccpu_tfidf[_count]`, `ccpu_rule[_count]`, `ccpu_climate`.

## `data/published/` — the authors' series

`national_{daily,monthly,annual}.csv`, `province_{monthly,annual}.csv`, and everything combined in `ccpu_final.csv`.
Fields: `date, frequency, level, region_code, region_en, region_zh, ccpu, series_origin, source_file, source_sheet,
source_excel_row, paper_period`. The workbooks are a later vintage than the paper and extend to 2026.
`paper_period = 1` marks 2000–2022.

## Intermediate files (`data/interim/`, not tracked)

| File | Rows | Content |
|---|---|---|
| `article_index.csv` | 1,755,826 | `article_id, date, newspaper, chars, climate_hits, rule_label` (index denominators) |
| `candidates.csv` | 115,192 | Candidates with `title, top_terms, focus_text` |
| `predictions_macbert.csv` | 115,192 | `article_id, date, newspaper, cpu_probability, cpu_label` |
| `predictions_tfidf.csv` | 115,192 | Same schema, TF-IDF model |
