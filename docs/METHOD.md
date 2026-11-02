# Method

This project rebuilds the national CCPU index of Ma et al. (2023) end to end, starting from the raw newspaper text.
It follows the paper's steps and uses its classifier (MacBERT) and aggregation formula. Two inputs the authors did not
release, the 28,800 manual labels and the fine-tuned weights, are rebuilt here. Several engineering choices also make
the pipeline run on a laptop CPU.

## Pipeline

| Step | Paper (Ma et al., 2023) | This project | Code |
|---|---|---|---|
| Corpus | 1,755,826 articles, 6 newspapers, WiseNews, 2000–2022 | Same deposit (Figshare 24071310) | `text.py`, `corpus.py` |
| Cleaning | Title + body; drop punctuation, URLs, spaces | Same; WiseNews export headers also stripped, NFKC normalisation | `text.py` |
| Candidate filter | Not described | Articles with no climate term are CPU = 0; filter recall checked on a random sample | `lexicon.py` |
| Labels | 28,800 items, 48 trained auditors, 3 readers each | ~2,000 candidates + 300 non-candidates, sampled by year × newspaper; LLM-assisted annotation following `ANNOTATION_GUIDE.md`; 300 double-annotated (Cohen's κ); disagreements adjudicated; 100-item human review sheet | `annotate.py` |
| Classifier | MacBERT-base + linear head on [CLS], cross-entropy, 70/30 split | Same architecture; top 6 of 12 layers fine-tuned; class-weighted cross-entropy; warmup; 70/15/15 split; epoch and decision threshold (0.56) chosen on validation F1 | `model.py` |
| Input length | 500 tokens (released code) | 256 tokens of a *focus window*: title, lead, then passages around climate terms | `lexicon.focus_window` |
| Benchmarks | Dictionary methods (Xu et al.; Lin & Zhao), other BERTs | Dictionary rule (climate ∧ policy ∧ uncertainty), TF-IDF + logistic regression | `baselines.py` |
| Index | X_it = CPU share; Y_it = X_it / sd_T(X_i); Z_t = mean_i Y_it | Same, at daily, monthly and annual frequency; T = 2000–2022 | `index.py` |
| Missing newspaper-months | Not described | 60 of 1,656 paper-months are absent from the deposit; Z_t averages the papers observed in month t | `index.py` |
| Validation | Correlation with alternative indices | Pearson / Spearman / RMSE vs the published series | `index.compare_series`, `report.py` |

## Why a candidate filter

On a CPU, MacBERT classifies about 10–20 articles per second. Running it on all 1.76M articles would take more than
a day. Fewer than 10% of articles contain any climate-related term, and an article that never mentions climate,
carbon or emissions cannot be about climate policy uncertainty. Restricting the transformer to candidates cuts the
cost by an order of magnitude. The assumption is tested, not just asserted: 300 random non-candidate articles are
annotated with the rest, and their CPU rate is reported in `data/labels/label_report.json`.

## Why a focus window

Chinese BERT tokenises roughly one character per token, so a 256-token model reads about 250 characters. In long
articles, the climate-policy passage often sits well below the lead. The focus window keeps the title and the first
100 characters, then appends ±60-character passages around climate terms. Annotators read up to 600 characters of
the same window; the model reads its first 256 tokens. As in the paper, humans see more context than the model.

## Labels: what they are and what they are not

The labels are this project's own. They come from an explicit guideline that operationalises the paper's definition
(uncertainty about *who* makes climate policy, *when/what* policy is adopted, and its *consequences*). They were
produced with LLM assistance and checked for consistency with double annotation. They are not the authors' 28,800
human labels, so differences between the reconstructed and published indices partly reflect differences in label
definitions. The author blind-reviewed 100 of the labelled articles (`data/labels/human_review_sample.csv`):
70% agreement, Cohen's κ = 0.40, with the human reading broader than the labels (README, "About the labels").

## Corpus composition

The deposit is not a constant fraction of each newspaper's output:

| Year | People's Daily articles | climate-term share | China News Service articles | climate-term share |
|---|---:|---:|---:|---:|
| 2005 | 8,469 | 3% | 7,734 | 4% |
| 2008 | 8,106 | 14% | 5,597 | 11% |
| 2010 | 39,674 | 6% | 9,032 | 49% |
| 2015 | 37,490 | 4% | 2,586 | 46% |
| 2022 | 6,048 | 26% | 19,083 | 17% |

Parts of the deposit look like full exports and parts like keyword-search results. With the deposit's own article
counts as N_all, the CPU share (paper formula) mostly tracks these export decisions. `--measure count` fixes each
newspaper's denominator. After dividing by sd_T, this gives Y_it = N_cpu_it / sd_T(N_cpu_i), which is what the paper's
formula yields if N_all is the newspaper's full, slowly changing output. Total deposit volume alone correlates 0.69
with the published monthly index. Half of the published series' ten highest months are Marches (the National People's
Congress), so the published construction appears to depend on denominators that were not released. The count version
is still affected by the climate-enriched 2007–2009 exports.

## Deviations that matter for interpretation

* **Vintage.** The published workbooks are later updates (to 2026). The comparison uses 2000–2022 only, but the authors
  may have renormalised the updated series.
* **Sequence length, epochs and frozen layers** were set for CPU training (`config.json`). On a GPU, set `max_length` to
  500 and `freeze_layers` to 0, and increase the number of epochs.
* **MacBERT vs TF-IDF.** On this budget MacBERT (test F1 0.70) does not beat TF-IDF + LR (0.77). With 52 test positives,
  the difference is within about 1.5 standard errors.
* **Regional indices** are not reconstructed. The paper does not say how articles are assigned to provinces or cities.
