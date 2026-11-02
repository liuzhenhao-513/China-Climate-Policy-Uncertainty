# Full pipeline, raw newspaper CSV -> national CCPU index. Run from the repository root.
# Timings in brackets: Intel Core Ultra 7 155H laptop, CPU only, on AC power.
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

& $py run.py published                      # tidy copy of the authors' published series      [seconds]
& $py run.py build-corpus                   # 1.76M articles -> 115k climate candidates       [~1 h, disk-bound]
& $py run.py sample-labels                  # stratified annotation sample + 24 batch files   [seconds]
#   ... annotate data/labels/batches/*.md following docs/ANNOTATION_GUIDE.md
#       -> data/labels/annotations/<batch>.csv ; resolve disagreements in data/labels/adjudication.csv
& $py run.py merge-labels                   # Cohen's kappa, final labels, blind review sheet
& $py run.py split-labels                   # 70/15/15 stratified split of candidate labels
& $py run.py train-baselines                # dictionary rule + TF-IDF/LR                     [~1 min]
& $py run.py train                          # MacBERT, top 6 layers, 6 epochs                 [~70 min]
& $py scripts/check_quantization.py         # int8 vs fp32 check (int8 rejected: 75% agreement)
& $py run.py predict                        # MacBERT on 115,192 candidates (resumable)       [~3.5 h]
& $py run.py predict-tfidf                  # TF-IDF on all candidates                        [~2 min]

foreach ($m in "share", "count") {
    $s = if ($m -eq "count") { "_count" } else { "" }
    & $py run.py build-index --predictions data/interim/predictions_macbert.csv --measure $m --name "ccpu_macbert$s"
    & $py run.py build-index --predictions data/interim/predictions_tfidf.csv --measure $m --name "ccpu_tfidf$s"
    & $py run.py build-index --label-column rule_label --measure $m --name "ccpu_rule$s"
}
& $py run.py build-index --label-column climate_any --name ccpu_climate
& $py run.py report                         # outputs/validation/*.json|csv, outputs/figures/*.png

# Optional, after filling human_label in data/labels/human_review_sample.csv:
# & $py run.py review-agreement
