"""Command line entry points. Run `python run.py --help` without installation."""
import argparse
import json
from pathlib import Path
from .common import ROOT, read_config

INTERIM = ROOT / "data" / "interim"
LABELS = ROOT / "data" / "labels"
OUTPUTS = ROOT / "outputs"


def keep_log(path, name):
    """Copy a run log into outputs/logs so it is versioned with the results."""
    import shutil
    (OUTPUTS / "logs").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, OUTPUTS / "logs" / name)


def main(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    preliminary, _ = config_parser.parse_known_args(argv)
    cfg = read_config(preliminary.config)

    parser = argparse.ArgumentParser(description="China climate policy uncertainty (CCPU) index from newspaper text")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--source-dir", type=Path, help="Directory containing the raw news CSV and Excel files")
    commands = parser.add_subparsers(dest="command", required=True)

    # 0. published series and raw-data audit
    published = commands.add_parser("published", help="Export the authors' Excel series to tidy CSVs")
    published.add_argument("--output", type=Path, default=ROOT / "data" / "published")
    published.add_argument("--report", type=Path, default=OUTPUTS / "published_audit.json")
    audit = commands.add_parser("audit-news", help="Profile the raw news CSV (counts, coverage, parse issues)")
    audit.add_argument("--input", type=Path)
    audit.add_argument("--limit", type=int)
    audit.add_argument("--output", type=Path, default=OUTPUTS)

    # 1. corpus
    corpus = commands.add_parser("build-corpus", help="Stream the news CSV once: article index, climate candidates, filter-check sample")
    corpus.add_argument("--input", type=Path)
    corpus.add_argument("--output", type=Path, default=INTERIM)
    corpus.add_argument("--start", default=cfg["paper_start"])
    corpus.add_argument("--end", default=cfg["paper_end"])
    corpus.add_argument("--limit", type=int)

    # 2. annotation
    sample = commands.add_parser("sample-labels", help="Draw the stratified annotation sample and write batch files")
    sample.add_argument("--candidates", type=Path, default=INTERIM / "candidates.csv")
    sample.add_argument("--noncandidates", type=Path, default=INTERIM / "noncandidate_sample.csv")
    sample.add_argument("--output", type=Path, default=LABELS)
    sample.add_argument("--n-main", type=int, default=2000)
    sample.add_argument("--n-noncandidate", type=int, default=300)
    sample.add_argument("--n-double", type=int, default=300)
    merge = commands.add_parser("merge-labels", help="Combine annotator files, compute Cohen's kappa, apply adjudication")
    merge.add_argument("--pool", type=Path, default=LABELS / "annotation_pool.csv")
    merge.add_argument("--annotations", type=Path, default=LABELS / "annotations")
    merge.add_argument("--adjudication", type=Path, default=LABELS / "adjudication.csv")
    merge.add_argument("--output", type=Path, default=LABELS)
    review = commands.add_parser("review-agreement", help="Agreement between the blind human review labels and the training labels")
    review.add_argument("--input", type=Path, default=LABELS / "human_review_sample.csv")
    split = commands.add_parser("split-labels", help="Stratified train/validation/test split")
    split.add_argument("--input", type=Path, default=LABELS / "cpu_labels.csv")
    split.add_argument("--output", type=Path, default=LABELS / "splits")
    split.add_argument("--seed", type=int, default=cfg["seed"])
    split.add_argument("--train-fraction", type=float, default=0.7)
    split.add_argument("--val-fraction", type=float, default=0.15)

    # 3. classifiers
    train = commands.add_parser("train", help="Fine-tune MacBERT")
    predict = commands.add_parser("predict", help="Classify candidate articles with a fine-tuned MacBERT")
    for sub in (train, predict):
        sub.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0")
        sub.add_argument("--max-length", type=int, default=cfg["max_length"])
    train.add_argument("--base-model", default=cfg["base_model"])
    train.add_argument("--splits", type=Path, default=LABELS / "splits")
    train.add_argument("--output", type=Path, default=ROOT / "models" / "macbert-ccpu")
    train.add_argument("--epochs", type=int, default=cfg["epochs"])
    train.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    train.add_argument("--learning-rate", type=float, default=cfg["learning_rate"])
    train.add_argument("--label-smoothing", type=float, default=cfg["label_smoothing"])
    train.add_argument("--seed", type=int, default=cfg["seed"])
    train.add_argument("--freeze-layers", type=int, default=cfg.get("freeze_layers", 0),
                       help="Freeze embeddings + the lowest N of 12 encoder layers (0 = full fine-tuning)")
    predict.add_argument("--articles", type=Path, default=INTERIM / "candidates.csv")
    predict.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "macbert-ccpu")
    predict.add_argument("--base-model", help="Only for original .pth.tar weights")
    predict.add_argument("--batch-size", type=int, default=32)
    predict.add_argument("--output", type=Path, default=INTERIM / "predictions_macbert.csv")
    predict.add_argument("--quantize", action="store_true", help="int8 dynamic quantization for faster CPU inference")
    baselines = commands.add_parser("train-baselines", help="TF-IDF + logistic regression and dictionary rule on the same split")
    baselines.add_argument("--splits", type=Path, default=LABELS / "splits")
    baselines.add_argument("--output", type=Path, default=ROOT / "models" / "baselines")
    tfidf = commands.add_parser("predict-tfidf", help="Classify candidates with the TF-IDF model")
    tfidf.add_argument("--model", type=Path, default=ROOT / "models" / "baselines" / "tfidf_lr.pkl")
    tfidf.add_argument("--articles", type=Path, default=INTERIM / "candidates.csv")
    tfidf.add_argument("--output", type=Path, default=INTERIM / "predictions_tfidf.csv")

    # 4. index
    index = commands.add_parser("build-index", help="Aggregate article labels into national daily/monthly/annual CCPU")
    index.add_argument("--article-index", type=Path, default=INTERIM / "article_index.csv")
    index.add_argument("--predictions", type=Path, help="Candidate predictions; omit to use --label-column of the article index")
    index.add_argument("--label-column", default="cpu_label")
    index.add_argument("--name", default="ccpu_macbert")
    index.add_argument("--output", type=Path, default=ROOT / "data" / "reconstructed")
    index.add_argument("--normalization-start", default=cfg["paper_start"])
    index.add_argument("--normalization-end", default=cfg["paper_end"])
    index.add_argument("--ddof", type=int, choices=[0, 1], default=cfg["standard_deviation_ddof"])
    index.add_argument("--missing", choices=["available", "strict"], default=cfg["missing_policy"])
    index.add_argument("--measure", choices=["share", "count"], default="share",
                       help="share = paper formula N_cpu/N_all; count = fixed per-paper denominator")
    compare = commands.add_parser("compare", help="Correlation/MAE/RMSE against the published series")
    compare.add_argument("--reconstructed", type=Path, required=True)
    compare.add_argument("--published", type=Path, default=ROOT / "data" / "published" / "national_monthly.csv")
    compare.add_argument("--frequency", choices=["monthly", "annual", "daily"], default="monthly")
    compare.add_argument("--output", type=Path, default=OUTPUTS / "validation")
    compare.add_argument("--name", default="comparison")
    report = commands.add_parser("report", help="Validation tables and figures for all reconstructed indices")
    report.add_argument("--output", type=Path, default=OUTPUTS)


    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    if args.source_dir:
        cfg["source_dir"] = str(args.source_dir.resolve())
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    news = lambda: getattr(args, "input", None) or Path(cfg["source_dir"]) / cfg["news_file"]

    if args.command == "published":
        from .published import build_published
        result = build_published(cfg, args.output, args.report)
        result = {k: result[k] for k in ("rows", "paper_period_rows", "duplicate_keys", "invalid_or_missing_ccpu")}
    elif args.command == "audit-news":
        from .audit import audit_news
        result = audit_news(news(), args.output, args.limit)
    elif args.command == "build-corpus":
        from .corpus import build_corpus
        result = build_corpus(news(), args.output, args.start, args.end, args.limit, seed=cfg["seed"])
    elif args.command == "sample-labels":
        from .annotate import design_sample
        result = design_sample(args.candidates, args.noncandidates, args.output, args.n_main,
                               args.n_noncandidate, args.n_double, seed=cfg["seed"])
    elif args.command == "merge-labels":
        from .annotate import merge_labels
        result = merge_labels(args.pool, args.annotations, args.output, args.adjudication)
    elif args.command == "review-agreement":
        from .annotate import review_agreement
        result = review_agreement(args.input)
    elif args.command == "split-labels":
        from .labels import split_labels
        result = split_labels(args.input, args.output, args.seed, args.train_fraction, args.val_fraction)
    elif args.command in ("train", "predict"):
        try:
            from .model import train_model, predict_articles
        except ImportError as exc:
            parser.error(f"Model dependencies unavailable: {exc}. Install requirements-model.txt.")
        if args.command == "train":
            s = args.splits
            result = train_model(s / "train.csv", s / "validation.csv", args.output, args.base_model, s / "test.csv",
                                 args.device, args.epochs, args.batch_size, args.learning_rate,
                                 args.max_length, args.seed, args.label_smoothing, args.freeze_layers)
            keep_log(args.output / "train.log", "train_macbert.log")
        else:
            result = {"predictions": predict_articles(args.articles, args.checkpoint, args.output,
                                                      args.base_model, args.device, args.batch_size, args.max_length,
                                                      args.quantize)}
            keep_log(args.output.with_suffix(".log"), "predict_macbert.log")
    elif args.command == "train-baselines":
        from .baselines import train_tfidf
        s = args.splits
        result = train_tfidf(s / "train.csv", s / "validation.csv", s / "test.csv", args.output, cfg["seed"])
    elif args.command == "predict-tfidf":
        from .baselines import predict_tfidf
        result = {"predictions": predict_tfidf(args.model, args.articles, args.output)}
    elif args.command == "build-index":
        from .index import build_index
        result = build_index(args.article_index, args.output, args.predictions, args.label_column,
                             args.normalization_start, args.normalization_end, args.ddof, args.missing,
                             name=args.name, measure=args.measure)
    elif args.command == "compare":
        from .index import compare_series
        result = compare_series(args.reconstructed, args.published, args.output, args.frequency, name=args.name)
    else:
        from .report import build_report
        result = build_report(cfg, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
