"""Benchmarks for the MacBERT classifier, evaluated on the same test split.

* rule:  climate AND policy AND uncertainty dictionary (BBD-style, cf. Xu et al. 2023)
* tfidf: character 1-3 gram TF-IDF + logistic regression

The TF-IDF model is also small enough to ship in the repository and to label the
full candidate set in minutes, which gives a robustness version of the index.
"""
import csv
import pickle
from pathlib import Path
from .common import utc_now, write_csv, write_json
from .labels import read_labels
from .lexicon import rule_label
from .metrics import binary_metrics
from .text import clean_model_text


def train_tfidf(train_file, validation_file, test_file, output, seed=666):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    train = read_labels(train_file) + read_labels(validation_file)
    test = read_labels(test_file)
    model = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, sublinear_tf=True, max_features=200_000),
        LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced", random_state=seed))
    model.fit([clean_model_text(r["text"]) for r in train], [r["label"] for r in train])
    probabilities = model.predict_proba([clean_model_text(r["text"]) for r in test])[:, 1]
    predicted = [int(p >= 0.5) for p in probabilities]
    rules = [rule_label(r["text"]) for r in test]
    truth = [r["label"] for r in test]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "tfidf_lr.pkl").open("wb") as f:
        pickle.dump(model, f)
    write_csv(output / "baseline_test_predictions.csv", ["article_id", "label", "tfidf_label", "tfidf_probability", "rule_label"],
              ({"article_id": r.get("article_id", r["text_sha256"]), "label": r["label"], "tfidf_label": p,
                "tfidf_probability": round(float(s), 6), "rule_label": q}
               for r, p, s, q in zip(test, predicted, probabilities, rules)))
    result = {"generated_at_utc": utc_now(), "train_n": len(train), "test_n": len(test),
              "tfidf_lr": binary_metrics(truth, predicted), "rule": binary_metrics(truth, rules),
              "note": "TF-IDF is fit on train+validation (no early stopping needed); same test split as MacBERT."}
    write_json(output / "baseline_metrics.json", result)
    return result


def predict_tfidf(model_file, candidates, output):
    csv.field_size_limit(100_000_000)
    with Path(model_file).open("rb") as f:
        model = pickle.load(f)
    with Path(candidates).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    probabilities = model.predict_proba([clean_model_text(r["focus_text"]) for r in rows])[:, 1]
    write_csv(output, ["article_id", "date", "newspaper", "cpu_probability", "cpu_label"],
              ({"article_id": r["article_id"], "date": r["date"], "newspaper": r["newspaper"],
                "cpu_probability": round(float(p), 6), "cpu_label": int(p >= 0.5)} for r, p in zip(rows, probabilities)))
    return len(rows)
