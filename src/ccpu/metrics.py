"""Binary metrics with an explicit confusion matrix and zero-division rule."""


def binary_metrics(labels, predictions):
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Nonempty equally sized label and prediction arrays required")
    if any(v not in (0, 1) for v in [*labels, *predictions]):
        raise ValueError("Labels and predictions must be binary")
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(labels), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp + tn) / len(labels), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
