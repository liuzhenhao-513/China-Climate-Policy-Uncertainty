"""Annotation sample design, batch export, agreement statistics and label merge.

Design (docs/ANNOTATION_GUIDE.md):
  * main: candidates drawn at random within year x newspaper strata,
    proportional to the number of candidates in each stratum;
  * noncandidate: random articles without any climate term (filter-recall check);
  * double: a random subset of main items given to a second annotator (Cohen's kappa).
"""
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from .common import utc_now, write_csv, write_json

POOL_FIELDS = ["article_id", "date", "newspaper", "title", "climate_hits", "rule_label", "role", "annotator", "batch", "focus_text"]
LABEL_FIELDS = ["article_id", "label", "dimension", "confidence", "reason"]


def read_rows(path):
    csv.field_size_limit(100_000_000)
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def proportional_sample(rows, n, rng, key=lambda r: (r["date"][:4], r["newspaper"])):
    strata = defaultdict(list)
    for r in rows:
        strata[key(r)].append(r)
    total = len(rows)
    quotas = {k: n * len(v) / total for k, v in strata.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    # largest remainder so the sample size is exactly n
    for k in sorted(quotas, key=lambda k: quotas[k] - alloc[k], reverse=True)[: n - sum(alloc.values())]:
        alloc[k] += 1
    sample = []
    for k in sorted(strata):
        sample.extend(rng.sample(strata[k], min(alloc[k], len(strata[k]))))
    return sample


def design_sample(candidates, noncandidates, out, n_main=2000, n_non=300, n_double=300,
                  annotators=8, batches_per_annotator=3, seed=666):
    rng = random.Random(seed)
    cand = read_rows(candidates)
    non = read_rows(noncandidates)
    main = proportional_sample(cand, n_main, rng)
    non = rng.sample(non, min(n_non, len(non)))
    items = [dict(r, role="main") for r in main] + [dict(r, role="noncandidate") for r in non]
    rng.shuffle(items)
    for i, r in enumerate(items):
        r["annotator"] = i % annotators
    double = rng.sample([r for r in items if r["role"] == "main"], n_double)
    items += [dict(r, role="double", annotator=(r["annotator"] + 1) % annotators) for r in double]
    by_annotator = defaultdict(list)
    for r in items:
        by_annotator[r["annotator"]].append(r)
    out = Path(out)
    (out / "batches").mkdir(parents=True, exist_ok=True)
    pool = []
    for a, rows in sorted(by_annotator.items()):
        rng.shuffle(rows)  # second-annotator items are mixed in, not flagged
        size = -(-len(rows) // batches_per_annotator)
        for b in range(batches_per_annotator):
            chunk = rows[b * size:(b + 1) * size]
            name = f"a{a}_b{b}"
            for r in chunk:
                r["batch"] = name
                pool.append(r)
            lines = [f"# Batch {name}: {len(chunk)} articles\n"]
            for r in chunk:
                lines.append(f"## {r['article_id']}\n{r['date']} | {r['newspaper']}\n{r['focus_text']}\n")
            (out / "batches" / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
    write_csv(out / "annotation_pool.csv", POOL_FIELDS, ({k: r[k] for k in POOL_FIELDS} for r in pool))
    summary = {"generated_at_utc": utc_now(), "seed": seed, "candidates_in_population": len(cand),
               "main": n_main, "noncandidate": len(non), "double": n_double, "annotators": annotators,
               "batches": annotators * batches_per_annotator,
               "main_by_year": dict(sorted(Counter(r["date"][:4] for r in main).items()))}
    write_json(out / "annotation_design.json", summary)
    return summary


def cohen_kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = Counter(a), Counter(b)
    pe = sum(pa[k] * pb[k] for k in set(a) | set(b)) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def read_annotations(folder):
    labels = defaultdict(dict)
    for path in sorted(Path(folder).glob("a*_b*.csv")):
        for r in read_rows(path):
            if r["label"] not in ("0", "1"):
                raise ValueError(f"Non-binary label in {path}: {r}")
            labels[path.stem][r["article_id"]] = r
    return labels


def write_review_sample(final, out, n=100, seed=666):
    """Blind review sheet: labels are not shown; review-agreement joins them back."""
    if (Path(out) / "human_review_sample.csv").exists():
        return  # never overwrite a sheet that may already contain review labels
    rng = random.Random(seed)
    main = [r for r in final if r["role"] == "main"]
    pos = [r for r in main if r["label"] == "1"]
    neg = [r for r in main if r["label"] == "0"]
    picked = rng.sample(pos, min(n // 2, len(pos))) + rng.sample(neg, min(n - n // 2, len(neg)))
    rng.shuffle(picked)
    write_csv(Path(out) / "human_review_sample.csv", ["article_id", "date", "newspaper", "title", "text", "human_label", "note"],
              ({"article_id": r["article_id"], "date": r["date"], "newspaper": r["newspaper"], "title": r["title"],
                "text": r["text"], "human_label": "", "note": ""} for r in picked))


def review_agreement(path, labels_file=None):
    """Compare a reviewer's `human_label` column with the training labels."""
    labels_file = labels_file or Path(path).with_name("cpu_labels.csv")
    labels = {r["article_id"]: r["label"] for r in read_rows(labels_file)}
    rows = [r for r in read_rows(path) if r.get("human_label", "").strip() in ("0", "1")]
    if not rows:
        raise ValueError("Fill the human_label column (0/1) before running review-agreement")
    a = [int(labels[r["article_id"]]) for r in rows]
    b = [int(r["human_label"]) for r in rows]
    from .metrics import binary_metrics
    return {"reviewed": len(rows), "agreement": sum(x == y for x, y in zip(a, b)) / len(rows),
            "cohen_kappa": cohen_kappa(a, b), "labels_vs_reviewer": binary_metrics(b, a)}


def merge_labels(pool_file, annotation_dir, out, adjudication_file=None):
    pool = read_rows(pool_file)
    ann = read_annotations(annotation_dir)
    by_batch = defaultdict(list)
    for r in pool:
        by_batch[r["batch"]].append(r)
    missing = [(b, r["article_id"]) for b, rows in by_batch.items() for r in rows if r["article_id"] not in ann.get(b, {})]
    if missing:
        raise ValueError(f"{len(missing)} pool items lack annotations, e.g. {missing[:5]}")
    first, second = {}, {}
    for r in pool:
        rec = {**r, **ann[r["batch"]][r["article_id"]]}
        (second if r["role"] == "double" else first)[r["article_id"]] = rec
    ids = sorted(second)
    a = [int(first[i]["label"]) for i in ids]
    b = [int(second[i]["label"]) for i in ids]
    disagreements = [i for i in ids if first[i]["label"] != second[i]["label"]]
    adjudicated = {}
    if adjudication_file and Path(adjudication_file).exists():
        adjudicated = {r["article_id"]: r for r in read_rows(adjudication_file)}
    unresolved = [i for i in disagreements if i not in adjudicated]
    if unresolved:
        write_csv(Path(out) / "disagreements_to_adjudicate.csv",
                  ["article_id", "label_1", "reason_1", "label_2", "reason_2", "focus_text"],
                  ({"article_id": i, "label_1": first[i]["label"], "reason_1": first[i]["reason"],
                    "label_2": second[i]["label"], "reason_2": second[i]["reason"],
                    "focus_text": first[i]["focus_text"]} for i in unresolved))
        raise ValueError(f"{len(unresolved)} disagreements need adjudication; see disagreements_to_adjudicate.csv")
    final = []
    for i, r in first.items():
        label, how = r["label"], "single"
        if i in second:
            how = "double_agree" if i not in adjudicated else "adjudicated"
            label = adjudicated[i]["label"] if i in adjudicated else label
        final.append({"article_id": i, "date": r["date"], "newspaper": r["newspaper"], "role": r["role"],
                      "label": label, "dimension": r["dimension"], "confidence": r["confidence"],
                      "label_source": how, "climate_hits": r["climate_hits"], "rule_label": r["rule_label"],
                      "title": r["title"], "text": r["focus_text"]})
    final.sort(key=lambda r: r["article_id"])
    write_csv(Path(out) / "cpu_labels.csv", list(final[0]), final)
    write_review_sample(final, out)
    roles = Counter(r["role"] for r in final)
    positives = Counter(r["role"] for r in final if r["label"] == "1")
    report = {"generated_at_utc": utc_now(), "labelled_articles": len(final),
              "by_role": dict(roles), "positives_by_role": dict(positives),
              "positive_rate_main": positives["main"] / max(roles["main"], 1),
              "positive_rate_noncandidate": positives["noncandidate"] / max(roles["noncandidate"], 1),
              "double_annotated": len(ids), "raw_agreement": 1 - len(disagreements) / len(ids),
              "cohen_kappa": cohen_kappa(a, b), "disagreements": len(disagreements),
              "adjudicated": len(adjudicated)}
    write_json(Path(out) / "label_report.json", report)
    return report
