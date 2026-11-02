"""CCPU index from article-level labels (Ma et al., 2023, p. 5; Baker et al., 2016).

For newspaper i and period t:
    X_it = N_cpu_it / N_all_it
    Y_it = X_it / sd_T(X_i)          (sd over the normalization window T)
    Z_t  = mean_i Y_it                (equal newspaper weights)

No mean subtraction and no rescaling to mean 100, as in the paper.
"""
import csv
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from .common import fingerprint, utc_now, write_csv, write_json
from .text import NEWSPAPERS

FREQUENCIES = ("monthly", "annual", "daily")


def period_of(day, frequency):
    return {"monthly": day[:7], "annual": day[:4], "daily": day[:10]}[frequency]


def period_range(start, end, frequency="monthly"):
    if frequency == "annual":
        yield from (str(y) for y in range(int(start[:4]), int(end[:4]) + 1))
    elif frequency == "monthly":
        current, stop = date.fromisoformat(start[:7] + "-01"), date.fromisoformat(end[:7] + "-01")
        while current <= stop:
            yield current.strftime("%Y-%m")
            current = date(current.year + (current.month == 12), current.month % 12 + 1, 1)
    else:
        current, stop = date.fromisoformat(start), date.fromisoformat(end)
        while current <= stop:
            yield current.isoformat()
            current += timedelta(days=1)


def month_range(start, end):
    return period_range(start, end, "monthly")


def period_start(period):
    return {4: period + "-01-01", 7: period + "-01", 10: period}[len(period)]


def standardize(counts, start, end, ddof=0, missing="strict", frequency="monthly", measure="share"):
    """Return (component rows, index rows).

    counts maps (period, newspaper) -> (articles, cpu_articles).
    missing="strict": every newspaper must have articles in every period.
    missing="available": a period averages the newspapers that published in it,
    and each sigma_i uses the periods in which newspaper i was observed.
    Days without any article from any newspaper are left out of a daily series.
    measure="share" is the paper's X_it = N_cpu/N_all. measure="count" uses a
    fixed per-newspaper denominator instead (X_it = N_cpu_it / const_i); after
    division by sd_T the constant cancels, so Y_it = N_cpu_it / sd_T(N_cpu_i).
    It is the right choice when the corpus is not a constant fraction of each
    newspaper's output (docs/METHOD.md, "Corpus composition").
    """
    if measure not in ("share", "count"):
        raise ValueError("measure must be 'share' or 'count'")
    if ddof not in (0, 1):
        raise ValueError("ddof must be 0 or 1")
    if missing not in ("strict", "available"):
        raise ValueError("missing must be 'strict' or 'available'")
    observed = sorted({m for m, _ in counts})
    if not observed:
        raise ValueError("No article counts")
    if start > end:
        raise ValueError("Normalization interval is reversed")
    normalization = list(period_range(start, end, frequency))
    if len(normalization) <= ddof:
        raise ValueError("Normalization interval is too short")
    if frequency == "daily":
        periods = observed
    else:
        periods = sorted(set(period_range(observed[0], observed[-1], frequency)) | set(normalization))
    ratios = {}
    for m in periods:
        for p in NEWSPAPERS:
            total, positive = counts.get((m, p), (0, 0))
            if positive < 0 or positive > total:
                raise ValueError("Positive articles must be between 0 and total")
            if total > 0:
                ratios[(m, p)] = positive / total if measure == "share" else float(positive)
            elif missing == "strict":
                raise ValueError(f"Missing newspaper/period or zero denominator: {p}, {m}")
    sigma = {}
    for p in NEWSPAPERS:
        values = [ratios[(m, p)] for m in normalization if (m, p) in ratios]
        if len(values) <= ddof:
            raise ValueError(f"Too few observed periods for {p} in the normalization window")
        mean = math.fsum(values) / len(values)
        sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (len(values) - ddof))
        if sd <= 0:
            raise ValueError(f"Zero standard deviation for {p}")
        sigma[p] = sd
    detail, series = [], []
    for m in observed:
        scaled = []
        for p in NEWSPAPERS:
            if (m, p) not in ratios:
                continue
            total, positive = counts[(m, p)]
            y = ratios[(m, p)] / sigma[p]
            scaled.append(y)
            detail.append(dict(period=m, newspaper=p, articles=total, cpu_articles=positive,
                               ratio=ratios[(m, p)], sigma=sigma[p], standardized_ratio=y))
        if scaled:
            series.append({"date": period_start(m), "ccpu": math.fsum(scaled) / len(scaled),
                           "newspapers": len(scaled), "frequency": frequency, "level": "national",
                           "series_origin": "reconstructed_from_text"})
    return detail, series


def load_counts(article_index, predictions=None, label_column="cpu_label", frequency="monthly"):
    """Count articles and CPU articles per (period, newspaper).

    article_index lists every article (denominator). predictions gives
    cpu_label for candidate articles; articles absent from predictions must have
    zero climate terms and count as CPU = 0. If predictions is None,
    label_column is read from article_index itself (e.g. rule_label).
    """
    csv.field_size_limit(100_000_000)
    labels = {}
    if predictions is not None:
        with Path(predictions).open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row["article_id"] in labels:
                    raise ValueError(f"Duplicate prediction ID: {row['article_id']}")
                labels[row["article_id"]] = row[label_column]
    counts = defaultdict(lambda: [0, 0])
    ids, n, matched = set(), 0, 0
    with Path(article_index).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            article_id = row["article_id"]
            if not article_id or article_id in ids:
                raise ValueError(f"Missing or duplicate article ID: {article_id}")
            ids.add(article_id)
            p = row["newspaper"]
            if p not in NEWSPAPERS:
                raise ValueError(f"Unknown newspaper at article {article_id}")
            if predictions is None and label_column == "climate_any":
                label = "1" if int(row["climate_hits"]) > 0 else "0"
            elif predictions is None:
                label = row[label_column]
            elif article_id in labels:
                label = labels[article_id]
                matched += 1
            elif int(row.get("climate_hits", 0) or 0) > 0:
                raise ValueError(f"Candidate article {article_id} has no prediction")
            else:
                label = "0"
            if label not in ("0", "1"):
                raise ValueError(f"Invalid label at article {article_id}")
            cell = counts[(period_of(row["date"], frequency), p)]
            cell[0] += 1
            cell[1] += int(label)
            n += 1
    if predictions is not None and matched != len(labels):
        raise ValueError(f"{len(labels) - matched} predictions do not match the article index")
    return counts, n


def build_index(article_index, output, predictions=None, label_column="cpu_label", start="2000-01-01",
                end="2022-12-31", ddof=0, missing="available", frequencies=FREQUENCIES, name="ccpu", measure="share"):
    output = Path(output)
    summary = {"generated_at_utc": utc_now(), "article_index": str(Path(article_index).name),
               "predictions": str(Path(predictions).name) if predictions else None,
               "label_column": label_column, "measure": measure, "normalization_start": start, "normalization_end": end,
               "ddof": ddof, "missing_policy": missing, "newspapers": NEWSPAPERS,
               "formula": "X_it=N_cpu_it/N_all_it; Y_it=X_it/sd_T(X_i); Z_t=mean over observed i of Y_it",
               "outputs": {}}
    if predictions:
        summary["predictions_sha256"] = fingerprint(predictions)
    for frequency in frequencies:
        counts, n = load_counts(article_index, predictions, label_column, frequency)
        detail, series = standardize(counts, start, end, ddof, missing, frequency, measure)
        write_csv(output / f"{name}_{frequency}.csv", list(series[0]), series)
        if frequency != "daily":
            write_csv(output / f"{name}_{frequency}_components.csv", list(detail[0]), detail)
        summary["articles"] = n
        summary["cpu_articles"] = sum(v[1] for v in counts.values())
        summary["outputs"][frequency] = len(series)
    write_json(output / f"{name}_run.json", summary)
    return summary


def load_series(path, frequency="monthly", level="national", region=None):
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("frequency", frequency) != frequency or row.get("level", level) != level:
                continue
            if region is not None and row.get("region_code") != region:
                continue
            key = period_of(row["date"], frequency)
            value = float(row["ccpu"])
            if key in result or not math.isfinite(value):
                raise ValueError(f"Duplicate or invalid period in {path}: {key}")
            result[key] = value
    return result


def pearson(x, y):
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    cross = math.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    denom = math.sqrt(math.fsum((a - mx) ** 2 for a in x) * math.fsum((b - my) ** 2 for b in y))
    return cross / denom if denom else None


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    r = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def compare_series(reconstructed, published, output=None, frequency="monthly", start=None, end=None, name="comparison"):
    a = load_series(reconstructed, frequency)
    b = load_series(published, frequency)
    common = sorted(k for k in a.keys() & b.keys()
                    if (start is None or k >= period_of(start, frequency)) and (end is None or k <= period_of(end, frequency)))
    if len(common) < 2:
        raise ValueError("At least two overlapping observations are required")
    x, y = [a[k] for k in common], [b[k] for k in common]
    differences = [u - v for u, v in zip(x, y)]
    metrics = {"frequency": frequency, "overlap": len(common), "date_min": common[0], "date_max": common[-1],
               "pearson_r": pearson(x, y), "spearman_rho": pearson(ranks(x), ranks(y)),
               "mean_reconstructed": math.fsum(x) / len(x), "mean_published": math.fsum(y) / len(y),
               "mae": math.fsum(abs(v) for v in differences) / len(common),
               "rmse": math.sqrt(math.fsum(v * v for v in differences) / len(common))}
    if output is not None:
        output = Path(output)
        write_json(output / f"{name}.json", metrics)
        write_csv(output / f"{name}.csv", ["period", "reconstructed", "published", "difference"],
                  ({"period": k, "reconstructed": a[k], "published": b[k], "difference": a[k] - b[k]} for k in common))
    return metrics
