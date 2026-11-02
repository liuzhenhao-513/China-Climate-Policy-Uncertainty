"""Full-corpus streaming audit with compact non-text evidence."""
from collections import Counter
from pathlib import Path
from .common import rel_path, utc_now, write_csv, write_json
from .text import NEWSPAPERS, iter_news, parse_article


def audit_news(path, out, limit=None):
    out = Path(out)
    counts = Counter()
    issues = Counter({k: 0 for k in ["empty_content", "header_date_mismatch", "repeated_export_index", "parse_error", "unknown_newspaper", "header_missing", "date_recovered_from_header_within_export_range"]})
    examples = []
    rows, min_date, max_date = 0, None, None
    source_ids = set()
    for i, row in iter_news(path, limit):
        rows += 1
        key = (row.get("") or "").strip()
        if key in source_ids:
            issues["repeated_export_index"] += 1
        source_ids.add(key)
        if not (row.get("content") or "").strip():
            issues["empty_content"] += 1
        try:
            a = parse_article(i, row, include_text=False)
        except ValueError as exc:
            issues["parse_error"] += 1
            if len(examples) < 20:
                examples.append({"row": i, "reason": str(exc)})
            continue
        counts[(a["month"], a["newspaper"])] += 1
        min_date = min(min_date or a["date"], a["date"])
        max_date = max(max_date or a["date"], a["date"])
        for flag in ["header_date_mismatch"]:
            issues[flag] += int(a[flag])
        if a["date_resolution"] == "header_within_export_range":
            issues["date_recovered_from_header_within_export_range"] += 1
        if a["newspaper"] not in NEWSPAPERS:
            issues["unknown_newspaper"] += 1
            if len(examples) < 20:
                examples.append({"row": i, "newspaper": a["newspaper"]})
        if not a["header_found"]:
            issues["header_missing"] += 1
        if rows % 100_000 == 0:
            print(f"Audited {rows:,} records", flush=True)
    newspaper_totals = Counter()
    for (month, paper), count in counts.items():
        newspaper_totals[paper] += count
    profile = {
        "generated_at_utc": utc_now(), "source_file": Path(path).name,
        "source_bytes": Path(path).stat().st_size, "records_scanned": rows,
        "full_scan": limit is None, "limit": limit,
        "date_min": min_date, "date_max": max_date,
        "counts_by_newspaper": dict(sorted(newspaper_totals.items())),
        "issues": dict(issues), "examples_without_article_text": examples,
        "article_text_duplicates_checked": False,
        "note": "The unnamed export index resets across source groups and is not an article key. Digital-paper aliases are merged. A 16-digit search interval is resolved only using an article-header date within that interval. Identical/reprinted article texts are not deduplicated. Unresolved multiple headers and invalid dates are parse errors."
    }
    write_json(out / "news_audit.json", profile)
    write_csv(out / "news_monthly_counts.csv", ["month", "newspaper", "articles"],
              ({"month": m, "newspaper": p, "articles": n} for (m, p), n in sorted(counts.items())))
    coverage = write_coverage(counts, out)
    profile["paper_month_coverage"] = coverage
    write_json(out / "news_audit.json", profile)
    return profile


def write_coverage(counts, out):
    """Make absent paper-months visible instead of treating absence as zero CPU."""
    from .index import month_range
    months = sorted({m for m, p in counts})
    if not months:
        return {"expected_cells": 0, "observed_cells": 0, "missing_cells": []}
    rows, missing = [], []
    for month in month_range(months[0], months[-1]):
        for paper in NEWSPAPERS:
            total = counts.get((month, paper), 0)
            row = {"month": month, "newspaper": paper, "articles": total, "has_articles": int(total > 0)}
            rows.append(row)
            if not total:
                missing.append({"month": month, "newspaper": paper})
    write_csv(Path(out) / "news_paper_month_coverage.csv", ["month", "newspaper", "articles", "has_articles"], rows)
    write_csv(Path(out) / "news_missing_paper_months.csv", ["month", "newspaper"], missing)
    return {"expected_cells": len(rows), "observed_cells": len(rows)-len(missing),
            "missing_cell_count": len(missing), "months_with_missing_newspapers": len({r["month"] for r in missing}),
            "missing_cells": missing,
            "note": "A zero article denominator is missing corpus coverage, not zero CPU. Strict full-period aggregation cannot run until a missing-cell policy or additional data are supplied."}


def prepare_news(path, output, limit=None, start=None, end=None):
    """Stream a selected period to article CSV. Fail on ambiguous parse/empty text."""
    import csv
    from datetime import date
    from .text import parse_article
    for value in (start, end):
        if value is not None:
            date.fromisoformat(value)
    if start and end and start > end:
        raise ValueError("Start date must not be after end date")
    fields = ["article_id", "source_row", "source_date", "date_resolution", "date", "month", "newspaper", "title", "text", "text_sha256", "clean_characters"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    count = 0
    with temporary.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for i, row in iter_news(path, limit):
            a = parse_article(i, row)
            if (start and a["date"] < start) or (end and a["date"] > end):
                continue
            if a["newspaper"] not in NEWSPAPERS or a["header_date_mismatch"] or not a["text"]:
                raise ValueError(f"Invalid article metadata or empty cleaned text at row {i}")
            writer.writerow(a)
            count += 1
            if count % 100_000 == 0:
                print(f"Prepared {count:,} articles", flush=True)
    temporary.replace(output)
    write_json(output.with_suffix(".meta.json"), {
        "generated_at_utc": utc_now(), "source": rel_path(Path(path).resolve()),
        "rows": count, "limit": limit, "start": start, "end": end,
        "cleaner": "title+body, remove export metadata, URLs, punctuation and whitespace; NFKC",
        "deduplication": "none; each original CSV row is retained as an observation"
    })
    return count
