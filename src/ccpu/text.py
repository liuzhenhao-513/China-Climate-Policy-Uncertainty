"""Parse one WiseNews export record at a time; no dictionary classification."""
import csv
import hashlib
import re
import unicodedata
from datetime import datetime
from pathlib import Path

NEWSPAPERS = ("人民日报", "光明日报", "经济日报", "环球时报", "科技日报", "中国新闻社")
ALIASES = {"中新社": "中国新闻社", "中国新闻网": "中国新闻社", "中国新闻社": "中国新闻社"}
HEADER = re.compile(r"^[ \t]*(\d+)\.[ \t]*(?P<paper>[^\r\n]+?)[ \t]+-[ \t]+[（(]", re.MULTILINE)
HEADER_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
SOURCE_ID = re.compile(r"^\s*:\s*\[([^\]]+)\]")
BOILERPLATE = re.compile(r"本内容经慧科的电子服务提供。[^\r\n]*")
SEPARATOR = re.compile(r"^[\s]*[-=]{5,}\s*$", re.MULTILINE)
COUNT = re.compile(r"文章总数\s*[:：]\s*\d+\s*篇")
URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def clean_model_text(text):
    """Paper-compatible removal of URLs, punctuation and whitespace.

    NFKC and export-boilerplate handling are explicit engineering decisions.
    Cleaning is deterministic; the exact original cleaner was not released.
    """
    text = unicodedata.normalize("NFKC", URL.sub("", text))
    return "".join(c for c in text if not c.isspace() and unicodedata.category(c)[0] not in "PZC")


def iter_news(path, limit=None):
    csv.field_size_limit(100_000_000)
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"date", "content"}.issubset(reader.fieldnames or []):
            raise ValueError("News CSV requires date and content columns")
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            if None in row:
                raise ValueError(f"CSV column overflow at record {i}")
            yield i, row


def light_clean(text):
    """Readable text for annotators: NFKC, no URLs, no whitespace; punctuation kept."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", URL.sub("", text)))


def parse_article(i, row, include_text=True, keep_parts=False):
    raw = row.get("content") or ""
    date_value = (row.get("date") or "").strip()
    matches = list(HEADER.finditer(raw))
    if len(matches) > 1:
        raise ValueError(f"Multiple article headers in news record {i}; manual parsing required")
    paper, title, body, mismatch, header_date = "UNKNOWN", "", raw, False, None
    if matches:
        match = matches[0]
        paper = match["paper"].strip().removesuffix("(数字报)").removesuffix("（数字报）").strip()
        paper = ALIASES.get(paper, paper)
        newline = raw.find("\n", match.end())
        end = newline if newline >= 0 else len(raw)
        header = raw[match.end():end].strip()
        date_match = HEADER_DATE.search(header)
        if date_match:
            header_date = date_match[1]
            title = header[:date_match.start()].strip().strip("()（） ")
        else:
            title = header.strip("()（） ")
        body = raw[end:]
    resolution = "date_column"
    try:
        if len(date_value) == 8 and date_value.isdigit():
            date = datetime.strptime(date_value, "%Y%m%d").date().isoformat()
            mismatch = header_date is not None and header_date != date
        elif len(date_value) == 16 and date_value.isdigit() and header_date:
            start = datetime.strptime(date_value[:8], "%Y%m%d").date().isoformat()
            end_date = datetime.strptime(date_value[8:], "%Y%m%d").date().isoformat()
            date = datetime.strptime(header_date, "%Y-%m-%d").date().isoformat()
            if not start <= date <= end_date:
                raise ValueError("Article header date is outside the export search interval")
            resolution = "header_within_export_range"
        else:
            raise ValueError("Expected an 8-digit date or a 16-digit export date interval with an article header")
    except ValueError as exc:
        raise ValueError(f"Invalid date at news record {i}: {date_value!r}: {exc}") from exc
    body = SOURCE_ID.sub("", body)
    body = SEPARATOR.sub("", COUNT.sub("", BOILERPLATE.sub("", body))).strip()
    result = {
        "article_id": f"news-{i:08d}", "source_row": i,
        "date": date, "month": date[:7], "newspaper": paper,
        "source_date": date_value, "date_resolution": resolution,
        "header_found": bool(matches), "header_date_mismatch": mismatch,
        "raw_characters": len(raw),
    }
    if keep_parts:
        result.update(title=title, body=body)
    if include_text:
        text = clean_model_text(title + body)
        result.update(title=title, text=text, clean_characters=len(text),
                      text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
    return result
