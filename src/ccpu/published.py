"""Tidy the supplied published Excel series without changing their values."""
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import openpyxl
from .common import fingerprint, utc_now, write_csv, write_json

FIELDS = ["date", "frequency", "level", "region_code", "region_en", "region_zh", "ccpu", "series_origin", "source_file", "source_sheet", "source_excel_row", "paper_period"]


def normalize_date(value, frequency):
    if frequency == "annual":
        return f"{int(value):04d}-01-01"
    if not isinstance(value, datetime):
        raise ValueError(f"Expected Excel date, got {value!r}")
    return value.strftime("%Y-%m-01" if frequency == "monthly" else "%Y-%m-%d")


def build_published(cfg, out, report):
    source = Path(cfg["source_dir"])
    national_path = source / cfg["national_file"]
    provincial_path = source / cfg["provincial_file"]
    national = openpyxl.load_workbook(national_path, read_only=True, data_only=True)
    provincial = openpyxl.load_workbook(provincial_path, read_only=True, data_only=True)
    codes = {}
    for en, zh, code in list(provincial["Region_Codes"].values)[1:]:
        if en in codes:
            raise ValueError(f"Duplicate province {en}")
        codes[en] = (str(code), zh)
    if len(codes) != 31:
        raise ValueError("Expected 31 provinces in Region_Codes")
    rows, keys, stats = [], set(), []
    for workbook, path, level in [(national, national_path, "national"), (provincial, provincial_path, "province")]:
        for sheet in workbook:
            frequency = {"Daily": "daily", "Monthly": "monthly", "Annual": "annual"}.get(sheet.title.strip())
            if frequency is None:
                continue
            values = iter(sheet.values)
            header = next(values)
            sheet_rows = []
            for row_num, row in enumerate(values, 2):
                if row[0] is None:
                    if any(v is not None for v in row):
                        raise ValueError(f"Missing date in {path.name}/{sheet.title}/{row_num}")
                    continue
                date = normalize_date(row[0], frequency)
                for col, value in enumerate(row[1:], 1):
                    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"Invalid CCPU: {path.name}/{sheet.title}/{row_num}/{col+1}")
                    en = "China" if level == "national" else header[col]
                    code, zh = ("CN", "中国") if level == "national" else codes[en]
                    key = (frequency, level, code, date)
                    if key in keys:
                        raise ValueError(f"Duplicate key: {key}")
                    keys.add(key)
                    record = dict(zip(FIELDS, [date, frequency, level, code, en, zh, value,
                        "published_author_series", path.name, sheet.title, row_num,
                        int(cfg["paper_start"] <= date <= cfg["paper_end"])]))
                    rows.append(record)
                    sheet_rows.append(record)
            dates = sorted({r["date"] for r in sheet_rows})
            stats.append({"file": path.name, "sheet": sheet.title, "frequency": frequency,
                "level": level, "observations": len(sheet_rows), "periods": len(dates),
                "date_min": min(dates), "date_max": max(dates),
                "minimum": min(r["ccpu"] for r in sheet_rows),
                "maximum": max(r["ccpu"] for r in sheet_rows)})
    national.close()
    provincial.close()
    rows.sort(key=lambda r: (r["frequency"], r["level"], r["region_code"], r["date"]))
    out = Path(out)
    write_csv(out / "ccpu_final.csv", FIELDS, rows)
    write_csv(out / "ccpu_paper_period.csv", FIELDS, (r for r in rows if r["paper_period"]))
    for name, predicate in [
        ("national_daily.csv", lambda r: r["level"] == "national" and r["frequency"] == "daily"),
        ("national_monthly.csv", lambda r: r["level"] == "national" and r["frequency"] == "monthly"),
        ("national_annual.csv", lambda r: r["level"] == "national" and r["frequency"] == "annual"),
        ("province_monthly.csv", lambda r: r["level"] == "province" and r["frequency"] == "monthly"),
        ("province_annual.csv", lambda r: r["level"] == "province" and r["frequency"] == "annual")]:
        write_csv(out / name, FIELDS, (r for r in rows if predicate(r)))
    write_csv(out / "region_codes.csv", ["region_code", "region_en", "region_zh"],
        ({"region_code": code, "region_en": en, "region_zh": zh} for en, (code, zh) in codes.items()))
    checks = check_continuity(rows)
    audit = {"generated_at_utc": utc_now(), "rows": len(rows),
             "paper_period_rows": sum(r["paper_period"] for r in rows), "duplicate_keys": 0,
             "invalid_or_missing_ccpu": 0, "sheets": stats, "continuity": checks,
             "source_files": [{"name": p.name, "sha256": fingerprint(p), "bytes": p.stat().st_size} for p in (national_path, provincial_path)],
             "origin": "Published author series, tidied; not reconstructed from text.",
             "date_note": "Monthly labels mapped to first day of month. Annual labels mapped to January 1; not a daily observation.",
             "vintage_note": "The 2000-2022 subset uses the supplied updated vintage, not the original publication vintage."}
    write_json(report, audit)
    return audit


def check_continuity(rows):
    from datetime import date, timedelta
    groups = defaultdict(list)
    for r in rows:
        groups[(r["frequency"], r["level"], r["region_code"])].append(r["date"])
    missing = []
    for (frequency, level, region), dates in sorted(groups.items()):
        observed = set(dates)
        current, last = date.fromisoformat(min(dates)), date.fromisoformat(max(dates))
        while current <= last:
            if current.isoformat() not in observed:
                missing.append({"frequency": frequency, "level": level, "region_code": region, "date": current.isoformat()})
            if frequency == "daily":
                current += timedelta(days=1)
            elif frequency == "monthly":
                current = date(current.year + (current.month == 12), current.month % 12 + 1, 1)
            else:
                current = date(current.year + 1, 1, 1)
    return {"series_checked": len(groups), "missing_periods": missing}
