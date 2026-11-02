"""Round-trip every delivered value against the supplied Excel input."""
import csv
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ccpu.common import ROOT, read_config
from ccpu.published import normalize_date


class PublishedDataTests(unittest.TestCase):
    def test_all_exported_values_match_source_cells(self):
        import openpyxl
        cfg = read_config()
        source = Path(cfg["source_dir"])
        if not (source / cfg["national_file"]).exists():
            source = ROOT / "data/raw"
        if not (source / cfg["national_file"]).exists():
            self.skipTest("Raw Excel inputs are not present in this checkout")
        books = {name: openpyxl.load_workbook(source / name, read_only=True, data_only=True)
                 for name in [cfg["national_file"], cfg["provincial_file"]]}
        tables = {(name, sheet.title): list(sheet.values) for name, book in books.items() for sheet in book}
        n, keys = 0, set()
        with (ROOT / "data/published/ccpu_final.csv").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                table = tables[(row["source_file"], row["source_sheet"])]
                column = 1 if row["level"] == "national" else table[0].index(row["region_en"])
                source_row = table[int(row["source_excel_row"]) - 1]
                self.assertEqual(float(row["ccpu"]), source_row[column])
                self.assertEqual(row["date"], normalize_date(source_row[0], row["frequency"]))
                key = tuple(row[k] for k in ["frequency", "level", "region_code", "date"])
                self.assertNotIn(key, keys)
                keys.add(key)
                n += 1
        self.assertEqual(n, 20749)
        for book in books.values():
            book.close()

    def test_paper_subset_has_correct_scope(self):
        with (ROOT / "data/published/ccpu_paper_period.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 17969)
        self.assertTrue(all("2000-01-01" <= r["date"] <= "2022-12-31" for r in rows))
        self.assertTrue(all(r["series_origin"] == "published_author_series" for r in rows))


if __name__ == "__main__":
    unittest.main()
