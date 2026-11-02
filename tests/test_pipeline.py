"""Numerical and failure-path tests. All toy text stays in temporary files."""
import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ccpu.common import write_csv
from ccpu.text import NEWSPAPERS, parse_article, clean_model_text
from ccpu.index import standardize, build_index, compare_series
from ccpu.labels import read_labels, split_labels, validate_disjoint
from ccpu.metrics import binary_metrics
from ccpu.audit import prepare_news, write_coverage


class CommandLineTests(unittest.TestCase):
    def test_top_level_help_lists_pipeline_commands(self):
        import contextlib
        import io
        from ccpu.cli import main
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as exc:
            main(["--help"])
        self.assertEqual(exc.exception.code, 0)
        self.assertIn("build-index", output.getvalue())
        self.assertIn("predict", output.getvalue())


class TextTests(unittest.TestCase):
    def test_metadata_removed_title_preserved(self):
        row = {"date": "20000101", "content": ": [example]\n本内容经慧科的电子服务提供。版权所有。\n\n1. 光明日报 -  (气候政策尚不明确) 2000-01-01 2\n-----\n政策 尚不明确。 https://example.org/a\n"}
        a = parse_article(0, row)
        self.assertEqual(a["newspaper"], "光明日报")
        self.assertEqual(a["text"], "气候政策尚不明确政策尚不明确")
        self.assertEqual(a["date"], "2000-01-01")
        self.assertFalse(a["header_date_mismatch"])

    def test_cleaner_keeps_chinese_and_digits(self):
        self.assertEqual(clean_model_text("碳排放， ２０３０!\n"), "碳排放2030")

    def test_date_mismatch_is_visible(self):
        a = parse_article(2, {"date": "20000102", "content": "1. 人民日报 - (题目) 2000-01-01\n正文"})
        self.assertTrue(a["header_date_mismatch"])

    def test_export_search_interval_uses_article_date(self):
        a = parse_article(1, {"date": "2000010120000131", "content": "1. 科技日报 - (标题) 2000-01-29\n正文"})
        self.assertEqual(a["date"], "2000-01-29")
        self.assertEqual(a["date_resolution"], "header_within_export_range")

    def test_header_outside_search_interval_fails(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            parse_article(1, {"date": "2000010120000131", "content": "1. 科技日报 - (标题) 2000-02-01\n正文"})

    def test_digital_alias_and_financial_table_are_not_extra_articles(self):
        a = parse_article(1, {"date": "20000101", "content": "1. 人民日报(数字报) - (标题) 2000-01-01\n正文\n1.5汇兑损益 （损失以 ‘-’ 号填列） - 0.3"})
        self.assertEqual(a["newspaper"], "人民日报")

    def test_multi_header_fails(self):
        with self.assertRaisesRegex(ValueError, "Multiple article"):
            parse_article(0, {"date": "20000101", "content": "1. 人民日报 - (一) 2000-01-01\n正文\n2. 光明日报 - (二) 2000-01-01\n正文"})

    def test_invalid_date_fails(self):
        with self.assertRaisesRegex(ValueError, "Invalid date"):
            parse_article(0, {"date": "20000230", "content": "x"})

    def test_prepare_uses_article_csv_not_column_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            write_csv(p / "input.csv", ["", "content", "date"], [{"": 0, "date": "20000101", "content": "1. 人民日报 - (标题) 2000-01-01\n正文"}])
            self.assertEqual(prepare_news(p / "input.csv", p / "out.csv"), 1)
            with (p / "out.csv").open(encoding="utf-8-sig", newline="") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["text"], "标题正文")


class IndexTests(unittest.TestCase):
    def test_corpus_coverage_exposes_zero_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            counts = {(m, p): 10 for p in NEWSPAPERS for m in ["2000-01", "2000-02"]}
            counts.pop(("2000-01", "人民日报"))
            result = write_coverage(counts, tmp)
            self.assertEqual(result["expected_cells"], 12)
            self.assertEqual(result["missing_cell_count"], 1)
            self.assertEqual(result["months_with_missing_newspapers"], 1)

    def counts(self):
        # All six papers: ratios 0.0 and 1.0, population SD = 0.5.
        return {(m, p): (10, k) for p in NEWSPAPERS for m, k in [("2000-01", 0), ("2000-02", 10)]}

    def test_formula_no_centering_or_mean100(self):
        detail, result = standardize(self.counts(), "2000-01", "2000-02")
        self.assertEqual([r["ccpu"] for r in result], [0.0, 2.0])
        self.assertEqual(detail[0]["sigma"], 0.5)

    def test_ddof_is_explicit(self):
        _, result = standardize(self.counts(), "2000-01", "2000-02", 1)
        self.assertAlmostEqual(result[1]["ccpu"], math.sqrt(2))

    def test_missing_newspaper_fails(self):
        counts = self.counts()
        counts.pop(("2000-02", NEWSPAPERS[0]))
        with self.assertRaisesRegex(ValueError, "Missing newspaper"):
            standardize(counts, "2000-01", "2000-02")

    def test_missing_whole_month_fails(self):
        counts = {(m, p): v for (m, p), v in self.counts().items() if m != "2000-02"}
        counts.update({("2000-03", p): (10, 10) for p in NEWSPAPERS})
        with self.assertRaisesRegex(ValueError, "2000-02"):
            standardize(counts, "2000-01", "2000-03")

    def test_zero_variance_fails(self):
        counts = {k: (10, 1) for k in self.counts()}
        with self.assertRaisesRegex(ValueError, "Zero standard"):
            standardize(counts, "2000-01", "2000-02")

    def test_available_policy_averages_observed_papers(self):
        counts = self.counts()
        counts.pop(("2000-02", NEWSPAPERS[0]))
        counts[("2000-03", NEWSPAPERS[0])] = (10, 10)
        for p in NEWSPAPERS[1:]:
            counts[("2000-03", p)] = (10, 0)
        _, result = standardize(counts, "2000-01", "2000-03", missing="available")
        self.assertEqual([r["newspapers"] for r in result], [6, 5, 6])

    def index_files(self, p):
        index = [{"article_id": f"{m}-{i}-{k}", "date": f"2000-{m:02d}-01", "newspaper": paper, "chars": 10,
                  "climate_hits": int(k == 0), "rule_label": 0}
                 for m in [1, 2] for i, paper in enumerate(NEWSPAPERS) for k in range(2)]
        write_csv(p / "article_index.csv", list(index[0]), index)
        predictions = [{"article_id": r["article_id"], "date": r["date"], "newspaper": r["newspaper"],
                        "cpu_label": int(r["date"][5:7] == "02")} for r in index if r["climate_hits"]]
        write_csv(p / "pred.csv", list(predictions[0]), predictions)

    def test_end_to_end_index_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            self.index_files(p)
            summary = build_index(p / "article_index.csv", p, p / "pred.csv", start="2000-01-01",
                                  end="2000-02-29", frequencies=("monthly",), name="x")
            self.assertEqual(summary["outputs"]["monthly"], 2)
            self.assertEqual(summary["cpu_articles"], 6)
            write_csv(p / "published.csv", ["date", "ccpu"], [{"date": "2000-01-01", "ccpu": 0}, {"date": "2000-02-28", "ccpu": 2}])
            metrics = compare_series(p / "x_monthly.csv", p / "published.csv", p)
            self.assertEqual(metrics["pearson_r"], 1.0)
            self.assertEqual(metrics["rmse"], 0.0)

    def test_candidate_without_prediction_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            self.index_files(p)
            write_csv(p / "pred.csv", ["article_id", "cpu_label"], [{"article_id": "1-0-0", "cpu_label": 1}])
            with self.assertRaisesRegex(ValueError, "no prediction"):
                build_index(p / "article_index.csv", p, p / "pred.csv", frequencies=("monthly",))

    def test_duplicate_article_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            row = {"article_id": "same", "date": "2000-01-01", "newspaper": NEWSPAPERS[0], "climate_hits": 0, "rule_label": 0}
            write_csv(p / "a.csv", list(row), [row, row])
            with self.assertRaisesRegex(ValueError, "duplicate article"):
                build_index(p / "a.csv", p, label_column="rule_label", frequencies=("monthly",))


class LexiconAndAnnotationTests(unittest.TestCase):
    def test_rule_needs_all_three_term_groups(self):
        from ccpu.lexicon import rule_label
        self.assertEqual(rule_label("碳排放政策前景不确定"), 1)
        self.assertEqual(rule_label("碳排放政策"), 0)
        self.assertEqual(rule_label("经济政策不确定"), 0)

    def test_focus_window_brings_late_climate_passage_forward(self):
        from ccpu.lexicon import focus_window
        text = "无关内容" * 200 + "国家将出台碳达峰行动方案" + "其他" * 50
        window = focus_window("标题", text)
        self.assertTrue(window.startswith("标题。"))
        self.assertIn("碳达峰", window[:260])
        self.assertLessEqual(len(window), 600)

    def test_kappa(self):
        from ccpu.annotate import cohen_kappa
        self.assertEqual(cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)
        self.assertAlmostEqual(cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]), 0.0)

    def test_proportional_sample_exact_size(self):
        import random
        from ccpu.annotate import proportional_sample
        rows = [{"date": f"{2000 + i % 3}-01-01", "newspaper": NEWSPAPERS[i % 2], "id": i} for i in range(100)]
        sample = proportional_sample(rows, 17, random.Random(1))
        self.assertEqual(len(sample), 17)
        self.assertEqual(len({r["id"] for r in sample}), 17)

    def test_build_corpus_on_toy_export(self):
        from ccpu.corpus import build_corpus
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            rows = [{"": 0, "date": "20100101", "content": "1. 人民日报 - (应对气候变化) 2010-01-01\n减排目标尚不确定"},
                    {"": 1, "date": "20100101", "content": "1. 光明日报 - (文化) 2010-01-01\n书法展览开幕"}]
            write_csv(p / "news.csv", ["", "date", "content"], rows)
            summary = build_corpus(p / "news.csv", p / "out", noncandidate_sample=5)
            self.assertEqual((summary["articles"], summary["candidates"], summary["noncandidate_sample"]), (2, 1, 1))
            with (p / "out" / "candidates.csv").open(encoding="utf-8-sig", newline="") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["rule_label"], "1")


class LabelTests(unittest.TestCase):
    def test_conflicting_labels_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "labels.csv"
            write_csv(p, ["text", "label"], [{"text": "一样的文字", "label": 1}, {"text": "一样的文字", "label": 0}])
            with self.assertRaisesRegex(ValueError, "Conflicting labels"):
                read_labels(p)

    def test_split_is_deterministic_disjoint_and_stratified(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            rows = [{"text": f"类别{label}示例{i}", "label": label} for label in [0, 1] for i in range(20)]
            write_csv(p / "labels.csv", ["text", "label"], rows)
            a = split_labels(p / "labels.csv", p / "a", seed=666)
            b = split_labels(p / "labels.csv", p / "b", seed=666)
            self.assertEqual(a, b)
            for name in ["train", "validation", "test"]:
                self.assertEqual((p / "a" / f"{name}.csv").read_bytes(), (p / "b" / f"{name}.csv").read_bytes())
            datasets = [read_labels(p / "a" / f"{name}.csv") for name in ["train", "validation", "test"]]
            validate_disjoint(*datasets)
            self.assertEqual([len(d) for d in datasets], [28, 6, 6])

    def test_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "across"):
            validate_disjoint([{"text_sha256": "a"}], [{"text_sha256": "a"}])

    def test_metrics(self):
        result = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual([result[k] for k in ["tp", "tn", "fp", "fn"]], [1, 1, 1, 1])
        self.assertEqual(result["f1"], 0.5)
        self.assertEqual(binary_metrics([0, 0], [0, 0])["precision"], 0.0)


if __name__ == "__main__":
    unittest.main()
