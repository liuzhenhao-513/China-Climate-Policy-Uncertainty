"""Offline tiny-random-encoder test; it does not estimate an empirical index."""
import importlib.util
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HAS_MODEL = all(importlib.util.find_spec(p) is not None for p in ["torch", "transformers", "numpy"])


@unittest.skipUnless(HAS_MODEL, "Optional torch/transformers dependencies not installed")
class ModelSmokeTests(unittest.TestCase):
    def test_training_saving_loading_and_streaming_prediction(self):
        import csv
        import hashlib
        import json
        import tempfile
        import torch
        from transformers import BertConfig, BertModel, BertTokenizerFast
        from ccpu.common import write_csv
        from ccpu.model import train_model, predict_articles
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            base = p / "tiny_encoder"
            base.mkdir()
            vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *list("气候政策不确定经济发展甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥")]
            (base / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
            tokenizer = BertTokenizerFast(vocab_file=str(base / "vocab.txt"))
            tokenizer.save_pretrained(base)
            encoder = BertModel(BertConfig(vocab_size=len(vocab), hidden_size=24, num_hidden_layers=1,
                num_attention_heads=4, intermediate_size=32, max_position_embeddings=32))
            encoder.save_pretrained(base)
            for name, suffix in [("train", "甲乙丙丁"), ("validation", "戊己"), ("test", "庚辛")]:
                rows = [{"text": text + char, "label": label} for char in suffix
                    for text, label in [("气候政策不确定", 1), ("经济发展", 0)]]
                write_csv(p / f"{name}.csv", ["text", "label"], rows)
            result = train_model(p / "train.csv", p / "validation.csv", p / "trained", str(base),
                p / "test.csv", device="cpu", epochs=1, batch_size=2, max_length=24)
            self.assertEqual(result["train_n"], 8)
            self.assertEqual(result["test"]["n"], 4)
            rows = [{"article_id": str(i), "date": "2000-01-01", "newspaper": "人民日报",
                "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
                for i, text in enumerate(["气候政策不确定壬", "经济发展癸"])]
            write_csv(p / "articles.csv", list(rows[0]), rows)
            count = predict_articles(p / "articles.csv", p / "trained", p / "predictions.csv",
                device="cpu", batch_size=2, max_length=24)
            self.assertEqual(count, 2)
            with (p / "predictions.csv").open(encoding="utf-8-sig", newline="") as f:
                predictions = list(csv.DictReader(f))
            self.assertEqual(len(predictions), 2)
            self.assertTrue(all(r["cpu_label"] in ("0", "1") for r in predictions))
            self.assertTrue(all(0 <= float(r["cpu_probability"]) <= 1 for r in predictions))

    def test_cls_shape_gradient_and_checkpoint(self):
        import tempfile
        import torch
        from transformers import BertConfig, BertModel
        from ccpu.model import BinaryMacBERT
        encoder = BertModel(BertConfig(vocab_size=64, hidden_size=24, num_hidden_layers=1,
            num_attention_heads=4, intermediate_size=32, max_position_embeddings=32))
        model = BinaryMacBERT(encoder)
        inputs = {"input_ids": torch.tensor([[2, 4, 5, 3], [2, 6, 7, 3]]), "attention_mask": torch.ones(2, 4, dtype=torch.long)}
        logits = model(**inputs)
        self.assertEqual(tuple(logits.shape), (2, 2))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
        loss.backward()
        self.assertIsNotNone(model.cls.weight.grad)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.pt"
            torch.save({"state_dict": {"module." + k: v for k, v in model.state_dict().items()}}, path)
            stored = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict({k.removeprefix("module."): v for k, v in stored["state_dict"].items()}, strict=True)
        model.eval()
        with torch.inference_mode():
            probabilities = model(**inputs).softmax(-1)
        self.assertTrue(torch.allclose(probabilities.sum(-1), torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
