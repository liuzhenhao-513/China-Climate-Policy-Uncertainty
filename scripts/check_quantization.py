"""Does int8 dynamic quantization change MacBERT's test-set predictions? How much faster is it?

python scripts/check_quantization.py   ->   outputs/validation/quantization_check.json
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccpu.common import read_config, write_json  # noqa: E402
from ccpu.labels import read_labels  # noqa: E402
from ccpu.model import evaluate, load_classifier  # noqa: E402

cfg = read_config()
test = read_labels(ROOT / "data" / "labels" / "splits" / "test.csv")
result = {}
for quantize in (False, True):
    model, tokenizer, _ = load_classifier(ROOT / "models" / "macbert-ccpu", None, "cpu", quantize)
    start = time.time()
    metrics, predictions, _ = evaluate(model, tokenizer, test, "cpu", 32, cfg["max_length"], True)
    seconds = time.time() - start
    result["int8" if quantize else "fp32"] = {"metrics": metrics, "seconds": round(seconds, 1),
                                               "articles_per_second": round(len(test) / seconds, 2),
                                               "predictions": predictions}
a, b = result["fp32"].pop("predictions"), result["int8"].pop("predictions")
result["agreement"] = sum(x == y for x, y in zip(a, b)) / len(a)
result["speedup"] = round(result["fp32"]["seconds"] / result["int8"]["seconds"], 2)
write_json(ROOT / "outputs" / "validation" / "quantization_check.json", result)
print(result)
