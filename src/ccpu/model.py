"""MacBERT encoder + CLS linear head, compatible with the released key names.

Heavy imports live only in this module: data work does not require a GPU or torch.
The design follows models.py in the authors' released code (Ma et al., Figshare 10.6084/m9.figshare.24071250).
"""
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer
from .common import fingerprint, logger, rel_path, utc_now, write_csv, write_json
from .labels import read_labels, validate_disjoint
from .metrics import binary_metrics
from .text import clean_model_text


class BinaryMacBERT(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.nlp_model = encoder
        self.cls = nn.Linear(encoder.config.hidden_size, 2)

    def forward(self, **inputs):
        hidden = self.nlp_model(**inputs).last_hidden_state[:, 0]
        return self.cls(hidden)


def device_for(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LabeledTexts(Dataset):
    """Paper cleaning (no punctuation/whitespace/URLs) is applied at model input."""
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return clean_model_text(row["text"]), row["label"]


def token_batch(tokenizer, texts, device, max_length):
    return {k: v.to(device) for k, v in tokenizer(list(texts), padding=True, truncation=True,
        max_length=max_length, return_tensors="pt").items()}


def evaluate(model, tokenizer, records, device, batch_size=8, max_length=500, return_scores=False):
    model.eval()
    truth, predictions, scores, total_loss = [], [], [], 0.0
    loader = DataLoader(LabeledTexts(records), batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for texts, labels in loader:
            logits = model(**token_batch(tokenizer, texts, device, max_length))
            total_loss += nn.functional.cross_entropy(logits, labels.to(device), reduction="sum").item()
            truth.extend(labels.tolist())
            predictions.extend(logits.argmax(-1).cpu().tolist())
            scores.extend(logits.softmax(-1)[:, 1].cpu().tolist())
    metrics = binary_metrics(truth, predictions)
    metrics["cross_entropy"] = total_loss / len(records)
    return (metrics, predictions, scores) if return_scores else metrics


def train_model(train_file, validation_file, output, base_model="hfl/chinese-macbert-base",
                test_file=None, device="auto", epochs=15, batch_size=4, learning_rate=1e-5,
                max_length=500, seed=666, label_smoothing=0.2, freeze_layers=0):
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or not 0 <= label_smoothing < 1:
        raise ValueError("Invalid training hyperparameters")
    seed_all(seed)
    train = read_labels(train_file)
    validation = read_labels(validation_file)
    test = read_labels(test_file) if test_file else []
    validate_disjoint(train, validation, test)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    encoder = AutoModel.from_pretrained(base_model)
    if not 2 <= max_length <= encoder.config.max_position_embeddings:
        raise ValueError("max_length exceeds encoder limits")
    run_device = device_for(device)
    model = BinaryMacBERT(encoder).to(run_device)
    # Freeze embeddings and the lowest `freeze_layers` transformer blocks (CPU budget).
    if freeze_layers:
        for p in encoder.embeddings.parameters():
            p.requires_grad = False
        for layer in encoder.encoder.layer[:freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, eps=1e-8, weight_decay=0.01)
    loader = DataLoader(LabeledTexts(train), batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed))
    total_steps = epochs * len(loader)
    warmup = max(1, int(0.1 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / warmup, max(0.0, (total_steps - s) / max(1, total_steps - warmup))))
    # Inverse-frequency class weights: positives are a minority (~17%).
    n_pos = sum(r["label"] for r in train)
    class_weight = (torch.tensor([len(train) / (2 * (len(train) - n_pos)), len(train) / (2 * n_pos)], device=run_device)
                    if 0 < n_pos < len(train) else torch.ones(2, device=run_device))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log = logger(output / "train.log")
    log(f"train={len(train)} validation={len(validation)} test={len(test)} device={run_device} "
        f"base={rel_path(base_model)} epochs={epochs} batch={batch_size} lr={learning_rate} max_length={max_length} "
        f"frozen_layers={freeze_layers} trainable_params={sum(p.numel() for p in trainable):,} class_weight={class_weight.tolist()}")
    best_key, history = None, []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, (texts, labels) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(**token_batch(tokenizer, texts, run_device, max_length))
            loss = nn.functional.cross_entropy(logits, labels.to(run_device), weight=class_weight,
                                               label_smoothing=label_smoothing)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(labels)
            if step % 25 == 0:
                log(f"epoch={epoch} step={step}/{len(loader)} loss={loss.item():.4f}")
        metrics = evaluate(model, tokenizer, validation, run_device, batch_size, max_length)
        entry = {"epoch": epoch, "train_loss": total_loss / len(train), **metrics}
        history.append(entry)
        log(f"epoch={epoch} train_loss={entry['train_loss']:.5f} val_loss={metrics['cross_entropy']:.5f} val_f1={metrics['f1']:.5f}")
        key = (metrics["f1"], -metrics["cross_entropy"])  # select on validation F1, then loss
        if best_key is None or key > best_key:
            best_key = key
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, output / "best.pt")
            encoder.config.save_pretrained(output)
            tokenizer.save_pretrained(output)
        write_json(output / "history.json", history)
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    # Decision threshold chosen on the validation set (maximum F1), never on the test set.
    _, _, val_scores = evaluate(model, tokenizer, validation, run_device, batch_size, max_length, True)
    val_truth = [r["label"] for r in validation]
    threshold = max((t / 100 for t in range(5, 96)),
                    key=lambda t: (binary_metrics(val_truth, [int(s >= t) for s in val_scores])["f1"], -abs(t - 0.5)))
    log(f"validation-selected threshold={threshold:.2f}")
    test_metrics = test_argmax = None
    if test:
        test_argmax, _, scores = evaluate(model, tokenizer, test, run_device, batch_size, max_length, True)
        predictions = [int(s >= threshold) for s in scores]
        test_metrics = binary_metrics([r["label"] for r in test], predictions)
        log(f"test argmax f1={test_argmax['f1']:.4f}; test f1 at threshold {threshold:.2f} = {test_metrics['f1']:.4f}")
        write_csv(output / "test_predictions.csv", ["article_id", "label", "macbert_label", "macbert_probability"],
                  ({"article_id": r.get("article_id", r["text_sha256"]), "label": r["label"],
                    "macbert_label": p, "macbert_probability": round(s, 6)} for r, p, s in zip(test, predictions, scores)))
    result = {"generated_at_utc": utc_now(), "base_model": rel_path(base_model), "seed": seed,
        "device": str(run_device), "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "label_smoothing": label_smoothing,
        "max_length": max_length, "frozen_layers": freeze_layers, "class_weight": class_weight.tolist(),
        "scheduler": "linear warmup (10%) and linear decay", "checkpoint_selection": "maximum validation F1 (ties: lower cross entropy)",
        "best_epoch": checkpoint["epoch"], "train_n": len(train), "validation_n": len(validation),
        "validation": next(h for h in history if h["epoch"] == checkpoint["epoch"]),
        "decision_threshold": threshold, "threshold_selection": "maximum validation F1",
        "test": test_metrics, "test_argmax": test_argmax,
        "input_sha256": {rel_path(p): fingerprint(p) for p in [train_file, validation_file, *([test_file] if test_file else [])]},
        "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__,
        "note": "Model selection and hardware handling differ from the released script; these are new run results, not the paper's reported metrics."}
    write_json(output / "training_run.json", result)
    return result


def load_classifier(checkpoint, base_model, device, quantize=False):
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(checkpoint, local_files_only=True))
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        weight_path = checkpoint / "best.pt"
    else:
        if not base_model:
            raise ValueError("Original .pth.tar weights require --base-model (matching original encoder/tokenizer)")
        encoder = AutoModel.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        weight_path = checkpoint
    model = BinaryMacBERT(encoder)
    stored = torch.load(weight_path, map_location="cpu", weights_only=True)
    weights = stored.get("state_dict", stored)
    weights = {k.removeprefix("module."): v for k, v in weights.items()}
    model.load_state_dict(weights, strict=True)
    model = model.to(device).eval()
    if quantize:  # int8 dynamic quantization of Linear layers: ~2x faster CPU inference
        model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    return model, tokenizer, weight_path


def predict_articles(articles, checkpoint, output, base_model=None, device="auto", batch_size=8, max_length=500,
                     quantize=False):
    csv.field_size_limit(100_000_000)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    run_device = device_for(device)
    model, tokenizer, weight_path = load_classifier(checkpoint, base_model, run_device, quantize)
    threshold = 0.5
    run_file = Path(checkpoint) / "training_run.json"
    if Path(checkpoint).is_dir() and run_file.exists():
        threshold = json.loads(run_file.read_text(encoding="utf-8")).get("decision_threshold", 0.5)
    if not 2 <= max_length <= model.nlp_model.config.max_position_embeddings:
        raise ValueError("max_length exceeds model limits")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    log = logger(output.with_suffix(".log"))
    fields = ["article_id", "date", "newspaper", "cpu_probability", "cpu_label"]
    done = set()
    if temporary.exists():  # resume an interrupted multi-hour CPU run
        with temporary.open(encoding="utf-8-sig", newline="") as g:
            done = {r["article_id"] for r in csv.DictReader(g)}
        log(f"Resuming: {len(done):,} articles already predicted")
    seen, n = set(), 0
    with Path(articles).open(encoding="utf-8-sig", newline="") as f, \
         temporary.open("a" if done else "w", encoding="utf-8-sig" if not done else "utf-8", newline="") as g:
        reader = csv.DictReader(f)
        text_field = "focus_text" if "focus_text" in (reader.fieldnames or []) else "text"
        if not {"article_id", "date", "newspaper", text_field}.issubset(reader.fieldnames or []):
            raise ValueError("Input needs article_id, date, newspaper and focus_text (build-corpus) or text (prepare)")
        writer = csv.DictWriter(g, fieldnames=fields)
        if not done:
            writer.writeheader()
        batch = []
        def consume(records):
            texts = [clean_model_text(r[text_field]) for r in records]
            with torch.inference_mode():
                logits = model(**token_batch(tokenizer, texts, run_device, max_length))
                probabilities = logits.softmax(-1)[:, 1].cpu().tolist()
                labels = [int(p >= threshold) for p in probabilities]
            for row, probability, label in zip(records, probabilities, labels):
                writer.writerow({**{k: row[k] for k in fields[:3]}, "cpu_probability": round(probability, 6), "cpu_label": label})
            g.flush()
        for row in reader:
            if not row["article_id"] or row["article_id"] in seen or not row[text_field].strip():
                raise ValueError("Missing/duplicate article ID or empty text")
            seen.add(row["article_id"])
            n += 1
            if row["article_id"] in done:
                continue
            batch.append(row)
            if len(batch) == batch_size:
                consume(batch)
                batch = []
            if n % 2000 == 0:
                log(f"Predicted {n:,} articles")
        if batch:
            consume(batch)
    temporary.replace(output)
    write_json(output.with_suffix(".meta.json"), {"generated_at_utc": utc_now(), "rows": n,
        "checkpoint_sha256": fingerprint(weight_path), "input_sha256": fingerprint(articles),
        "device": str(run_device), "max_length": max_length, "batch_size": batch_size, "int8_dynamic_quantization": quantize,
        "label_rule": f"cpu_label = 1 if P(CPU) >= {threshold} (validation-selected threshold; 0.5 = argmax)",
        "cleaning": "clean_model_text applied to the focus window; tokenizer truncates to max_length tokens.",
        "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__})
    return n
