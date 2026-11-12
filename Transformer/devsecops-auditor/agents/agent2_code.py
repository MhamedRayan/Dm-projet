"""
Agent 2 - Source Code Vulnerability Detector (Transformer) | CodeXGLUE Defect Detection
Usage (run from the project root):
    python agents/agent2_code.py prepare --out data/codexglue   # dataset + encoder download
    python agents/agent2_code.py train --data data/codexglue
    python agents/agent2_code.py eval-external --n 5000          # optional DiverseVul check
    python -c "from agents.agent2_code import predict_code_risk; print(predict_code_risk('int main(){}'))"
Requires: torch transformers datasets numpy scikit-learn
NOTE: the encoder is FROZEN (pre-trained CodeBERTa-small); only a small MLP head is trained
on cached embeddings. Reason: CPU-only machine, full BERT fine-tuning is hours per epoch.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve, recall_score,
                             roc_auc_score)
from torch import nn
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "huggingface/CodeBERTa-small-v1"
HF_REPO = "huggingface/CodeBERTa-small-v1"
PRETRAINED_DIR = Path("data/pretrained/codeberta-small")
# Prefer the locally fetched copy (the HF hub client stalls on some networks); fall
# back to the hub id. `prepare` downloads into PRETRAINED_DIR before training.
if (PRETRAINED_DIR / "config.json").exists():
    MODEL_NAME = str(PRETRAINED_DIR)
MAX_LEN = 256                     # tokens; 512 doubles encoding time on CPU
BATCH_SIZE = 16
SEED = 42
ARTIFACT = Path("models/agent2_transformer")
CACHE_DIR = Path("data/cache")
HEAD_EPOCHS = 30
HEAD_PATIENCE = 4

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------- 1. Data
_MODEL_FILES = ("config.json", "vocab.json", "merges.txt", "tokenizer_config.json",
                "pytorch_model.bin")


def _download_file(url, target, force=False):
    """Plain-HTTP download with resume support. The huggingface_hub client stalls
    at 0 MB on this network (Xet/LFS), so we avoid it entirely."""
    import urllib.request
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            size = int(r.headers.get("content-length", 0))
    except Exception:
        size = 0
    if not force and target.exists() and (size == 0 or target.stat().st_size == size):
        print(f"  {target.name}: already present")
        return
    have = target.stat().st_size if target.exists() and not force else 0
    print(f"  {target.name}: downloading ({have / 1e6:.1f}/{size / 1e6:.1f} MB) ...",
          flush=True)
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=120) as r, \
            open(target, "ab" if have else "wb") as fh:
        written = have
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            written += len(chunk)
            if written % (20 << 20) < (1 << 20):
                print(f"    {written / 1e6:.0f} MB", flush=True)
    print(f"  {target.name}: done")


def fetch_encoder(force=False):
    """Download the frozen encoder into PRETRAINED_DIR."""
    dest = PRETRAINED_DIR
    for name in _MODEL_FILES:
        _download_file(f"https://huggingface.co/{HF_REPO}/resolve/main/{name}",
                       dest / name, force=force)
    print(f"Encoder ready -> {dest}")


def prepare(out_dir, force=False):
    """Download the encoder and CodeXGLUE Defect Detection (Devign) as JSONL files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("Encoder:")
    fetch_encoder(force=force)
    splits = ("train", "validation", "test")
    if not force and all((out / f"{s}.jsonl").exists() for s in splits):
        print(f"Dataset already prepared in {out} (use --force to re-download)")
        return
    from datasets import load_dataset
    ds = load_dataset("google/code_x_glue_cc_defect_detection")
    for split in splits:
        path = out / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for func, target, project in zip(ds[split]["func"], ds[split]["target"],
                                             ds[split]["project"]):
                fh.write(json.dumps({"func": func, "target": int(target),
                                     "project": project}) + "\n")
        print(f"{split}: {len(ds[split])} functions -> {path}")


def load_data(data_dir):
    """Read the JSONL splits; drop empty and exact-duplicate functions within a split."""
    data = Path(data_dir)
    out = {}
    for split in ("train", "validation", "test"):
        texts, labels, seen = [], [], set()
        with open(data / f"{split}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                code = (row.get("func") or "").strip()
                if not code or code in seen:
                    continue
                seen.add(code)
                texts.append(code)
                labels.append(int(bool(row["target"])))
        out[split] = (texts, np.asarray(labels, dtype=np.int64))
        print(f"{split}: {len(texts)} functions | vulnerable: {out[split][1].mean():.2%}")
    return out["train"], out["validation"], out["test"]


# ------------------------------------------------- 2. Frozen-transformer encoding
def _cache_path(split_file):
    split_file = Path(split_file)
    key = f"{split_file.resolve()}|{split_file.stat().st_mtime_ns}|{MODEL_NAME}|{MAX_LEN}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"emb_{split_file.stem}_{digest}.npy"


def encode_texts(texts, cache_path, batch_size=None):
    """Mean-pooled [CLS] embeddings from the frozen encoder, cached to disk."""
    batch_size = batch_size or BATCH_SIZE
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"Loading cached embeddings: {cache_path}")
        return np.load(cache_path)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tok(batch, truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt")
            hidden = encoder(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            chunks.append(pooled.cpu().numpy())
            if (start // batch_size) % 25 == 0:
                print(f"  encoded {start + len(batch)}/{len(texts)}", flush=True)
    emb = np.vstack(chunks).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    print(f"Embeddings {emb.shape} -> {cache_path}")
    return emb


# ------------------------------------------------------------- 3. Risk head
class RiskHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_head(X, y, X_val, y_val):
    torch.manual_seed(SEED)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    Xv, yv = torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.float32))
    model = RiskHead(X.shape[1])
    pos_weight = torch.tensor([(len(y) - y.sum()) / max(1, int(y.sum()))])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best_ap, best_state, bad = -1.0, None, 0
    bs = 256
    for epoch in range(1, HEAD_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(Xv)).numpy()
        ap = average_precision_score(y_val, val_scores)
        print(f"epoch {epoch:02d} | val PR-AUC {ap:.4f}")
        if ap > best_ap + 1e-4:
            best_ap, bad = ap, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= HEAD_PATIENCE:
                print(f"early stop (best val PR-AUC {best_ap:.4f})")
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def scores_for(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).numpy()


def choose_threshold(model, X_val, y_val):
    """F2-optimal threshold on the VALIDATION set only (never the test set)."""
    scores = scores_for(model, X_val)
    prec, rec, thr = precision_recall_curve(y_val, scores)
    f2 = 5 * prec[:-1] * rec[:-1] / (4 * prec[:-1] + rec[:-1] + 1e-9)
    return float(thr[int(np.argmax(f2))])


def evaluate(name, y, proba, threshold):
    print(f"\n=== {name} ===")
    for label, t in (("default 0.5", 0.5), (f"chosen {threshold:.3f}", threshold)):
        pred = (proba >= t).astype(int)
        print(f"\n--- Threshold: {label} ---")
        print(classification_report(y, pred, digits=3, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y, pred))
    print("\nROC-AUC:", round(roc_auc_score(y, proba), 3),
          "| PR-AUC:", round(average_precision_score(y, proba), 3))


def baselines(train_texts, y_tr, test_texts, y_te):
    dummy = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((len(y_tr), 1)), y_tr)
    print("\nBaseline (majority) recall(vulnerable):",
          round(recall_score(y_te, dummy.predict(np.zeros((len(y_te), 1))),
                             zero_division=0), 3))
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
    Xtr = tfidf.fit_transform(train_texts)
    Xte = tfidf.transform(test_texts)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, y_tr)
    proba = clf.predict_proba(Xte)[:, 1]
    print("TF-IDF + LogReg | ROC-AUC:", round(roc_auc_score(y_te, proba), 3),
          "| PR-AUC:", round(average_precision_score(y_te, proba), 3))


# ---------------------------------------------------------------- 4. Training
def train(data_dir):
    data = Path(data_dir)
    (train_texts, y_tr), (val_texts, y_val), (test_texts, y_te) = load_data(data)

    X_tr = encode_texts(train_texts, _cache_path(data / "train.jsonl"))
    X_val = encode_texts(val_texts, _cache_path(data / "validation.jsonl"))
    X_te = encode_texts(test_texts, _cache_path(data / "test.jsonl"))

    model = train_head(X_tr, y_tr, X_val, y_val)
    threshold = choose_threshold(model, X_val, y_val)
    print(f"Chosen threshold (val, F2): {threshold:.3f}")

    evaluate("Frozen CodeBERTa + MLP head (test)", y_te,
             scores_for(model, X_te), threshold)
    baselines(train_texts, y_tr, test_texts, y_te)

    ARTIFACT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME)
    tok.save_pretrained(ARTIFACT / "encoder")
    encoder.save_pretrained(ARTIFACT / "encoder")
    torch.save({"state_dict": model.state_dict(), "input_dim": int(X_tr.shape[1])},
               ARTIFACT / "head.pt")
    (ARTIFACT / "meta.json").write_text(json.dumps(
        {"model_name": MODEL_NAME, "max_len": MAX_LEN, "seed": SEED,
         "dataset": str(data_dir), "threshold": threshold,
         "head_epochs_max": HEAD_EPOCHS, "pooling": "attention-masked mean"},
        indent=2))
    print("Saved ->", ARTIFACT)


# --------------------------------------------------------------- 5. Inference
_bundle = None


def _error(reason):
    return {"code_risk_score": None, "high_risk": None, "status": "error", "reason": reason}


def _load_bundle():
    global _bundle
    if _bundle is None:
        tok = AutoTokenizer.from_pretrained(ARTIFACT / "encoder")
        encoder = AutoModel.from_pretrained(ARTIFACT / "encoder").eval()
        ckpt = torch.load(ARTIFACT / "head.pt", map_location="cpu", weights_only=True)
        head = RiskHead(ckpt["input_dim"])
        head.load_state_dict(ckpt["state_dict"])
        head.eval()
        meta = json.loads((ARTIFACT / "meta.json").read_text())
        _bundle = {"tok": tok, "encoder": encoder, "head": head,
                   "threshold": float(meta.get("threshold", 0.5)),
                   "max_len": int(meta.get("max_len", MAX_LEN))}
    return _bundle


def predict_code_risk(code: str) -> dict:
    """Standardized output for Agent 3. Never raises: on any problem (missing model,
    empty/non-string code) returns status='error' and score=None, so the gatekeeper
    must route to REVIEW (fail-safe), never to PASS.
    NOTE: code_risk_score is an uncalibrated model score, not a calibrated probability."""
    try:
        if not isinstance(code, str) or not code.strip():
            return _error("empty or non-string code")
        b = _load_bundle()
        score = float(_score_texts([code])[0])
        return {"code_risk_score": round(score, 4),
                "high_risk": score >= b["threshold"],
                "threshold": round(b["threshold"], 4),
                "status": "ok", "reason": ""}
    except Exception as exc:  # model files missing/corrupted, tokenizer failure, etc.
        return _error(str(exc))


def _score_texts(texts, batch_size=None):
    """Batched risk scores using the saved artifact (encoder + head + threshold)."""
    batch_size = batch_size or BATCH_SIZE
    b = _load_bundle()
    scores = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            enc = b["tok"](texts[start:start + batch_size], truncation=True,
                           max_length=b["max_len"], padding=True, return_tensors="pt")
            hidden = b["encoder"](**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            scores.append(torch.sigmoid(b["head"](pooled)).numpy())
            if len(texts) > batch_size and (start // batch_size) % 25 == 0:
                print(f"  scored {min(start + batch_size, len(texts))}/{len(texts)}",
                      flush=True)
    return np.concatenate(scores)


# ------------------------------------------------- 6. DiverseVul external check
DIVERSEVUL_BASE = "https://huggingface.co/datasets/claudios/DiverseVul/resolve/main/data/"


def fetch_diversevul(force=False):
    dest = Path("data/diversevul")
    for name in ("test-00000-of-00002.parquet", "test-00001-of-00002.parquet"):
        _download_file(DIVERSEVUL_BASE + name, dest / name, force=force)
    return dest


def external_eval(n=5000):
    """Zero-shot transfer of the trained agent to DiverseVul (never used for
    training or threshold tuning). Uses a stratified sample to stay CPU-affordable."""
    import pyarrow.parquet as pq
    dest = fetch_diversevul()
    funcs, targets = [], []
    for f in sorted(dest.glob("*.parquet")):
        table = pq.read_table(f, columns=["func", "target"])
        funcs.extend(table["func"].to_pylist())
        targets.extend(table["target"].to_pylist())
    y = np.asarray([int(bool(v)) for v in targets])
    print(f"DiverseVul test rows: {len(y)} | vulnerable: {y.mean():.2%}")
    rng = np.random.default_rng(SEED)
    k = min(n // 2, int((y == 1).sum()), int((y == 0).sum()))
    sel = np.concatenate([rng.choice(np.flatnonzero(y == 1), k, replace=False),
                          rng.choice(np.flatnonzero(y == 0), k, replace=False)])
    texts, ys, seen = [], [], set()
    for i in sel:
        code = (funcs[i] or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        texts.append(code)
        ys.append(int(y[i]))
    ys = np.asarray(ys)
    print(f"Evaluating on {len(texts)} unique functions | vulnerable: {ys.mean():.2%}")
    meta = json.loads((ARTIFACT / "meta.json").read_text())
    evaluate("DiverseVul external (zero-shot)", ys, _score_texts(texts),
             float(meta["threshold"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "train", "eval-external"])
    ap.add_argument("--out", default="data/codexglue", help="prepare: output directory")
    ap.add_argument("--data", default="data/codexglue", help="train: dataset directory")
    ap.add_argument("--n", type=int, default=5000, help="eval-external: sample size")
    ap.add_argument("--force", action="store_true", help="prepare: re-download files")
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare(args.out, force=args.force)
    elif args.mode == "train":
        train(args.data)
    else:
        external_eval(args.n)
