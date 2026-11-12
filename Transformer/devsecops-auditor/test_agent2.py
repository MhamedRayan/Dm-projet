"""
Smoke tests for Agent 2. Run from the project root:
    python test_agent2.py
Uses a tiny randomly-initialised BERT and a temp model directory, so it never
downloads/overwrites the real model (models/agent2_transformer) and never needs
the real dataset.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from agents import agent2_code as a2

tmp = Path(tempfile.mkdtemp())

# ------------------------------------------------ tiny local encoder + tokenizer
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
         "int", "main", "char", "buf", "strcpy", "return", "if", "else", "while",
         "for", "danger", "safe", "malloc", "free", "void", "0", "1"]
backend = Tokenizer(models.WordPiece({w: i for i, w in enumerate(vocab)},
                                     unk_token="[UNK]"))
backend.normalizer = normalizers.BertNormalizer(lowercase=True)
backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                              pad_token="[PAD]", cls_token="[CLS]",
                              sep_token="[SEP]", mask_token="[MASK]")
cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                 num_attention_heads=2, intermediate_size=64, max_position_embeddings=512)
tiny = BertModel(cfg)
(tmp / "tiny_model").mkdir()
tiny.save_pretrained(tmp / "tiny_model")
tok.save_pretrained(tmp / "tiny_model")

# redirect the agent away from the real model/cache and shrink the training cost
a2.MODEL_NAME = str(tmp / "tiny_model")
a2.ARTIFACT = tmp / "artifact"
a2.CACHE_DIR = tmp / "cache"
a2.MAX_LEN = 128
a2.BATCH_SIZE = 8
a2.HEAD_EPOCHS = 3
a2.HEAD_PATIENCE = 2
a2._bundle = None

# --------------------------------------------------------- synthetic jsonl splits
rng = np.random.default_rng(0)


def make_rows(n):
    rows = []
    for i in range(n):
        dangerous = rng.random() < 0.5
        words = ["int", "main", "char", "buf"]
        words += ["strcpy", "danger", "malloc"] if dangerous else ["safe", "return", "void"]
        # unique suffix so load_data()'s exact-duplicate filter keeps every row
        rows.append({"func": " ".join(words) + f" unique{i}", "target": int(dangerous),
                     "project": "synthetic"})
    return rows


data_dir = tmp / "data"
data_dir.mkdir()
for split, n in (("train", 600), ("validation", 150), ("test", 150)):
    with open(data_dir / f"{split}.jsonl", "w", encoding="utf-8") as fh:
        for row in make_rows(n):
            fh.write(json.dumps(row) + "\n")

# 1) training runs end-to-end and saves the artifact
a2.train(str(data_dir))
assert (a2.ARTIFACT / "head.pt").exists(), "head not saved"
assert (a2.ARTIFACT / "meta.json").exists(), "meta.json not saved"
assert (a2.ARTIFACT / "encoder").is_dir(), "encoder not saved"

# 2) valid input -> score in [0,1], status ok
r = a2.predict_code_risk("int main char buf strcpy malloc")
print("valid input   ->", r)
assert r["status"] == "ok" and 0.0 <= r["code_risk_score"] <= 1.0
assert isinstance(r["high_risk"], bool)

# 3) the signal is learned: dangerous-looking code scores higher than safe code
r_danger = a2.predict_code_risk("int main char buf strcpy danger malloc")
r_safe = a2.predict_code_risk("int main char buf safe return void")
print("dangerous code ->", r_danger, "| safe code ->", r_safe)
assert r_danger["code_risk_score"] > r_safe["code_risk_score"], \
    "model did not learn the training signal"

# 4) fail-safe: empty code -> error + None (Agent 3 must map to REVIEW)
r_empty = a2.predict_code_risk("   ")
print("empty code     ->", r_empty)
assert r_empty["status"] == "error" and r_empty["code_risk_score"] is None

# 5) fail-safe: non-string input -> error
r_none = a2.predict_code_risk(None)
print("None input     ->", r_none)
assert r_none["status"] == "error" and r_none["code_risk_score"] is None

# 6) long code is truncated, not rejected
r_long = a2.predict_code_risk("int main " * 500)
print("long code      ->", r_long)
assert r_long["status"] == "ok"

# 7) fail-safe: model files missing -> error, no crash
a2._bundle = None
a2.ARTIFACT = tmp / "does_not_exist"
r_nomodel = a2.predict_code_risk("int main() {}")
print("no model files ->", r_nomodel)
assert r_nomodel["status"] == "error" and r_nomodel["code_risk_score"] is None

print("\nALL TESTS PASSED")
