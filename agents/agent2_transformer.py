"""
Agent 2 - Transformer Vulnerability Detector (CodeBERT)
Wraps the fine-tuned model in models/agent2_transformer/.

Usage (run from the project root):
    from agents.agent2_transformer import predict_vuln
    predict_vuln("def f():\\n    return 1\\n")

    python agents/agent2_transformer.py --code "os.system(cmd)"
    python agents/agent2_transformer.py --file path/to/source.py

The model is loaded lazily on the first call and cached. Inference runs on GPU
when available, otherwise on CPU. Requires: torch, transformers.
"""
import argparse
import json
from pathlib import Path

ARTIFACT = Path("models/agent2_transformer")

# Loaded bundles are cached per artifact path, so tests can point ARTIFACT at a
# different directory without touching the real model.
_cache = {}


def _error(reason):
    return {"transformer_risk_score": None, "vulnerable": None, "threshold": None,
            "status": "error", "reason": reason}


def _load():
    """Load the tokenizer, model and metadata once per artifact directory."""
    key = str(ARTIFACT.resolve())
    if key in _cache:
        return _cache[key]

    # Imported lazily so the module still imports (and fails safe) when the
    # optional heavy dependencies are missing.
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    meta_path = ARTIFACT / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing metadata: {meta_path}")
    meta = json.loads(meta_path.read_text())

    tokenizer = AutoTokenizer.from_pretrained(ARTIFACT, use_fast=True,
                                              local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        ARTIFACT, local_files_only=True)
    model.eval()

    # Prefer the label mapping stored in the model config; fall back to meta.json.
    vuln_idx = None
    for name, idx in (getattr(model.config, "label2id", None) or {}).items():
        if str(name).strip().lower() == "vulnerable":
            vuln_idx = int(idx)
            break
    if vuln_idx is None:
        labels = [str(x).strip().lower() for x in meta.get("labels", [])]
        vuln_idx = labels.index("vulnerable") if "vulnerable" in labels else 1

    bundle = {
        "torch": torch,
        "tokenizer": tokenizer,
        "model": model,
        "max_len": int(meta.get("max_len", 384)),
        "threshold": float(meta.get("threshold", 0.5)),
        "vuln_idx": vuln_idx,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    model.to(bundle["device"])
    _cache[key] = bundle
    return bundle


def predict_vuln(code: str) -> dict:
    """Score one source file and return a standardized result for Agent 3.

    Returns:
        transformer_risk_score: probability that the code is vulnerable, in [0, 1]
        vulnerable:             True when the score is at/above the tuned threshold
        threshold:              threshold read from meta.json
        status:                 "ok", or "error" on any problem
        reason:                 human-readable error detail ("" when ok)

    Never raises: missing dependencies, a missing/corrupted model, or invalid
    input all return status="error" with score=None, so the gatekeeper must route
    to REVIEW (fail-safe), never PASS.
    """
    try:
        if not isinstance(code, str) or not code.strip():
            return _error("code must be a non-empty string")

        b = _load()
        torch = b["torch"]
        inputs = b["tokenizer"](
            code,
            return_tensors="pt",
            truncation=True,
            max_length=b["max_len"],
        )
        inputs = {k: v.to(b["device"]) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = b["model"](**inputs).logits

        if logits.shape[-1] == 1:  # single-logit head
            score = torch.sigmoid(logits[0, 0]).item()
        else:
            score = torch.softmax(logits, dim=-1)[0, b["vuln_idx"]].item()

        threshold = b["threshold"]
        return {"transformer_risk_score": round(float(score), 4),
                "vulnerable": bool(score >= threshold),
                "threshold": round(float(threshold), 4),
                "status": "ok", "reason": ""}
    except Exception as exc:  # missing model file, missing deps, bad input, ...
        return _error(f"{type(exc).__name__}: {exc}")


def _read_input(args):
    if args.code is not None:
        return args.code
    if args.file is not None:
        return Path(args.file).read_text(errors="replace")
    raise SystemExit("provide --code or --file")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score source code with Agent 2.")
    ap.add_argument("--code", help="source code as a string")
    ap.add_argument("--file", help="path to a source file")
    args = ap.parse_args()
    print(json.dumps(predict_vuln(_read_input(args)), indent=2))
