"""
Agent 3 - Verdict Gatekeeper (Decision Tree)
Combines Agent 1 (code metrics) and Agent 2 (transformer) outputs into a verdict.

Policy (consensus + fail-safe):
    Agent 1 low  + Agent 2 safe        -> PASS
    Agent 1 high + Agent 2 vulnerable  -> FAIL
    the two disagree                   -> REVIEW
    any error / missing / invalid data -> REVIEW

The tree is trained on a synthetic grid labelled by that policy, so it encodes
the policy rather than discovering it from ground-truth verdicts. A post-check
ensures the tree can never return an unsafe PASS/FAIL.

Usage (run from the project root):
    python agents/agent3_verdict.py train
    from agents.agent3_verdict import decide
    decide(agent1_result, agent2_result)
"""
import argparse
import json
import math
import numbers
from pathlib import Path

ARTIFACT = Path("models/agent3_dt.joblib")
POLICY = "consensus_fail_safe"
VERDICTS = ("PASS", "REVIEW", "FAIL")
FEATURE_COLS = ["a1_high_risk", "a2_vulnerable", "a1_risk_ratio", "a2_risk_ratio"]
SEED = 42
MAX_DEPTH = 3

_cache = {}


def _error(reason):
    return {"verdict": "REVIEW", "confidence": None, "status": "error", "reason": reason}


def _is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _load():
    """Load and validate the model bundle once per artifact path."""
    key = str(ARTIFACT.resolve())
    if key in _cache:
        return _cache[key]

    import joblib

    if not ARTIFACT.is_file():
        raise FileNotFoundError(f"missing model: {ARTIFACT}")
    bundle = joblib.load(ARTIFACT)
    if not isinstance(bundle, dict):
        raise ValueError("invalid model bundle: expected a dict")
    for field in ("model", "feature_cols", "classes"):
        if field not in bundle:
            raise ValueError(f"model bundle missing {field!r}")
    if list(bundle["feature_cols"]) != FEATURE_COLS:
        raise ValueError("model feature order does not match this version of Agent 3")
    _cache[key] = bundle
    return bundle


def _policy_verdict(a1_high, a2_high):
    """The explicit consensus policy, also used to generate training labels."""
    if a1_high and a2_high:
        return "FAIL", "both agents flagged risk"
    if not a1_high and not a2_high:
        return "PASS", ""
    return "REVIEW", "agents disagree"


def _read_agent(result, score_key, threshold_key, bundle, name):
    """Validate one agent result and return (high_risk, risk_ratio, threshold)."""
    if not isinstance(result, dict):
        raise ValueError(f"{name} result must be a dict")
    if result.get("status") != "ok":
        detail = result.get("reason", "")
        raise ValueError(f"{name} status is {result.get('status')!r}"
                         + (f": {detail}" if detail else ""))

    score = result.get(score_key)
    if not _is_number(score):
        raise ValueError(f"{name} missing/invalid {score_key}: {score!r}")
    score = float(score)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} {score_key} out of range: {score!r}")

    threshold = result.get(threshold_key)
    if threshold is None:
        threshold = (bundle.get("source_thresholds") or {}).get(name)
    if not _is_number(threshold):
        raise ValueError(f"{name} missing/invalid {threshold_key}: {threshold!r}")
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"{name} invalid threshold: {threshold!r}")

    return score >= threshold, score / threshold, threshold


def decide(agent1_result: dict, agent2_result: dict) -> dict:
    """Return the final verdict for one file.

    Output:
        {"verdict": "PASS"|"REVIEW"|"FAIL", "confidence": float,
         "status": "ok"|"error", "reason": str}

    Never raises. Any error, missing field, invalid score, or missing model
    returns REVIEW with status="error", so the gate can never fail open.
    """
    try:
        bundle = _load()
        a1_high, a1_ratio, _ = _read_agent(
            agent1_result, "struct_risk_score", "threshold", bundle, "agent1")
        a2_high, a2_ratio, _ = _read_agent(
            agent2_result, "transformer_risk_score", "threshold", bundle, "agent2")

        features = [[float(a1_high), float(a2_high), a1_ratio, a2_ratio]]
        proba = bundle["model"].predict_proba(features)[0]
        idx = int(proba.argmax())
        verdict = str(bundle["classes"][idx])
        confidence = float(proba[idx])

        expected, reason = _policy_verdict(a1_high, a2_high)
        if verdict != expected:
            return _error(f"gatekeeper/policy mismatch: model={verdict}, policy={expected}")

        return {"verdict": verdict, "confidence": round(confidence, 4),
                "status": "ok", "reason": reason}
    except Exception as exc:
        return _error(f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------ Training
def _training_ratios():
    """Risk ratios around the decision boundary (1.0 = at the agent threshold)."""
    ratios = {round(i * 0.05, 3) for i in range(41)}  # 0.00 .. 2.00
    ratios.update({0.9, 0.95, 0.99, 0.999, 1.0, 1.001, 1.01, 1.05, 1.1})
    return sorted(ratios)


def _training_data():
    ratios = _training_ratios()
    X, y = [], []
    for a1_high in (0, 1):
        for a2_high in (0, 1):
            label, _ = _policy_verdict(bool(a1_high), bool(a2_high))
            for r1 in ratios:
                for r2 in ratios:
                    X.append([a1_high, a2_high, r1, r2])
                    y.append(label)
    return X, y


def _source_thresholds():
    """Record the current agent thresholds in the metadata when available."""
    thresholds = {}
    for name, path in (
        ("agent1", Path("models/agent1_rf.meta.json")),
        ("agent2", Path("models/agent2_transformer/meta.json")),
    ):
        try:
            thresholds[name] = float(json.loads(path.read_text())["threshold"])
        except Exception:
            pass
    return thresholds


def train(out=ARTIFACT):
    """Train the gatekeeper on the synthetic policy grid and save the artifact."""
    from sklearn.tree import DecisionTreeClassifier, export_text

    X, y = _training_data()
    model = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=SEED)
    model.fit(X, y)

    classes = [str(c) for c in model.classes_]
    agreement = float((model.predict(X) == y).mean())
    thresholds = _source_thresholds()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump({"model": model, "feature_cols": FEATURE_COLS, "classes": classes,
                 "policy": POLICY, "source_thresholds": thresholds}, out)

    meta = {
        "policy": POLICY,
        "features": FEATURE_COLS,
        "classes": classes,
        "max_depth": MAX_DEPTH,
        "seed": SEED,
        "grid": {"ratio_min": 0.0, "ratio_max": 2.0,
                 "ratios": len(_training_ratios()), "rows": len(X)},
        "policy_agreement": round(agreement, 4),
        "train_accuracy": round(float(model.score(X, y)), 4),
        "source_thresholds": thresholds,
        "note": ("Synthetic policy grid; the tree encodes the consensus policy, "
                 "not ground-truth verdicts."),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    print(export_text(model, feature_names=FEATURE_COLS))
    print(f"Policy agreement: {agreement:.3f}")
    print("Saved ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the Agent 3 gatekeeper.")
    ap.add_argument("mode", choices=["train"])
    ap.add_argument("--out", default=str(ARTIFACT))
    args = ap.parse_args()
    if args.mode == "train":
        train(args.out)
