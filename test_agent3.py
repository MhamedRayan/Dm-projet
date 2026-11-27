"""
Tests for Agent 3. Run from the project root:
    python test_agent3.py

Trains a temporary tree (the real models/agent3_dt.joblib is not touched).
"""
import json
import tempfile
from pathlib import Path

from agents import agent3_verdict as a3

tmp = Path(tempfile.mkdtemp())
a3.ARTIFACT = tmp / "agent3_dt.joblib"
a3._cache.clear()

# 1) training runs end-to-end and the tree reproduces the policy
a3.train(a3.ARTIFACT)
assert a3.ARTIFACT.exists(), "model artifact not saved"
meta_path = a3.ARTIFACT.with_suffix(".meta.json")
assert meta_path.exists(), "metadata not saved"
meta = json.loads(meta_path.read_text())
assert meta["policy"] == a3.POLICY
assert meta["policy_agreement"] >= 0.99, meta["policy_agreement"]
print("training          -> agreement", meta["policy_agreement"])

A1_THR = 0.3183
A2_THR = 0.51


def a1(score, status="ok"):
    high = score >= A1_THR if isinstance(score, (int, float)) else False
    return {"struct_risk_score": score, "high_risk": high,
            "threshold": A1_THR, "status": status,
            "reason": "" if status == "ok" else "agent1 failed"}


def a2(score, status="ok"):
    high = score >= A2_THR if isinstance(score, (int, float)) else False
    return {"transformer_risk_score": score, "vulnerable": high,
            "threshold": A2_THR, "status": status,
            "reason": "" if status == "ok" else "agent2 failed"}


def check(result, verdict, status="ok"):
    assert result["verdict"] == verdict, result
    assert result["status"] == status, result
    if status == "ok":
        assert 0.0 <= result["confidence"] <= 1.0, result
    else:
        assert result["confidence"] is None, result


# 2) consensus truth table
check(a3.decide(a1(0.10), a2(0.10)), "PASS")
check(a3.decide(a1(0.90), a2(0.90)), "FAIL")
r = a3.decide(a1(0.90), a2(0.10))
check(r, "REVIEW")
assert "disagree" in r["reason"], r
check(a3.decide(a1(0.10), a2(0.90)), "REVIEW")
print("truth table       -> PASS / FAIL / REVIEW / REVIEW")

# 3) deterministic
assert a3.decide(a1(0.90), a2(0.90)) == a3.decide(a1(0.90), a2(0.90))

# 4) threshold fallback from the saved bundle metadata
a1_no_thr = a1(0.90)
del a1_no_thr["threshold"]
check(a3.decide(a1_no_thr, a2(0.90)), "FAIL")
print("threshold fallback-> ok")

# 5) fail-safe: agent errors
check(a3.decide(a1(0.10, status="error"), a2(0.10)), "REVIEW", "error")
check(a3.decide(a1(0.10), a2(0.10, status="error")), "REVIEW", "error")
print("agent errors      -> REVIEW")

# 6) fail-safe: missing / invalid inputs
for bad_a1, bad_a2 in (
    ({}, a2(0.10)),
    (a1(0.10), {}),
    (a1(float("nan")), a2(0.10)),
    (a1(float("inf")), a2(0.10)),
    (a1(1.5), a2(0.10)),
    (a1("0.5"), a2(0.10)),
    (a1(0.10), a2(-0.1)),
    (a1(0.10), a2(float("nan"))),
):
    check(a3.decide(bad_a1, bad_a2), "REVIEW", "error")
print("invalid inputs    -> REVIEW (error)")

# 7) fail-safe: no threshold anywhere -> error
bundle = a3._load()
saved = bundle["source_thresholds"]
bundle["source_thresholds"] = {}
check(a3.decide(a1_no_thr, a2(0.90)), "REVIEW", "error")
bundle["source_thresholds"] = saved

# 8) safety guard: a corrupted tree cannot produce an unsafe verdict
class _AlwaysPass:
    def predict_proba(self, _features):
        # classes are sorted: FAIL, PASS, REVIEW
        return [[0.0, 1.0, 0.0]]


key = str(a3.ARTIFACT.resolve())
a3._cache[key] = {"model": _AlwaysPass(), "feature_cols": a3.FEATURE_COLS,
                  "classes": ["FAIL", "PASS", "REVIEW"], "source_thresholds": {}}
check(a3.decide(a1(0.90), a2(0.90)), "REVIEW", "error")
a3._cache.pop(key, None)
print("policy guard      -> mismatch forced REVIEW")

# 9) fail-safe: missing model file
a3.ARTIFACT = tmp / "does_not_exist.joblib"
a3._cache.clear()
check(a3.decide(a1(0.10), a2(0.10)), "REVIEW", "error")
print("no model file     -> REVIEW (error)")

print("\nALL TESTS PASSED")
