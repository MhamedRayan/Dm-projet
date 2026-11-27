"""
Tests for Agent 2. Run from the project root:
    python test_agent2.py

Loads the real model in models/agent2_transformer/ (about 500 MB; CPU is fine).
"""
import json
import tempfile
from pathlib import Path

from agents import agent2_transformer as a2

VULNERABLE_CODE = '''
import os

def run_user_command():
    cmd = input("command: ")
    os.system(cmd)
'''

SAFE_CODE = '''
def add(a, b):
    return a + b
'''

# 1) valid input -> score in [0,1], status ok, consistent threshold decision
r = a2.predict_vuln(VULNERABLE_CODE)
print("vulnerable sample ->", r)
assert r["status"] == "ok", r
assert 0.0 <= r["transformer_risk_score"] <= 1.0
assert isinstance(r["vulnerable"], bool)
assert r["vulnerable"] == (r["transformer_risk_score"] >= r["threshold"])

meta = json.loads(Path("models/agent2_transformer/meta.json").read_text())
assert abs(r["threshold"] - round(float(meta["threshold"]), 4)) < 1e-9, "threshold mismatch"

# 2) a second call reuses the cached model and gives the same score
r2 = a2.predict_vuln(VULNERABLE_CODE)
print("repeat call       ->", r2)
assert r2["status"] == "ok" and r2["transformer_risk_score"] == r["transformer_risk_score"]

# 3) safe-looking code is still scored successfully
r_safe = a2.predict_vuln(SAFE_CODE)
print("safe sample       ->", r_safe)
assert r_safe["status"] == "ok" and 0.0 <= r_safe["transformer_risk_score"] <= 1.0

# 4) fail-safe: invalid input -> error + None
for bad in (None, "", "   \n", 123, ["code"]):
    r_bad = a2.predict_vuln(bad)
    assert r_bad["status"] == "error" and r_bad["transformer_risk_score"] is None, r_bad
print("invalid inputs    -> fail safe")

# 5) fail-safe: missing model directory -> error, no crash
a2.ARTIFACT = Path(tempfile.mkdtemp()) / "does_not_exist"
r_nomodel = a2.predict_vuln(VULNERABLE_CODE)
print("no model dir      ->", r_nomodel)
assert r_nomodel["status"] == "error" and r_nomodel["transformer_risk_score"] is None

print("\nALL TESTS PASSED")
