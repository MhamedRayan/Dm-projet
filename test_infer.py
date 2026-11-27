"""
Tests for the inference CLI. Run from the project root:
    python test_infer.py

Uses fake agents, so it does not load either model.
"""
import contextlib
import io
import json
import tempfile
from pathlib import Path

import infer

TMP = Path(tempfile.mkdtemp())

VULN = TMP / "vuln.py"
VULN.write_text("import os\nos.system('whoami')\n")
SAFE = TMP / "safe.py"
SAFE.write_text("def add(a, b):\n    return a + b\n")
LONELY = TMP / "lonely.py"
LONELY.write_text("def f():\n    return 1\n")

VULN_METRICS = TMP / "vuln.metrics.json"
VULN_METRICS.write_text(json.dumps({"loc": 200}))
SAFE_METRICS = TMP / "safe.metrics.json"
SAFE_METRICS.write_text(json.dumps({"loc": 8}))


def fake_a1(metrics):
    if not metrics:
        return infer._agent1_error("missing features: ['loc', ...]")
    score = 0.9 if float(metrics.get("loc", 0)) > 100 else 0.1
    return {"struct_risk_score": score, "high_risk": score >= 0.5,
            "threshold": 0.5, "status": "ok", "reason": ""}


def fake_a2(code):
    score = 0.9 if "os.system" in code else 0.1
    return {"transformer_risk_score": score, "vulnerable": score >= 0.5,
            "threshold": 0.5, "status": "ok", "reason": ""}


def fake_decide(a1, a2):
    if a1["status"] != "ok" or a2["status"] != "ok":
        return {"verdict": "REVIEW", "confidence": None, "status": "error",
                "reason": "agent error"}
    h1 = a1["struct_risk_score"] >= a1["threshold"]
    h2 = a2["transformer_risk_score"] >= a2["threshold"]
    if h1 and h2:
        return {"verdict": "FAIL", "confidence": 1.0, "status": "ok", "reason": "both"}
    if not h1 and not h2:
        return {"verdict": "PASS", "confidence": 1.0, "status": "ok", "reason": ""}
    return {"verdict": "REVIEW", "confidence": 1.0, "status": "ok", "reason": "disagree"}


infer.predict_struct_risk = fake_a1
infer.predict_vuln = fake_a2
infer.decide = fake_decide


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = infer.main(argv)
    return code, out.getvalue(), err.getvalue()


try:
    # 1) single file with metrics -> FAIL, exit 1
    code, out, err = run(["--code", str(VULN), "--metrics", str(VULN_METRICS)])
    print("single full       ->", code)
    assert code == 1, (code, out, err)
    assert "verdict=FAIL" in out and "File:" in out
    assert err == ""

    # 2) code only -> Agent 1 error, REVIEW, hint on stderr
    code, out, err = run(["--code", str(SAFE)])
    print("code only         ->", code)
    assert code == 2, (code, out, err)
    assert "status=error" in out
    assert "Agent 1 needs metrics" in err

    # 3) batch with paired metrics -> PASS + FAIL, exit 1
    code, out, err = run(["--code", str(VULN), str(SAFE),
                          "--metrics-dir", str(TMP), "--json"])
    print("batch json        ->", code)
    assert code == 1, (code, out, err)
    payload = json.loads(out)
    assert len(payload["results"]) == 2
    assert payload["summary"] == {"PASS": 1, "REVIEW": 0, "FAIL": 1}
    assert payload["warnings"] == []
    assert err == ""

    # 4) batch human output includes a summary
    code, out, err = run(["--code", str(VULN), str(SAFE), "--metrics-dir", str(TMP)])
    assert code == 1
    assert "Summary: PASS=1 REVIEW=0 FAIL=1" in out

    # 5) missing paired metrics -> warning + REVIEW
    code, out, err = run(["--code", str(LONELY), "--metrics-dir", str(TMP)])
    print("missing metrics   ->", code)
    assert code == 2, (code, out, err)
    assert "no metrics supplied" in err

    # 6) JSON includes warnings
    code, out, err = run(["--code", str(LONELY), "--json"])
    assert code == 2
    payload = json.loads(out)
    assert payload["warnings"]
    assert payload["summary"]["REVIEW"] == 1

    # 7) --no-hints suppresses stderr warnings
    code, out, err = run(["--code", str(SAFE), "--no-hints"])
    assert code == 2
    assert err == ""

    # 8) missing code file -> usage error 3
    code, out, err = run(["--code", str(TMP / "nope.py")])
    print("missing code      ->", code)
    assert code == 3, (code, out, err)
    assert "code file not found" in err

    # 9) malformed metrics JSON -> usage error 3
    bad = TMP / "bad.json"
    bad.write_text("{not json")
    code, out, err = run(["--code", str(VULN), "--metrics", str(bad)])
    print("bad metrics       ->", code)
    assert code == 3, (code, out, err)
    assert "invalid metrics JSON" in err

    # 10) --metrics with multiple files -> usage error 3
    code, out, err = run(["--code", str(VULN), str(SAFE), "--metrics", str(VULN_METRICS)])
    assert code == 3
    assert "--metrics-dir" in err

    # 11) --metrics and --metrics-dir together -> usage error 3
    code, out, err = run(["--code", str(VULN), "--metrics", str(VULN_METRICS),
                          "--metrics-dir", str(TMP)])
    assert code == 3
    assert "either --metrics or --metrics-dir" in err

    print("\nALL TESTS PASSED")
finally:
    infer.predict_struct_risk = None
    infer.predict_vuln = None
    infer.decide = None
