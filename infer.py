"""
Inference CLI for the DevSecOps auditor.

Runs Agent 1 (metrics) -> Agent 2 (transformer) -> Agent 3 (gatekeeper) on one
or more source files.

Examples (run from the project root):
    python infer.py --code examples/vulnerable.py \
        --metrics examples/vulnerable.metrics.json
    python infer.py --code examples/vulnerable.py
    python infer.py --code examples/*.py --metrics-dir examples/
    python infer.py --code examples/vulnerable.py \
        --metrics examples/vulnerable.metrics.json --json

Exit codes: 0 = all PASS, 1 = any FAIL, 2 = any REVIEW/error, 3 = usage error.
"""
import argparse
import json
import sys
from pathlib import Path

# Agent 1 needs these exact PC1 metrics; aliases like v(g) are normalized by Agent 1.
REQUIRED_METRICS = [
    "loc", "vg", "evg", "ivg", "n", "v", "l", "d", "i", "e", "b", "t",
    "locode", "locomment", "loccodeandcomment", "loblank",
    "uniq_op", "uniq_opnd", "total_op", "total_opnd", "branchcount",
]

# Resolved lazily so the CLI still starts (and fails safe) without optional deps.
predict_struct_risk = None
predict_vuln = None
decide = None


class _UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        raise _UsageError(message)


def _agent1_error(reason):
    return {"struct_risk_score": None, "high_risk": None, "threshold": None,
            "status": "error", "reason": reason}


def _agent2_error(reason):
    return {"transformer_risk_score": None, "vulnerable": None, "threshold": None,
            "status": "error", "reason": reason}


def _agent1(metrics):
    global predict_struct_risk
    try:
        if predict_struct_risk is None:
            from agents.agent1_metrics import predict_struct_risk as fn
            predict_struct_risk = fn
        return predict_struct_risk(metrics)
    except Exception as exc:
        return _agent1_error(f"{type(exc).__name__}: {exc}")


def _agent2(code):
    global predict_vuln
    try:
        if predict_vuln is None:
            from agents.agent2_transformer import predict_vuln as fn
            predict_vuln = fn
        return predict_vuln(code)
    except Exception as exc:
        return _agent2_error(f"{type(exc).__name__}: {exc}")


def _agent3(agent1_result, agent2_result):
    global decide
    try:
        if decide is None:
            from agents.agent3_verdict import decide as fn
            decide = fn
        return decide(agent1_result, agent2_result)
    except Exception as exc:
        return {"verdict": "REVIEW", "confidence": None, "status": "error",
                "reason": f"{type(exc).__name__}: {exc}"}


def _read_code(path):
    if path == "-":
        return "<stdin>", sys.stdin.read()
    p = Path(path)
    if not p.is_file():
        raise _UsageError(f"code file not found: {path}")
    try:
        return str(p), p.read_text(errors="replace")
    except OSError as exc:
        raise _UsageError(f"cannot read code file {path}: {exc}") from exc


def _load_metrics(path):
    p = Path(path)
    if not p.is_file():
        raise _UsageError(f"metrics file not found: {path}")
    try:
        data = json.loads(p.read_text())
    except OSError as exc:
        raise _UsageError(f"cannot read metrics file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise _UsageError(f"invalid metrics JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise _UsageError(f"metrics file must contain a JSON object: {path}")
    return data


def _metrics_for(code_path, args):
    if args.metrics:
        return _load_metrics(args.metrics), args.metrics
    if args.metrics_dir and code_path != "-":
        candidate = Path(args.metrics_dir) / f"{Path(code_path).stem}.metrics.json"
        if candidate.is_file():
            return _load_metrics(candidate), str(candidate)
    return None, None


def _run_one(code_path, args, warnings):
    label, code = _read_code(code_path)
    metrics, metrics_source = _metrics_for(code_path, args)

    lacked_metrics = not metrics
    if lacked_metrics:
        metrics = {}
        warnings.append(
            f"{label}: no metrics supplied; Agent 1 cannot run, so the verdict will be REVIEW.")

    a1 = _agent1(metrics)
    a2 = _agent2(code)
    a3 = _agent3(a1, a2)
    record = {"file": label, "metrics": metrics_source,
              "agent1": a1, "agent2": a2, "agent3": a3}
    return record, lacked_metrics


def _num(value):
    if value is None:
        return "None"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _agent1_line(result):
    if result.get("status") == "ok":
        return (f"status=ok    score={_num(result.get('struct_risk_score'))} "
                f"high_risk={result.get('high_risk')} "
                f"threshold={_num(result.get('threshold'))}")
    return f"status=error reason={result.get('reason', '')}"


def _agent2_line(result):
    if result.get("status") == "ok":
        return (f"status=ok    score={_num(result.get('transformer_risk_score'))} "
                f"vulnerable={result.get('vulnerable')} "
                f"threshold={_num(result.get('threshold'))}")
    return f"status=error reason={result.get('reason', '')}"


def _agent3_line(result):
    verdict = result.get("verdict", "REVIEW")
    if result.get("status") == "ok":
        reason = result.get("reason") or ""
        line = f"verdict={verdict} confidence={_num(result.get('confidence'))}"
        return line + (f" reason={reason}" if reason else "")
    return f"verdict={verdict} status=error reason={result.get('reason', '')}"


def _print_human(results, summary, warnings, show_summary, show_hints):
    for i, record in enumerate(results):
        if i:
            print()
        print(f"File: {record['file']}")
        print(f"Agent 1: {_agent1_line(record['agent1'])}")
        print(f"Agent 2: {_agent2_line(record['agent2'])}")
        print(f"Agent 3: {_agent3_line(record['agent3'])}")
    if show_summary:
        print()
        print(f"Summary: PASS={summary['PASS']} REVIEW={summary['REVIEW']} "
              f"FAIL={summary['FAIL']}")
    if show_hints and warnings:
        sys.stdout.flush()
        print(file=sys.stderr)
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)


def _summarize(results):
    summary = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
    for record in results:
        verdict = record["agent3"].get("verdict")
        summary[verdict if verdict in summary else "REVIEW"] += 1
    return summary


def _exit_code(summary):
    if summary["FAIL"]:
        return 1
    if summary["REVIEW"]:
        return 2
    return 0


def _validate(args):
    if args.metrics and args.metrics_dir:
        raise _UsageError("use either --metrics or --metrics-dir, not both")
    if args.metrics and len(args.code) > 1:
        raise _UsageError("--metrics takes a single file; use --metrics-dir for batch mode")
    if "-" in args.code and len(args.code) > 1:
        raise _UsageError("stdin ('-') can only be used with a single --code")
    if args.metrics_dir and not Path(args.metrics_dir).is_dir():
        raise _UsageError(f"metrics directory not found: {args.metrics_dir}")


def main(argv=None):
    parser = _Parser(description="Run Agent 1 -> Agent 2 -> Agent 3 on source files.")
    parser.add_argument("--code", nargs="+", required=True, metavar="PATH",
                        help="source file(s) to audit; use '-' for stdin")
    parser.add_argument("--metrics", metavar="JSON",
                        help="Agent 1 metrics JSON (single --code only)")
    parser.add_argument("--metrics-dir", metavar="DIR",
                        help="batch mode: pair foo.ext with DIR/foo.metrics.json")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--no-hints", action="store_true",
                        help="do not print warnings/hints to stderr")

    try:
        args = parser.parse_args(argv)
        _validate(args)

        results = []
        warnings = []
        missing_metrics = False
        for code_path in args.code:
            record, lacked = _run_one(code_path, args, warnings)
            results.append(record)
            missing_metrics = missing_metrics or lacked

        if missing_metrics:
            warnings.append(
                "Agent 1 needs metrics: pass --metrics FILE or --metrics-dir DIR. "
                "Required keys: " + ", ".join(REQUIRED_METRICS))

        summary = _summarize(results)
        if args.json:
            print(json.dumps({"results": results, "summary": summary,
                              "warnings": warnings}, indent=2))
        else:
            _print_human(results, summary, warnings,
                         show_summary=len(results) > 1,
                         show_hints=not args.no_hints)
        return _exit_code(summary)
    except _UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"error: unexpected failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
