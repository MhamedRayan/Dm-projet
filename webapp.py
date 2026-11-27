#!/usr/bin/env python3
"""Local web UI for the DevSecOps auditor.

Serves a single-page frontend and one JSON endpoint that runs
Agent 1 -> Agent 2 -> Agent 3.

Run from anywhere:
    .venv/bin/python webapp.py
    .venv/bin/python webapp.py --host 0.0.0.0 --port 8080

Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
MAX_CODE_BYTES = 2 * 1024 * 1024
MAX_METRICS_BYTES = 512 * 1024
MAX_BODY_BYTES = 4 * 1024 * 1024

_STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}

_agents = None
_analyze_lock = threading.Lock()


def _load_agents():
    global _agents
    if _agents is None:
        from agents.agent1_metrics import predict_struct_risk
        from agents.agent2_transformer import predict_vuln
        from agents.agent3_verdict import decide

        _agents = {
            "agent1": predict_struct_risk,
            "agent2": predict_vuln,
            "agent3": decide,
        }
    return _agents


def analyze(code: str, metrics: dict | None) -> dict:
    """Run the three agents. Serialized so only one inference runs at a time."""
    with _analyze_lock:
        agents = _load_agents()
        started = time.perf_counter()

        if metrics is None:
            agent1 = {
                "struct_risk_score": None,
                "high_risk": None,
                "status": "error",
                "reason": "no metrics provided (Agent 1 skipped)",
            }
        else:
            agent1 = agents["agent1"](metrics)

        agent2 = agents["agent2"](code)
        agent3 = agents["agent3"](agent1, agent2)

        return {
            "agent1": agent1,
            "agent2": agent2,
            "agent3": agent3,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "DevSecOpsAuditor/1.0"

    def _send_bytes(self, status, body, content_type):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        entry = _STATIC.get(path)
        if entry is None:
            self._send_json(404, {"error": "not found"})
            return

        name, content_type = entry
        file_path = WEB_ROOT / name
        if not file_path.is_file():
            self._send_json(404, {"error": f"missing static file: {name}"})
            return
        self._send_bytes(200, file_path.read_bytes(), content_type)

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/analyze":
            self._send_json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return

        if length <= 0:
            self._send_json(400, {"error": "empty request body"})
            return
        if length > MAX_BODY_BYTES:
            self._send_json(413, {"error": "request too large"})
            return

        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_json(400, {"error": f"invalid JSON: {exc}"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "request body must be a JSON object"})
            return

        code = payload.get("code")
        if not isinstance(code, str) or not code.strip():
            self._send_json(400, {"error": "code must be a non-empty string"})
            return
        if len(code.encode("utf-8")) > MAX_CODE_BYTES:
            self._send_json(413, {"error": "code is too large (max 2 MB)"})
            return

        metrics = payload.get("metrics")
        if metrics is not None:
            if not isinstance(metrics, dict):
                self._send_json(400, {"error": "metrics must be a JSON object"})
                return
            if len(json.dumps(metrics).encode("utf-8")) > MAX_METRICS_BYTES:
                self._send_json(413, {"error": "metrics are too large (max 512 KB)"})
                return

        try:
            result = analyze(code, metrics)
        except Exception as exc:  # missing model/deps, unexpected failure
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._send_json(200, result)

    def log_message(self, fmt, *args):
        # Keep the console quiet unless the response is a client/server error.
        if len(args) > 1 and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description="Serve the DevSecOps Auditor web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.chdir(ROOT)  # model paths are relative to the project root
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"DevSecOps Auditor UI -> http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
