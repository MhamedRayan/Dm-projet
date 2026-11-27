# DevSecOps Auditor
 
## Agent 1 (Member 1)
- Dataset: NASA PROMISE (PC1, KC1) from https://github.com/ApoorvaKrisna/NASA-promise-dataset-repository
- Put pc1.csv in data/ then run from the project root:
  python agents/agent1_metrics.py train --data data/pc1.csv
- Usage: from agents.agent1_metrics import predict_struct_risk
  Returns struct_risk_score, high_risk, threshold, status. status == "error" -> REVIEW.


## Layout
```
.
├── agents/
│   ├── __init__.py
│   ├── agent1_metrics.py        # Agent 1: Random Forest on code metrics
│   ├── agent2_transformer.py    # Agent 2: CodeBERT wrapper
│   └── agent3_verdict.py        # Agent 3: Decision Tree gatekeeper
├── models/
│   ├── agent1_rf.joblib
│   ├── agent1_rf.meta.json
│   ├── agent2_transformer/      # Fine-tuned CodeBERT files
│   ├── agent3_dt.joblib
│   └── agent3_dt.meta.json
├── demo.py
├── infer.py                     # CLI: run Agent 1 -> 2 -> 3 on files
├── webapp.py                    # Local web UI server
├── web/                         # Single-page frontend (HTML/CSS/JS)
├── test_agent1.py
├── test_agent2.py
├── test_agent3.py
├── test_infer.py
├── examples/                    # sample code + metrics JSON
├── requirements.txt
├── results_pc1.txt
└── results_pc1_v2.txt
```

Run everything from the project root.

## Setup
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
For a CPU-only PyTorch wheel (smaller download):
`.venv/bin/pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`

## Web UI
```
.venv/bin/python webapp.py
```
Open http://127.0.0.1:8000. Paste code or upload a source file, optionally
paste/upload the 21 metrics JSON, and click **Analyze code**. Without metrics,
Agent 1 is skipped and the verdict is `REVIEW`.

Options: `--host 0.0.0.0 --port 8080`. No extra dependencies; the UI is served
by Python's standard library. Uploaded code is analyzed, never executed.

## Inference CLI
Run the full chain on one file:
```
python infer.py --code examples/vulnerable.py \
  --metrics examples/vulnerable.metrics.json
```
Batch mode pairs each `foo.ext` with `DIR/foo.metrics.json`:
```
python infer.py --code examples/*.py --metrics-dir examples/
```
Code-only mode (Agent 1 cannot run, so the verdict is REVIEW):
```
python infer.py --code examples/vulnerable.py
```
Machine-readable output:
```
python infer.py --code examples/vulnerable.py \
  --metrics examples/vulnerable.metrics.json --json
```
Exit codes: `0` all PASS, `1` any FAIL, `2` any REVIEW/error, `3` usage error.

`--metrics` expects a flat JSON object with these keys (see
`examples/metrics_template.json`):
```
loc, vg, evg, ivg, n, v, l, d, i, e, b, t, locode, locomment,
loccodeandcomment, loblank, uniq_op, uniq_opnd, total_op, total_opnd, branchcount
```
`v(g)` / `iv(G)`-style aliases also work because Agent 1 normalizes them.
These are NASA/PROMISE metrics, not generic source metrics. If you only have
source code, omit `--metrics`; Agent 2 still runs and the verdict is REVIEW.

With the current model, `examples/safe.py` returns PASS and
`examples/vulnerable.py` returns FAIL.

## Agent 1 (Member 1)
- Dataset: NASA PROMISE (PC1, KC1) from https://github.com/ApoorvaKrisna/NASA-promise-dataset-repository
- Put pc1.csv in data/ then run:
  `python agents/agent1_metrics.py train --data data/pc1.csv`
- Usage: `from agents.agent1_metrics import predict_struct_risk`
  Returns struct_risk_score, high_risk, threshold, status. status == "error" -> REVIEW.
- Test: `python test_agent1.py`

## Agent 2 (Transformer)
- Wraps `models/agent2_transformer/` (CodeBERT, labels: safe/vulnerable, threshold
  from `meta.json`, max length 384).
- Usage: `from agents.agent2_transformer import predict_vuln`
  Returns transformer_risk_score, vulnerable, threshold, status, reason.
  status == "error" -> REVIEW.
- CLI: `python agents/agent2_transformer.py --file path/to/source.py`
  or `--code "os.system(cmd)"`.
- Test: `python test_agent2.py` (loads the ~500 MB model; CPU is fine).
- The model is loaded lazily and cached; missing deps/model/input fails safe.
- Model notes: CodeBERT fine-tuned on CodeXGLUE defect detection + DiverseVul
  (C/C++ vulnerability data). `meta.json` reports F1 0.6881, precision 0.5804,
  recall 0.845, ROC-AUC 0.7475; the wrapper applies the tuned threshold (0.45)
  faithfully, but the scores are not calibrated. Non-C/C++ code may be
  out-of-distribution.

## Agent 3 (Verdict Gatekeeper)
- Decision Tree combining Agent 1 and Agent 2 outputs.
- Policy (consensus + fail-safe):

  | Agent 1 | Agent 2 | Verdict |
  |---|---|---|
  | low risk | not vulnerable | `PASS` |
  | high risk | vulnerable | `FAIL` |
  | high risk | not vulnerable | `REVIEW` |
  | low risk | vulnerable | `REVIEW` |
  | any error / missing / invalid input | | `REVIEW` |

- Usage: `from agents.agent3_verdict import decide`
  `decide(agent1_result, agent2_result)` returns
  `{"verdict", "confidence", "status", "reason"}`.
  `status == "error"` always comes with `verdict == "REVIEW"`.
  `confidence` is the tree's class probability, not a calibrated probability.
- Train: `python agents/agent3_verdict.py train`
  The tree is trained on a synthetic grid labelled by the policy above, so it
  encodes the policy rather than learning from ground-truth verdicts. A
  post-check forces REVIEW if the tree ever contradicts the policy.
- Test: `python test_agent3.py`
- Demo: `python demo.py` (shows PASS, REVIEW, and FAIL cases)

