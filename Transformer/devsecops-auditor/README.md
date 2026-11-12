# DevSecOps Auditor
## Agent 2 (Member 2) - Transformer / source-code vulnerability detection

- Dataset: CodeXGLUE Defect Detection (Devign) - C functions, 21,854 train / 2,732
  validation / 2,732 test, ~46% vulnerable
  (HF: `google/code_x_glue_cc_defect_detection`)
- Model: `huggingface/CodeBERTa-small-v1` (6-layer, code-pretrained) with the encoder
  **frozen**; only a small MLP head is trained on cached mean-pooled embeddings.
  Reason: CPU-only machine - fully fine-tuning a 12-layer BERT would take hours per epoch.
- Install: `pip install -r requirements.txt`
- One-time download (dataset + frozen encoder, ~350 MB, resumable):
  `python agents/agent2_code.py prepare --out data/codexglue`
- Train (embeddings are cached in `data/cache/` so re-runs are fast):
  `python agents/agent2_code.py train --data data/codexglue`
- Usage: `from agents.agent2_code import predict_code_risk`
  Returns `code_risk_score`, `high_risk`, `threshold`, `status`.
  `status == "error"` -> Agent 3 must route to REVIEW (fail-safe), never PASS.
- Tests: `python test_agent2.py` (tiny local model, temp dirs, no real model touched)
- Optional external validation (zero-shot transfer, ~10 min on CPU):
  `python agents/agent2_code.py eval-external --n 5000`
- Limitations: 256-token truncation; CodeBERTa was pretrained on 6 languages (C is not
  among them) - this is the same setup as the CodeXGLUE benchmark, reported ~62 F1 for
  CodeBERT-base; Devign's official splits are known to contain cross-split duplicates,
  and metrics should be read with that in mind.

## Results (CodeXGLUE test split, 2,731 functions)

| Metric | Value |
|---|---|
| ROC-AUC | 0.668 |
| PR-AUC | 0.657 |
| Precision / Recall / F1 (vulnerable) @ 0.5 | 0.524 / 0.797 / 0.632 |
| F2-optimal threshold (validation) | 0.216 -> precision 0.477, recall 0.986 |
| Majority baseline | recall(vulnerable) 0.0 |
| TF-IDF + LogReg baseline | ROC-AUC 0.679, PR-AUC 0.629 |

The F2 threshold deliberately favours recall (missing a vulnerability is worse than a
false alarm), so Agent 3's decision tree combines this score with Agent 1's structural
score instead of relying on it alone. See `results_agent2.txt` for the full run.

### External validation (DiverseVul, zero-shot)

Evaluated on a stratified 5,000-function sample of the DiverseVul test split (50/50
balanced for metric symmetry; the raw split is only 5.73% vulnerable):

| Metric | Value |
|---|---|
| ROC-AUC | 0.549 |
| PR-AUC | 0.528 |

**Finding: no meaningful zero-shot transfer.** The Devign-trained head scores almost every
DiverseVul function as vulnerable (recall ~99%, precision ~50% at both thresholds), so the
ranking is near chance (ROC-AUC ~0.55). Devign covers 4 large C projects labelled from
fix commits, while DiverseVul spans many projects, CWEs and a very different base rate -
a textbook domain shift. Consequence for the report: the code-risk score is only
meaningful on its training distribution; Agent 3 must route out-of-distribution code to
REVIEW, or the head must be retrained/calibrated on target data. See
`results_agent2_diversevul.txt`.
