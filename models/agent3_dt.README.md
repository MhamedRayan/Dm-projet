---
license: other
library_name: scikit-learn
pipeline_tag: tabular-classification
tags:
  - cybersecurity
  - gatekeeper
  - decision-tree
  - scikit-learn
  - joblib
---

# Agent 3 — Verdict Gatekeeper (Decision Tree)

Decision tree that turns two agent scores into a final verdict using a
consensus + fail-safe policy. Part of a three-agent DevSecOps auditor
(metrics → transformer → gatekeeper).

> For a Hugging Face repo, upload this file as `README.md` alongside
> `agent3_dt.joblib` and `agent3_dt.meta.json`.

## Files

| File | Contents |
|---|---|
| `agent3_dt.joblib` | `model`, `feature_cols`, `classes`, `policy`, `source_thresholds` |
| `agent3_dt.meta.json` | Policy, features, grid size, agreement, thresholds |

## Input

Four features per sample:

```
a1_high_risk   # 1.0 if agent1_score   >= agent1_threshold
a2_vulnerable  # 1.0 if agent2_score   >= agent2_threshold
a1_risk_ratio  # agent1_score / agent1_threshold
a2_risk_ratio  # agent2_score / agent2_threshold
```

## Output

One of `PASS`, `REVIEW`, `FAIL`, with the tree's class probability as
`confidence`. That confidence is **not** calibrated.

## Policy

| Agent 1 | Agent 2 | Verdict |
|---|---|---|
| low risk | not vulnerable | `PASS` |
| high risk | vulnerable | `FAIL` |
| high risk | not vulnerable | `REVIEW` |
| low risk | vulnerable | `REVIEW` |
| any error / missing / invalid input | | `REVIEW` |

The tree is trained on a synthetic grid labelled by this policy, so it encodes
the policy rather than learning from ground truth. The reference wrapper adds a
post-check that forces `REVIEW` on any tree/policy mismatch — keep it if you
deploy this model.

## Usage

```python
import joblib

b = joblib.load("agent3_dt.joblib")

a1_high = a1_score >= a1_threshold
a2_high = a2_score >= a2_threshold
x = [[float(a1_high), float(a2_high),
      a1_score / a1_threshold, a2_score / a2_threshold]]

proba = b["model"].predict_proba(x)[0]
verdict = b["classes"][int(proba.argmax())]
confidence = float(proba.max())
```

## License

Not specified. Update the `license` field before publishing.
