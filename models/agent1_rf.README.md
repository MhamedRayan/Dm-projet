---
license: other
library_name: scikit-learn
pipeline_tag: tabular-classification
tags:
  - cybersecurity
  - vulnerability-detection
  - defect-prediction
  - random-forest
  - scikit-learn
  - joblib
  - nasa-promise
---

# Agent 1 — Structural Risk Random Forest

Random Forest that estimates defect risk from 21 NASA/PROMISE code metrics.
Part of a three-agent DevSecOps auditor (metrics → transformer → gatekeeper).

> For a Hugging Face repo, upload this file as `README.md` alongside
> `agent1_rf.joblib` and `agent1_rf.meta.json`.

## Files

| File | Contents |
|---|---|
| `agent1_rf.joblib` | `model`, `raw_cols`, `feature_cols`, `threshold` |
| `agent1_rf.meta.json` | Dataset, seed, threshold, best params |

## Input

A row of 21 numeric metrics:

```
loc, vg, evg, ivg, n, v, l, d, i, e, b, t, locode, locomment,
loccodeandcomment, loblank, uniq_op, uniq_opnd, total_op, total_opnd, branchcount
```

Five ratios are derived at inference time (already expected by `feature_cols`):

```
complexity_density = vg / (loc + 1)
effort_per_loc     = e / (loc + 1)
comment_ratio      = locomment / (loc + 1)
branch_density     = branchcount / (loc + 1)
operand_reuse      = total_opnd / (uniq_opnd + 1)
```

## Output

`predict_proba(X)[:, 1]` — defect risk in `[0, 1]`, uncalibrated.
Decision threshold: **0.3183** (tuned on training CV with F2, recall-weighted).

## Usage

```python
import joblib
import pandas as pd

b = joblib.load("agent1_rf.joblib")
df = pd.DataFrame([metrics])[b["raw_cols"]].apply(pd.to_numeric)

df["complexity_density"] = df["vg"] / (df["loc"] + 1)
df["effort_per_loc"]     = df["e"] / (df["loc"] + 1)
df["comment_ratio"]      = df["locomment"] / (df["loc"] + 1)
df["branch_density"]     = df["branchcount"] / (df["loc"] + 1)
df["operand_reuse"]      = df["total_opnd"] / (df["uniq_opnd"] + 1)

X = df[b["feature_cols"]]
score = b["model"].predict_proba(X)[0, 1]
high_risk = score >= b["threshold"]
```

Trained on `data/pc1.csv` (NASA PROMISE). Metrics from other languages or
domains are out-of-distribution.

## License

Not specified. Update the `license` field before publishing.
