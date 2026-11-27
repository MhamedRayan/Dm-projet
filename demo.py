"""
End-to-end demo: Agent 1 -> Agent 2 -> Agent 3.
Run from the project root:
    python demo.py
"""
from agents.agent1_metrics import predict_struct_risk
from agents.agent2_transformer import predict_vuln
from agents.agent3_verdict import decide

# NASA/PROMISE PC1-style metrics for one function (matches models/agent1_rf.meta.json).
METRICS = {
    "loc": 200, "v(g)": 25, "ev(g)": 5, "iv(G)": 8, "n": 600, "v": 2500.0,
    "l": 30, "d": 5, "i": 10, "e": 80000.0, "b": 3, "t": 20,
    "lOCode": 150, "lOComment": 20, "locCodeAndComment": 0, "lOBlank": 30,
    "uniq_Op": 25, "uniq_Opnd": 30, "total_Op": 300, "total_Opnd": 120,
    "branchCount": 50,
}

CODE = '''
def read_user_input():
    import os
    cmd = input("command: ")
    os.system(cmd)
'''

# Explicit agent outputs so every verdict is visible regardless of how the
# models happen to score this sample.
SAFE_A1 = {"struct_risk_score": 0.05, "high_risk": False, "threshold": 0.3183,
           "status": "ok", "reason": ""}
SAFE_A2 = {"transformer_risk_score": 0.05, "vulnerable": False, "threshold": 0.51,
           "status": "ok", "reason": ""}
RISKY_A1 = {"struct_risk_score": 0.90, "high_risk": True, "threshold": 0.3183,
            "status": "ok", "reason": ""}
RISKY_A2 = {"transformer_risk_score": 0.90, "vulnerable": True, "threshold": 0.51,
            "status": "ok", "reason": ""}


def show(label, a1, a2):
    print(f"{label:20s} -> {decide(a1, a2)}")


def main():
    a1 = predict_struct_risk(METRICS)
    a2 = predict_vuln(CODE)
    verdict = decide(a1, a2)

    print("Agent 1 (metrics)     ->", a1)
    print("Agent 2 (transformer) ->", a2)
    print("Agent 3 (verdict)     ->", verdict)

    print("\nGatekeeper cases:")
    show("PASS (both low)", SAFE_A1, SAFE_A2)
    show("REVIEW (disagree)", RISKY_A1, SAFE_A2)
    show("FAIL (both high)", RISKY_A1, RISKY_A2)
    show("REVIEW (agent error)", RISKY_A1,
         {**RISKY_A2, "status": "error", "reason": "model unavailable"})


if __name__ == "__main__":
    main()
