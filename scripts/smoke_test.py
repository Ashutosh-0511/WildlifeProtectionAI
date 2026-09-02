from pathlib import Path

from ml.risk.engine import RiskInput, score_risk

print("Risk engine:", score_risk(RiskInput(
    species="elephant", behaviour="RUNNING", human_present=True,
    distance_m=20, persistence_s=12, confidence=0.92,
)))

print("Repository smoke test passed:", Path("configs/pipeline.yaml").exists())
