"""Merge my reviewed tiers into the labeling sheet -> gold_labels.csv"""
import pandas as pd
from pathlib import Path

GOLD = Path(__file__).resolve().parent.parent / "gold"
sheet = pd.read_csv(GOLD / "to_label.csv")
mine = pd.read_csv(GOLD / "my_tiers.csv")

merged = sheet.merge(mine[["candidate_id", "final_tier", "reason"]],
                     on="candidate_id", how="left", suffixes=("_old", ""))
merged["final_tier"] = merged["final_tier"].astype(int)
merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_old")])
merged.to_csv(GOLD / "gold_labels.csv", index=False)

print(f"Wrote {len(merged)} rows -> gold_labels.csv")
print("\nFinal tier distribution:")
print(merged["final_tier"].value_counts().sort_index())
print(f"\nRelevant (tier>=3): {(merged.final_tier>=3).sum()}   "
      f"Top (tier 5): {(merged.final_tier==5).sum()}")
print("\nDisagreements with my auto-heuristic (auto_tier vs final_tier):",
      (merged.auto_tier != merged.final_tier).sum())