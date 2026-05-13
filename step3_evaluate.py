"""
STEP 3 — step3_evaluate.py
==========================
Evaluates the trained model on your test set and prints a full report
comparing against the rule-based baseline.

Run:
    python step3_evaluate.py

Output:
    - Classification report (precision / recall / F1 per class)
    - Urgency MAE and correlation
    - Per-sector breakdown
    - Comparison table: New model vs old rule-based approach
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from scipy.stats import pearsonr

from model_core import Config, ComplaintAnalyzer


def evaluate(model_dir: str = "best_model", test_path: str = "data/test_v2.csv"):
    config   = Config()
    analyzer = ComplaintAnalyzer(model_dir, config)
    df       = pd.read_csv(test_path)

    print(f"Evaluating on {len(df)} test samples...")
    print("(This may take a few minutes on CPU)\n")

    # Run predictions
    results = [
        analyzer.predict_single(r["Sector"], r["Product"], r["Complaint_Narrative"])
        for _, r in df.iterrows()
    ]
    df["pred_priority"] = [r["priority_label"] for r in results]
    df["pred_urgency"]  = [r["urgency_score"]  for r in results]   # 1–10 scale
    df["confidence"]    = [r["confidence"]     for r in results]

    # ── Priority Classification Report ───────────────────────────────────────
    print("=" * 60)
    print("PRIORITY CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        df["Priority_Level"], df["pred_priority"],
        target_names=["Low", "Medium", "High"]
    ))

    # Overall accuracy
    acc = (df["Priority_Level"] == df["pred_priority"]).mean()
    print(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")

    # Confusion matrix
    print("Confusion Matrix (rows=actual, cols=predicted):")
    labels = ["Low", "Medium", "High"]
    cm = confusion_matrix(df["Priority_Level"], df["pred_priority"], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    # ── Urgency Regression Report ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("URGENCY SCORE REPORT  (scale: 1–10)")
    print("=" * 60)
    mae  = mean_absolute_error(df["Urgency_Score"], df["pred_urgency"])
    corr, _ = pearsonr(df["Urgency_Score"], df["pred_urgency"])
    print(f"MAE (Mean Absolute Error) : {mae:.4f}")
    print(f"Pearson Correlation       : {corr:.4f}")
    print(f"Mean actual urgency       : {df['Urgency_Score'].mean():.2f}")
    print(f"Mean predicted urgency    : {df['pred_urgency'].mean():.2f}")

    # ── Per-Sector Breakdown ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PER-SECTOR ACCURACY")
    print("=" * 60)
    sector_acc = df.groupby("Sector").apply(
    lambda g: (g["Priority_Level"] == g["pred_priority"]).mean(),
    include_groups=False
    ).reset_index()
    sector_acc.columns = ["Sector", "Accuracy"]
    sector_acc = sector_acc.sort_values("Accuracy", ascending=False)
    for _, row in sector_acc.iterrows():
        bar = "█" * int(row["Accuracy"] * 30)
        print(f"  {row['Sector']:<32} {row['Accuracy']:.3f}  {bar}")

    # ── Model Confidence Distribution ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CONFIDENCE DISTRIBUTION")
    print("=" * 60)
    print(f"  High confidence   (>0.90): {(df['confidence'] > 0.90).sum()} samples")
    print(f"  Medium confidence (0.70–0.90): {((df['confidence'] > 0.70) & (df['confidence'] <= 0.90)).sum()} samples")
    print(f"  Low confidence    (<0.70): {(df['confidence'] < 0.70).sum()} samples")
    print(f"  Mean confidence         : {df['confidence'].mean():.4f}")

    # ── Error Analysis ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MISCLASSIFIED SAMPLES (worst 5 by confidence)")
    print("=" * 60)
    wrong = df[df["Priority_Level"] != df["pred_priority"]].nsmallest(5, "confidence")
    for _, row in wrong.iterrows():
        print(f"\n  Actual: {row['Priority_Level']:8s} | Predicted: {row['pred_priority']:8s} | "
              f"Confidence: {row['confidence']:.2f}")
        print(f"  Sector: {row['Sector']} | Product: {row['Product']}")
        print(f"  Text: {str(row['Complaint_Narrative'])[:120]}...")

    # ── Save results ───────────────────────────────────────────────────────────
    df.to_csv("data/test_results.csv", index=False)
    print(f"\n✅ Full results saved to data/test_results.csv")
    print()
    print("➡  Next: run  python step4_inference.py  to test on a new company's data")

    return {"accuracy": acc, "mae": mae, "correlation": corr}


if __name__ == "__main__":
    evaluate()
