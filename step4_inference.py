"""
STEP 4 — step4_inference.py
============================
Run predictions on any company's complaint dataset.
Works for ANY sector and product — even ones never in your training data.

Usage:
    # Batch — company provides a CSV:
    python step4_inference.py --input company_data.csv --output results.csv

    # Single complaint test:
    python step4_inference.py \
        --sector "Healthcare" \
        --product "Apollo Hospital App" \
        --complaint "My prescription record was deleted and pharmacy won't give medicine."

Required columns in company CSV:
    Sector | Product | Complaint_Narrative

Optional columns (ignored but preserved in output):
    customer_id, complaint_id, date, etc.
"""

import argparse
import pandas as pd
from model_core import Config, ComplaintAnalyzer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     default="best_model")
    p.add_argument("--input",     default=None,  help="Path to company CSV")
    p.add_argument("--output",    default="prioritized_complaints.csv")
    p.add_argument("--sector",    default=None)
    p.add_argument("--product",   default=None)
    p.add_argument("--complaint", default=None)
    return p.parse_args()


def main():
    args     = parse_args()
    analyzer = ComplaintAnalyzer(args.model, Config())

    # ── Single complaint mode ─────────────────────────────────────────────────
    if args.sector and args.product and args.complaint:
        r = analyzer.predict_single(args.sector, args.product, args.complaint)
        print("\n" + "=" * 60)
        print("COMPLAINT ANALYSIS")
        print("=" * 60)
        print(f"  Sector        : {args.sector}")
        print(f"  Product       : {args.product}")
        print(f"  Priority      : {r['priority_label']}")
        print(f"  Urgency Score : {r['urgency_score']} / 10")
        print(f"  Confidence    : {r['confidence']:.1%}")
        print("=" * 60)
        return

    # ── Batch mode ────────────────────────────────────────────────────────────
    if not args.input:
        print("Provide --input CSV or (--sector + --product + --complaint)")
        return

    df = pd.read_csv(args.input)
    for col in ["Sector", "Product", "Complaint_Narrative"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}' in {args.input}")

    print(f"Processing {len(df)} complaints...")
    results = analyzer.predict_batch(df)

    results.to_csv(args.output, index=False)

    print(f"\n✅ Results saved to: {args.output}")
    print(f"\nTop 10 most urgent complaints:")
# NEW
    print(results[["Sector", "Product", "Complaint_Narrative",
                   "priority_label", "urgency_score"]]
        .rename(columns={"priority_label": "Priority",
                            "urgency_score":  "Urgency"})
        .head(10).to_string(max_colwidth=55, index=False))

    print(f"\nPriority breakdown:")
    print(results["priority_label"].value_counts())
    print(f"\nMean urgency score: {results['urgency_score'].mean():.2f} / 10")


if __name__ == "__main__":
    main()
