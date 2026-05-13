"""
STEP 2 — step2_train.py
=======================
Trains the new generalised model on your prepared data splits.

Run:
    python step2_train.py

    # Faster (less RAM, nearly same accuracy):
    python step2_train.py --model distilbert-base-uncased

    # More epochs:
    python step2_train.py --epochs 8

Output:
    best_model/model.pt            ← best checkpoint (by val loss)
    best_model/tokenizer files     ← saved alongside for portability
"""

import argparse
import os
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model_core import Config, ComplaintDataset, ComplaintPrioritizer, Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train",   default="data/train_new.csv")
    p.add_argument("--val",     default="data/val_new.csv")
    p.add_argument("--output",  default="best_model")
    p.add_argument("--model",   default="bert-base-uncased",
                   help="HuggingFace model name. "
                        "Use 'distilbert-base-uncased' for 2x faster training.")
    p.add_argument("--epochs",  type=int,   default=5)
    p.add_argument("--batch",   type=int,   default=32)
    p.add_argument("--lr",      type=float, default=2e-5)
    return p.parse_args()


def main():
    args          = parse_args()
    config        = Config()
    config.BASE_MODEL = args.model
    config.EPOCHS     = args.epochs
    config.BATCH_SIZE = args.batch
    config.LR         = args.lr

    print("=" * 60)
    print(f"Device     : {config.DEVICE}")
    print(f"Base model : {config.BASE_MODEL}")
    print(f"Epochs     : {config.EPOCHS}")
    print(f"Batch size : {config.BATCH_SIZE}")
    print(f"LR         : {config.LR}")
    print("=" * 60)

    train_df = pd.read_csv(args.train)
    val_df   = pd.read_csv(args.val)
    print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows")

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)

    train_ds     = ComplaintDataset(train_df, tokenizer, config)
    val_ds       = ComplaintDataset(val_df,   tokenizer, config)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=2)

    model   = ComplaintPrioritizer(config)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_p:,}")

    trainer = Trainer(model, config)
    trainer.train(train_loader, val_loader, save_dir=args.output)

    # Save tokenizer alongside model weights
    tokenizer.save_pretrained(args.output)
    print(f"\n Training complete. Model saved to: {args.output}/")

    # ── Quick sanity check ────────────────────────────────────────────────────
    print("\nSanity check on unseen sectors (generalization test):")
    from model_core import ComplaintAnalyzer
    analyzer = ComplaintAnalyzer(args.output, config)

    test_cases = [
        # Sectors from your training data
        ("Aviation",          "IndiGo",           "Flight was delayed by 5 hours with no explanation given."),
        ("Food & Beverage",   "Zomato Food Delivery", "Wrong order delivered and app shows delivered."),
        # Sectors NEVER seen in training ← this is what your old model failed on
        ("Healthcare",        "Apollo Hospital App",  "My prescription was lost and pharmacy says no record."),
        ("E-commerce",        "Flipkart Electronics", "Received a broken laptop, return request denied."),
        ("Automobile",        "Maruti Service",       "Car returned after service with new scratches on body."),
        ("Real Estate",       "MagicBricks Portal",   "Builder not responding after full payment received."),
    ]

    print(f"\n{'Sector':<28} {'Product':<28} {'Priority':<10} {'Urgency':>8}")
    print("-" * 80)
    for sector, product, complaint in test_cases:
        r = analyzer.predict_single(sector, product, complaint)
        marker = "" if sector in ["Aviation","Food & Beverage","Banking","Retail","Tech & Telecommunications"] else "← NEW"
        print(f"{sector:<28} {product:<28} {r['priority_label']:<10} {r['urgency_score']:>8.1f}  {marker}")

    print()
    print("➡  Next: run  python step3_evaluate.py")


if __name__ == "__main__":
    main()
